//! Explicit, integrity-checked lifecycle for the curated local OCR model bundle.

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    fs::OpenOptions,
    io,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use fs2::FileExt;
use futures_util::{
    StreamExt,
    future::{Either, select},
};
use impossible_ocr_domain::{OcrError, OcrErrorCode};
use impossible_ocr_pipeline::CtcDecoder;
use impossible_server_core::CancellationToken;
use reqwest::{Client, StatusCode, header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    fs::{self, File},
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    time::{Instant, sleep, timeout_at},
};
use url::Url;

const OWNER: &str = ".impossible-ocr-owner.json";
const LEGACY_LEASE: &str = ".lease";
const MAX_YAML: u64 = 64 * 1024;
/// Stable identifier of the only curated detector/recognizer bundle.
pub const CURATED_BUNDLE_ID: &str = "paddlex-ocr-3.7-max960";
const BUNDLE: &str = CURATED_BUNDLE_ID;
const DET_REV: &str = "e6f4fa85f00e168c862bc462aebca69eef9b3d3d";
const REC_REV: &str = "3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b";
const HOSTS: &[&str] = &[
    "huggingface.co",
    "cdn-lfs.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cas-bridge.xethub.hf.co",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Detector,
    EnglishRecognizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    OnnxGraph,
    InferenceConfig,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sha256Digest(String);
impl Sha256Digest {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelStoreError> {
        let value = value.into();
        if value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            Ok(Self(value))
        } else {
            Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest))
        }
    }
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sha256Digest([REDACTED])")
    }
}

#[derive(Clone)]
pub struct ModelArtifact {
    pub role: ModelRole,
    pub kind: ArtifactKind,
    pub upstream_project: String,
    pub revision: String,
    pub filename: String,
    pub byte_length: u64,
    pub sha256: Sha256Digest,
    pub license: String,
    source_url: Url,
}
impl fmt::Debug for ModelArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelArtifact")
            .field("role", &self.role)
            .field("kind", &self.kind)
            .field("upstream_project", &self.upstream_project)
            .field("revision", &self.revision)
            .field("filename", &self.filename)
            .field("byte_length", &self.byte_length)
            .field("sha256", &self.sha256)
            .field("license", &self.license)
            .field("source_url", &"[REDACTED]")
            .finish()
    }
}
impl ModelArtifact {
    #[allow(clippy::too_many_arguments)]
    fn reviewed(
        role: ModelRole,
        kind: ArtifactKind,
        project: &str,
        revision: &str,
        filename: &str,
        size: u64,
        sha: &str,
        url: &str,
    ) -> Result<Self, ModelStoreError> {
        let value = Self {
            role,
            kind,
            upstream_project: project.into(),
            revision: revision.into(),
            filename: filename.into(),
            byte_length: size,
            sha256: Sha256Digest::new(sha)?,
            license: "Apache-2.0".into(),
            source_url: Url::parse(url)
                .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::InvalidManifest))?,
        };
        value.validate(false)?;
        Ok(value)
    }
    fn validate(&self, test_http: bool) -> Result<(), ModelStoreError> {
        let valid = safe_name(&self.filename)
            && !self.upstream_project.is_empty()
            && self.revision.len() == 40
            && self.revision.bytes().all(|b| b.is_ascii_hexdigit())
            && valid_url(&self.source_url, test_http)
            && self.source_url.path().contains(&self.revision)
            && self.byte_length > 0
            && (self.kind == ArtifactKind::OnnxGraph || self.byte_length <= MAX_YAML)
            && self.license == "Apache-2.0";
        if valid {
            Ok(())
        } else {
            Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest))
        }
    }
}

#[derive(Clone)]
pub struct ModelBundleManifest {
    pub bundle_id: String,
    pub schema_version: u32,
    pub artifacts: Vec<ModelArtifact>,
}
impl fmt::Debug for ModelBundleManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelBundleManifest")
            .field("bundle_id", &self.bundle_id)
            .field("schema_version", &self.schema_version)
            .field("artifacts", &self.artifacts)
            .finish()
    }
}
impl ModelBundleManifest {
    pub fn validate(&self) -> Result<(), ModelStoreError> {
        self.validate_policy(false)
    }
    fn validate_policy(&self, test_http: bool) -> Result<(), ModelStoreError> {
        if self.schema_version != 1 || !safe_name(&self.bundle_id) || self.artifacts.len() != 4 {
            return Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest));
        }
        let mut keys = BTreeSet::new();
        let mut names = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate(test_http)?;
            if !keys.insert((artifact.role, artifact.kind)) || !names.insert(&artifact.filename) {
                return Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest));
            }
        }
        for role in [ModelRole::Detector, ModelRole::EnglishRecognizer] {
            for kind in [ArtifactKind::OnnxGraph, ArtifactKind::InferenceConfig] {
                if !keys.contains(&(role, kind)) {
                    return Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest));
                }
            }
        }
        Ok(())
    }
    fn artifact(
        &self,
        role: ModelRole,
        kind: ArtifactKind,
    ) -> Result<&ModelArtifact, ModelStoreError> {
        self.artifacts
            .iter()
            .find(|a| a.role == role && a.kind == kind)
            .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::InvalidManifest))
    }
    fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"impossible-ocr-bundle-v1");
        h.update(self.bundle_id.as_bytes());
        for a in &self.artifacts {
            h.update([a.role as u8, a.kind as u8]);
            h.update(a.revision.as_bytes());
            h.update(a.filename.as_bytes());
            h.update(a.byte_length.to_be_bytes());
            h.update(a.sha256.as_str().as_bytes());
        }
        hex(&h.finalize())
    }
}

#[derive(Clone, Default)]
pub struct ModelCatalog {
    bundles: BTreeMap<String, Arc<ModelBundleManifest>>,
}
impl fmt::Debug for ModelCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelCatalog")
            .field("bundle_ids", &self.bundles.keys().collect::<Vec<_>>())
            .finish()
    }
}
impl ModelCatalog {
    pub fn curated() -> Result<Self, ModelStoreError> {
        let mut c = Self::default();
        let base_det = format!(
            "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/{DET_REV}"
        );
        let base_rec = format!(
            "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/resolve/{REC_REV}"
        );
        c.insert(ModelBundleManifest {
            bundle_id: BUNDLE.into(),
            schema_version: 1,
            artifacts: vec![
                ModelArtifact::reviewed(
                    ModelRole::Detector,
                    ArtifactKind::OnnxGraph,
                    "PaddlePaddle/PP-OCRv5_mobile_det_onnx",
                    DET_REV,
                    "detector.onnx",
                    4_826_518,
                    "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d",
                    &format!("{base_det}/inference.onnx"),
                )?,
                ModelArtifact::reviewed(
                    ModelRole::Detector,
                    ArtifactKind::InferenceConfig,
                    "PaddlePaddle/PP-OCRv5_mobile_det_onnx",
                    DET_REV,
                    "detector-inference.yml",
                    903,
                    "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
                    &format!("{base_det}/inference.yml"),
                )?,
                ModelArtifact::reviewed(
                    ModelRole::EnglishRecognizer,
                    ArtifactKind::OnnxGraph,
                    "PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx",
                    REC_REV,
                    "english-recognizer.onnx",
                    7_848_423,
                    "b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f",
                    &format!("{base_rec}/inference.onnx"),
                )?,
                ModelArtifact::reviewed(
                    ModelRole::EnglishRecognizer,
                    ArtifactKind::InferenceConfig,
                    "PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx",
                    REC_REV,
                    "english-recognizer-inference.yml",
                    3_964,
                    "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
                    &format!("{base_rec}/inference.yml"),
                )?,
            ],
        })?;
        Ok(c)
    }
    pub fn insert(&mut self, bundle: ModelBundleManifest) -> Result<(), ModelStoreError> {
        bundle.validate()?;
        self.insert_checked(bundle)
    }
    fn insert_checked(&mut self, bundle: ModelBundleManifest) -> Result<(), ModelStoreError> {
        if self.bundles.contains_key(&bundle.bundle_id) {
            return Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest));
        }
        self.bundles
            .insert(bundle.bundle_id.clone(), Arc::new(bundle));
        Ok(())
    }
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Arc<ModelBundleManifest>> {
        self.bundles.get(id).cloned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelStoreErrorCode {
    InvalidManifest,
    UnsafeFilesystem,
    UnapprovedSource,
    IntegrityMismatch,
    ConfigurationMismatch,
    Cancelled,
    TimedOut,
    InUse,
    NotInstalled,
    Io,
}
pub struct ModelStoreError {
    code: ModelStoreErrorCode,
    source: Option<Box<dyn Error + Send + Sync>>,
}
impl ModelStoreError {
    fn new(code: ModelStoreErrorCode) -> Self {
        Self { code, source: None }
    }
    fn source(code: ModelStoreErrorCode, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            code,
            source: Some(Box::new(source)),
        }
    }
    #[must_use]
    pub const fn code(&self) -> ModelStoreErrorCode {
        self.code
    }
    #[must_use]
    pub fn diagnostic_source(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.source.as_deref()
    }
}
impl fmt::Debug for ModelStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelStoreError")
            .field("code", &self.code)
            .field("source", &self.source.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}
impl fmt::Display for ModelStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.code {
            ModelStoreErrorCode::InvalidManifest => "model metadata is invalid",
            ModelStoreErrorCode::UnsafeFilesystem => "the model store is unsafe",
            ModelStoreErrorCode::UnapprovedSource => "the model source is not approved",
            ModelStoreErrorCode::IntegrityMismatch => "model integrity verification failed",
            ModelStoreErrorCode::ConfigurationMismatch => "model configuration verification failed",
            ModelStoreErrorCode::Cancelled => "the model operation was cancelled",
            ModelStoreErrorCode::TimedOut => "the model operation timed out",
            ModelStoreErrorCode::InUse => "the model bundle is in use",
            ModelStoreErrorCode::NotInstalled => "the model bundle is not installed",
            ModelStoreErrorCode::Io => "the model operation failed",
        })
    }
}
impl Error for ModelStoreError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStoreStatus {
    Missing,
    Verified,
    Corrupt,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallOutcome {
    Installed,
    Repaired,
    AlreadyVerified,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteOutcome {
    Deleted,
    NotPresent,
}
pub struct ImportBundle {
    pub detector_graph: PathBuf,
    pub detector_config: PathBuf,
    pub recognizer_graph: PathBuf,
    pub recognizer_config: PathBuf,
}
impl fmt::Debug for ImportBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ImportBundle([PATHS REDACTED])")
    }
}
impl ImportBundle {
    fn path(&self, r: ModelRole, k: ArtifactKind) -> &Path {
        match (r, k) {
            (ModelRole::Detector, ArtifactKind::OnnxGraph) => &self.detector_graph,
            (ModelRole::Detector, ArtifactKind::InferenceConfig) => &self.detector_config,
            (ModelRole::EnglishRecognizer, ArtifactKind::OnnxGraph) => &self.recognizer_graph,
            (ModelRole::EnglishRecognizer, ArtifactKind::InferenceConfig) => {
                &self.recognizer_config
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct VerifiedBundle {
    pub bundle_id: String,
    pub source_dictionary_sha256: Sha256Digest,
    pub recognizer_class_count: usize,
    pub profile_id: String,
    pub upstream_detector_resize_long: u32,
    pub runtime_detector_max_side: u32,
    recognizer_dictionary: Vec<String>,
}

impl VerifiedBundle {
    /// Builds the CTC decoder from the exact ordered dictionary admitted with this bundle.
    pub fn ctc_decoder(&self) -> Result<CtcDecoder, OcrError> {
        let digest = decode_sha256(self.source_dictionary_sha256.as_str())
            .ok_or_else(|| OcrError::for_code(OcrErrorCode::ModelUnavailable))?;
        let decoder = CtcDecoder::new_verified(self.recognizer_dictionary.clone(), true, digest)
            .map_err(|_| OcrError::for_code(OcrErrorCode::ModelUnavailable))?;
        if decoder.class_count() != self.recognizer_class_count {
            return Err(OcrError::for_code(OcrErrorCode::ModelUnavailable));
        }
        Ok(decoder)
    }
}

impl fmt::Debug for VerifiedBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedBundle")
            .field("bundle_id", &self.bundle_id)
            .field("source_dictionary_sha256", &self.source_dictionary_sha256)
            .field("recognizer_class_count", &self.recognizer_class_count)
            .field("profile_id", &self.profile_id)
            .field(
                "upstream_detector_resize_long",
                &self.upstream_detector_resize_long,
            )
            .field("runtime_detector_max_side", &self.runtime_detector_max_side)
            .field("recognizer_dictionary", &"[REDACTED]")
            .finish()
    }
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Marker {
    schema_version: u32,
    bundle_id: String,
    manifest_sha256: String,
    dictionary_sha256: String,
    recognizer_class_count: usize,
    profile_id: String,
    upstream_resize_long: u32,
    runtime_max_side: u32,
    yaml_preprocessing_adopted: bool,
}
#[derive(Clone)]
struct Policy {
    total: Duration,
    idle: Duration,
    redirects: usize,
    test_http: bool,
    #[cfg(test)]
    fail_after_quarantine: bool,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            total: Duration::from_secs(1800),
            idle: Duration::from_secs(30),
            redirects: 4,
            test_http: false,
            #[cfg(test)]
            fail_after_quarantine: false,
        }
    }
}

pub struct ModelStore {
    root: PathBuf,
    catalog: ModelCatalog,
    client: Client,
    policy: Policy,
    leases: Arc<Mutex<BTreeMap<String, usize>>>,
}
impl fmt::Debug for ModelStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelStore")
            .field("root", &"[REDACTED]")
            .field("catalog", &self.catalog)
            .finish_non_exhaustive()
    }
}
impl ModelStore {
    pub async fn open(
        root: impl AsRef<Path>,
        catalog: ModelCatalog,
    ) -> Result<Self, ModelStoreError> {
        Self::open_policy(root, catalog, Policy::default()).await
    }
    async fn open_policy(
        root: impl AsRef<Path>,
        catalog: ModelCatalog,
        policy: Policy,
    ) -> Result<Self, ModelStoreError> {
        no_parent(root.as_ref())?;
        no_link_ancestry(root.as_ref()).await?;
        fs::create_dir_all(root.as_ref()).await.map_err(ioe)?;
        no_link_ancestry(root.as_ref()).await?;
        let root = fs::canonicalize(root.as_ref()).await.map_err(ioe)?;
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| ModelStoreError::source(ModelStoreErrorCode::Io, e))?;
        Ok(Self {
            root,
            catalog,
            client,
            policy,
            leases: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }
    pub async fn install(
        &self,
        id: &str,
        cancel: &CancellationToken,
    ) -> Result<InstallOutcome, ModelStoreError> {
        let deadline = operation_deadline(self.policy.total)?;
        let manifest = self.manifest(id)?;
        let _lock = self.lock(id, cancel, deadline).await?;
        let _lease = self.exclusive_lease(id)?;
        if self
            .verify_manifest(&manifest, cancel, deadline)
            .await
            .is_ok()
        {
            return Ok(InstallOutcome::AlreadyVerified);
        }
        check_budget(cancel, deadline)?;
        let replacing = io_with_budget(fs::try_exists(self.bundle(id)), cancel, deadline).await?;
        if replacing {
            verify_ownership(&self.bundle(id), &manifest, cancel, deadline).await?;
        }
        let stage = self.prepare(id, cancel, deadline).await?;
        let result = async {
            for artifact in &manifest.artifacts {
                self.download(artifact, &stage.join(&artifact.filename), cancel, deadline)
                    .await?;
            }
            let verified = verify_dir(&stage, &manifest, false, cancel, deadline).await?;
            write_marker(&stage, &manifest, &verified, cancel, deadline).await?;
            self.promote(id, &stage, replacing, cancel, deadline)
                .await?;
            Ok(if replacing {
                InstallOutcome::Repaired
            } else {
                InstallOutcome::Installed
            })
        }
        .await;
        if result.is_err() {
            let _ = remove_safe(&stage, cancel, deadline).await;
        }
        result
    }
    pub async fn import(
        &self,
        id: &str,
        source: &ImportBundle,
        cancel: &CancellationToken,
    ) -> Result<InstallOutcome, ModelStoreError> {
        let deadline = operation_deadline(self.policy.total)?;
        let manifest = self.manifest(id)?;
        let _lock = self.lock(id, cancel, deadline).await?;
        let _lease = self.exclusive_lease(id)?;
        if self
            .verify_manifest(&manifest, cancel, deadline)
            .await
            .is_ok()
        {
            return Ok(InstallOutcome::AlreadyVerified);
        }
        check_budget(cancel, deadline)?;
        let replacing = io_with_budget(fs::try_exists(self.bundle(id)), cancel, deadline).await?;
        if replacing {
            verify_ownership(&self.bundle(id), &manifest, cancel, deadline).await?;
        }
        let stage = self.prepare(id, cancel, deadline).await?;
        let result = async {
            for artifact in &manifest.artifacts {
                let p = source.path(artifact.role, artifact.kind);
                no_parent(p)?;
                no_link_ancestry(p).await?;
                copy_verified(
                    io_with_budget(File::open(p), cancel, deadline).await?,
                    &stage.join(&artifact.filename),
                    artifact,
                    cancel,
                    deadline,
                )
                .await?;
                no_link_ancestry(p).await?;
            }
            let verified = verify_dir(&stage, &manifest, false, cancel, deadline).await?;
            write_marker(&stage, &manifest, &verified, cancel, deadline).await?;
            self.promote(id, &stage, replacing, cancel, deadline)
                .await?;
            Ok(if replacing {
                InstallOutcome::Repaired
            } else {
                InstallOutcome::Installed
            })
        }
        .await;
        if result.is_err() {
            let _ = remove_safe(&stage, cancel, deadline).await;
        }
        result
    }
    pub async fn status(&self, id: &str) -> ModelStoreStatus {
        let Ok(m) = self.manifest(id) else {
            return ModelStoreStatus::Missing;
        };
        if !fs::try_exists(self.bundle(id)).await.unwrap_or(false) {
            ModelStoreStatus::Missing
        } else if self
            .verify_manifest(
                &m,
                &CancellationToken::new(),
                operation_deadline(self.policy.total).unwrap_or_else(|_| Instant::now()),
            )
            .await
            .is_ok()
        {
            ModelStoreStatus::Verified
        } else {
            ModelStoreStatus::Corrupt
        }
    }
    pub async fn verify(&self, id: &str) -> Result<VerifiedBundle, ModelStoreError> {
        self.verify_with_cancellation(id, &CancellationToken::new())
            .await
    }
    pub async fn verify_with_cancellation(
        &self,
        id: &str,
        cancel: &CancellationToken,
    ) -> Result<VerifiedBundle, ModelStoreError> {
        let m = self.manifest(id)?;
        self.verify_manifest(&m, cancel, operation_deadline(self.policy.total)?)
            .await
    }
    pub async fn lease(&self, id: &str) -> Result<ModelLease, ModelStoreError> {
        let m = self.manifest(id)?;
        let file = lock_file(&self.lease_path(id))?;
        FileExt::try_lock_shared(&file).map_err(|error| {
            if lock_is_contended(&error) {
                ModelStoreError::new(ModelStoreErrorCode::InUse)
            } else {
                ioe(error)
            }
        })?;
        *self
            .leases
            .lock()
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::Io))?
            .entry(id.into())
            .or_default() += 1;
        let lease = ModelLease {
            id: id.into(),
            path: self.bundle(id),
            manifest: m,
            file: Some(file),
            counts: Arc::clone(&self.leases),
        };
        self.verify_manifest(
            &lease.manifest,
            &CancellationToken::new(),
            operation_deadline(self.policy.total)?,
        )
        .await?;
        Ok(lease)
    }
    pub async fn delete(&self, id: &str) -> Result<DeleteOutcome, ModelStoreError> {
        let deadline = operation_deadline(self.policy.total)?;
        let cancel = CancellationToken::new();
        self.manifest(id)?;
        let _lock = self.lock(id, &cancel, deadline).await?;
        let _lease = self.exclusive_lease(id)?;
        let path = self.bundle(id);
        if !io_with_budget(fs::try_exists(&path), &cancel, deadline).await? {
            return Ok(DeleteOutcome::NotPresent);
        }
        no_link_ancestry(&path).await?;
        let manifest = self.manifest(id)?;
        verify_ownership(&path, manifest.as_ref(), &cancel, deadline).await?;
        remove_safe(&path, &cancel, deadline).await?;
        Ok(DeleteOutcome::Deleted)
    }
    fn manifest(&self, id: &str) -> Result<Arc<ModelBundleManifest>, ModelStoreError> {
        if !safe_name(id) {
            return Err(ModelStoreError::new(ModelStoreErrorCode::InvalidManifest));
        }
        self.catalog
            .get(id)
            .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::InvalidManifest))
    }
    fn bundle(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }
    fn lease_path(&self, id: &str) -> PathBuf {
        self.root.join(format!(".{id}.lease.lock"))
    }
    fn exclusive_lease(&self, id: &str) -> Result<StoreLock, ModelStoreError> {
        if self
            .leases
            .lock()
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::Io))?
            .get(id)
            .copied()
            .unwrap_or(0)
            > 0
        {
            return Err(ModelStoreError::new(ModelStoreErrorCode::InUse));
        }
        let file = lock_file(&self.lease_path(id))?;
        FileExt::try_lock_exclusive(&file)
            .map(|()| StoreLock(file))
            .map_err(|error| {
                if lock_is_contended(&error) {
                    ModelStoreError::new(ModelStoreErrorCode::InUse)
                } else {
                    ioe(error)
                }
            })
    }
    async fn lock(
        &self,
        id: &str,
        cancel: &CancellationToken,
        deadline: Instant,
    ) -> Result<StoreLock, ModelStoreError> {
        let f = lock_file(&self.root.join(format!(".{id}.lock")))?;
        loop {
            check_budget(cancel, deadline)?;
            match FileExt::try_lock_exclusive(&f) {
                Ok(()) => return Ok(StoreLock(f)),
                Err(e) if lock_is_contended(&e) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    sleep(Duration::from_millis(10).min(remaining)).await;
                }
                Err(e) => return Err(ioe(e)),
            }
        }
    }
    async fn prepare(
        &self,
        id: &str,
        cancel: &CancellationToken,
        deadline: Instant,
    ) -> Result<PathBuf, ModelStoreError> {
        check_budget(cancel, deadline)?;
        let p = self.root.join(format!(".{id}.staging"));
        if io_with_budget(fs::try_exists(&p), cancel, deadline).await? {
            no_link_ancestry(&p).await?;
            remove_safe(&p, cancel, deadline).await?;
        }
        check_budget(cancel, deadline)?;
        io_with_budget(fs::create_dir(&p), cancel, deadline).await?;
        no_link_ancestry(&p).await?;
        Ok(p)
    }
    async fn promote(
        &self,
        id: &str,
        stage: &Path,
        replacing: bool,
        cancel: &CancellationToken,
        deadline: Instant,
    ) -> Result<(), ModelStoreError> {
        check_budget(cancel, deadline)?;
        no_link_ancestry(stage).await?;
        let final_path = self.bundle(id);
        let q = self.root.join(format!(".{id}.quarantine"));
        if io_with_budget(fs::try_exists(&q), cancel, deadline).await? {
            no_link_ancestry(&q).await?;
            let manifest = self.manifest(id)?;
            verify_ownership(&q, manifest.as_ref(), cancel, deadline).await?;
            remove_safe(&q, cancel, deadline).await?;
        }
        if replacing {
            no_link_ancestry(&final_path).await?;
            check_budget(cancel, deadline)?;
            io_with_budget(fs::rename(&final_path, &q), cancel, deadline).await?;
            #[cfg(test)]
            if self.policy.fail_after_quarantine {
                io_with_budget(fs::rename(&q, &final_path), cancel, deadline).await?;
                return Err(ModelStoreError::new(ModelStoreErrorCode::Io));
            }
        }
        if let Err(error) = check_budget(cancel, deadline) {
            if replacing {
                let _ = fs::rename(&q, &final_path).await;
            }
            return Err(error);
        }
        if let Err(e) = io_with_budget(fs::rename(stage, &final_path), cancel, deadline).await {
            if replacing {
                let _ = fs::rename(&q, &final_path).await;
            }
            return Err(e);
        }
        let manifest = self.manifest(id)?;
        if let Err(error) = verify_dir(&final_path, &manifest, true, cancel, deadline).await {
            let moved_new_back = fs::rename(&final_path, stage).await.is_ok();
            if replacing && moved_new_back {
                let _ = fs::rename(&q, &final_path).await;
            }
            return Err(error);
        }
        sync_dir(&self.root)?;
        if let Err(error) = check_budget(cancel, deadline) {
            let moved_new_back = fs::rename(&final_path, stage).await.is_ok();
            if replacing && moved_new_back {
                let _ = fs::rename(&q, &final_path).await;
            }
            return Err(error);
        }
        if replacing {
            remove_safe(&q, cancel, deadline).await?;
            sync_dir(&self.root)?;
            check_budget(cancel, deadline)?;
        }
        Ok(())
    }
    async fn verify_manifest(
        &self,
        m: &ModelBundleManifest,
        cancel: &CancellationToken,
        deadline: Instant,
    ) -> Result<VerifiedBundle, ModelStoreError> {
        check_budget(cancel, deadline)?;
        let path = self.bundle(&m.bundle_id);
        if !io_with_budget(fs::try_exists(&path), cancel, deadline).await? {
            return Err(ModelStoreError::new(ModelStoreErrorCode::NotInstalled));
        }
        verify_dir(&path, m, true, cancel, deadline).await
    }
    async fn download(
        &self,
        a: &ModelArtifact,
        path: &Path,
        cancel: &CancellationToken,
        deadline: Instant,
    ) -> Result<(), ModelStoreError> {
        let mut url = a.source_url.clone();
        for redirects in 0..=self.policy.redirects {
            check_budget(cancel, deadline)?;
            if !valid_url(&url, self.policy.test_http) {
                return Err(ModelStoreError::new(ModelStoreErrorCode::UnapprovedSource));
            }
            let send = Box::pin(timeout_at(
                deadline,
                self.client
                    .get(url.clone())
                    .header(header::ACCEPT_ENCODING, "identity")
                    .send(),
            ));
            let cancellation = Box::pin(cancel.cancelled());
            let response = match select(cancellation, send).await {
                Either::Left(_) => {
                    return Err(ModelStoreError::new(ModelStoreErrorCode::Cancelled));
                }
                Either::Right((result, _)) => result
                    .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::TimedOut))?
                    .map_err(|e| ModelStoreError::source(ModelStoreErrorCode::Io, e))?,
            };
            if response.status().is_redirection() {
                if redirects == self.policy.redirects {
                    return Err(ModelStoreError::new(ModelStoreErrorCode::UnapprovedSource));
                }
                let location = response
                    .headers()
                    .get(header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::UnapprovedSource))?;
                url = url
                    .join(location)
                    .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::UnapprovedSource))?;
                continue;
            }
            if response.status() != StatusCode::OK {
                return Err(ModelStoreError::new(ModelStoreErrorCode::Io));
            }
            if response
                .content_length()
                .is_some_and(|n| n != a.byte_length)
            {
                return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
            }
            let mut out = io_with_budget(File::create(path), cancel, deadline).await?;
            let mut stream = response.bytes_stream();
            let mut h = Sha256::new();
            let mut n = 0_u64;
            loop {
                check_budget(cancel, deadline)?;
                let idle_deadline = Instant::now()
                    .checked_add(self.policy.idle)
                    .unwrap_or(deadline)
                    .min(deadline);
                let next_chunk = Box::pin(timeout_at(idle_deadline, stream.next()));
                let cancellation = Box::pin(cancel.cancelled());
                let next = match select(cancellation, next_chunk).await {
                    Either::Left(_) => {
                        return Err(ModelStoreError::new(ModelStoreErrorCode::Cancelled));
                    }
                    Either::Right((result, _)) => {
                        result.map_err(|_| ModelStoreError::new(ModelStoreErrorCode::TimedOut))?
                    }
                };
                let Some(chunk) = next else { break };
                let chunk =
                    chunk.map_err(|e| ModelStoreError::source(ModelStoreErrorCode::Io, e))?;
                n = n
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))?;
                if n > a.byte_length {
                    return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
                }
                h.update(&chunk);
                io_with_budget(out.write_all(&chunk), cancel, deadline).await?;
            }
            io_with_budget(out.flush(), cancel, deadline).await?;
            io_with_budget(out.sync_all(), cancel, deadline).await?;
            check_budget(cancel, deadline)?;
            return check_hash(n, &h.finalize(), a);
        }
        Err(ModelStoreError::new(ModelStoreErrorCode::UnapprovedSource))
    }
}

pub struct ModelLease {
    id: String,
    path: PathBuf,
    manifest: Arc<ModelBundleManifest>,
    file: Option<std::fs::File>,
    counts: Arc<Mutex<BTreeMap<String, usize>>>,
}
impl fmt::Debug for ModelLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelLease")
            .field("bundle_id", &self.id)
            .field("path", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}
impl ModelLease {
    #[must_use]
    pub fn artifact_path(&self, role: ModelRole, kind: ArtifactKind) -> Option<PathBuf> {
        self.manifest
            .artifact(role, kind)
            .ok()
            .map(|a| self.path.join(&a.filename))
    }
}
impl Drop for ModelLease {
    fn drop(&mut self) {
        if let Some(f) = self.file.take() {
            let _ = FileExt::unlock(&f);
        }
        if let Ok(mut m) = self.counts.lock() {
            if let Some(n) = m.get_mut(&self.id) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    m.remove(&self.id);
                }
            }
        }
    }
}
struct StoreLock(std::fs::File);
impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

async fn copy_verified<R: AsyncRead + Unpin>(
    mut input: R,
    path: &Path,
    a: &ModelArtifact,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<(), ModelStoreError> {
    let mut out = io_with_budget(File::create(path), cancel, deadline).await?;
    let mut buf = vec![0; 65536];
    let mut n = 0_u64;
    let mut h = Sha256::new();
    loop {
        check_budget(cancel, deadline)?;
        let read = io_with_budget(input.read(&mut buf), cancel, deadline).await?;
        if read == 0 {
            break;
        }
        n = n
            .checked_add(read as u64)
            .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))?;
        if n > a.byte_length {
            return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
        }
        h.update(&buf[..read]);
        io_with_budget(out.write_all(&buf[..read]), cancel, deadline).await?;
    }
    io_with_budget(out.flush(), cancel, deadline).await?;
    io_with_budget(out.sync_all(), cancel, deadline).await?;
    check_budget(cancel, deadline)?;
    check_hash(n, &h.finalize(), a)
}

async fn verify_dir(
    path: &Path,
    m: &ModelBundleManifest,
    marker: bool,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<VerifiedBundle, ModelStoreError> {
    check_budget(cancel, deadline)?;
    no_link_ancestry(path).await?;
    let mut allowed: BTreeSet<&str> = m.artifacts.iter().map(|a| a.filename.as_str()).collect();
    if marker {
        allowed.insert(OWNER);
        allowed.insert(LEGACY_LEASE);
    }
    let mut dir = io_with_budget(fs::read_dir(path), cancel, deadline).await?;
    while let Some(e) = io_with_budget(dir.next_entry(), cancel, deadline).await? {
        check_budget(cancel, deadline)?;
        no_link_ancestry(&e.path()).await?;
        let name = e.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::UnsafeFilesystem))?;
        if !allowed.contains(name) {
            return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
        }
    }
    for a in &m.artifacts {
        let p = path.join(&a.filename);
        no_link_ancestry(&p).await?;
        hash_file(&p, a, cancel, deadline).await?;
    }
    let verified = validate_yaml(path, m, cancel, deadline).await?;
    if marker {
        let bytes = read_bounded(&path.join(OWNER), cancel, deadline).await?;
        let actual: Marker = serde_json::from_slice(&bytes)
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))?;
        if actual != marker_for(m, &verified) {
            return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
        }
    }
    Ok(verified)
}
async fn hash_file(
    path: &Path,
    artifact: &ModelArtifact,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<(), ModelStoreError> {
    no_link_ancestry(path).await?;
    let mut file = io_with_budget(File::open(path), cancel, deadline).await?;
    let mut buffer = vec![0; 65536];
    let mut byte_count = 0;
    let mut hasher = Sha256::new();
    loop {
        check_budget(cancel, deadline)?;
        let read = io_with_budget(file.read(&mut buffer), cancel, deadline).await?;
        if read == 0 {
            break;
        }
        byte_count += u64::try_from(read)
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))?;
        if byte_count > artifact.byte_length {
            return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
        }
        hasher.update(&buffer[..read]);
    }
    check_hash(byte_count, &hasher.finalize(), artifact)
}
fn check_hash(n: u64, d: &[u8], a: &ModelArtifact) -> Result<(), ModelStoreError> {
    if n == a.byte_length && hex(d) == a.sha256.as_str() {
        Ok(())
    } else {
        Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))
    }
}

async fn validate_yaml(
    path: &Path,
    m: &ModelBundleManifest,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<VerifiedBundle, ModelStoreError> {
    check_budget(cancel, deadline)?;
    let det = std::str::from_utf8(
        &read_bounded(
            &path.join(
                &m.artifact(ModelRole::Detector, ArtifactKind::InferenceConfig)?
                    .filename,
            ),
            cancel,
            deadline,
        )
        .await?,
    )
    .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))?
    .to_owned();
    let rec = std::str::from_utf8(
        &read_bounded(
            &path.join(
                &m.artifact(ModelRole::EnglishRecognizer, ArtifactKind::InferenceConfig)?
                    .filename,
            ),
            cancel,
            deadline,
        )
        .await?,
    )
    .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))?
    .to_owned();
    check_budget(cancel, deadline)?;
    for (k, v) in [
        ("resize_long", "960"),
        ("thresh", "0.3"),
        ("box_thresh", "0.6"),
        ("max_candidates", "1000"),
        ("unclip_ratio", "1.5"),
    ] {
        scalar(&det, k, v)?;
    }
    key(&det, "x")?;
    // The pinned exported ONNX config omits PaddleOCR's `use_space_char` field. The
    // curated profile deliberately enables it; if a future pinned config starts
    // spelling the field out, only one exact `true` value remains admissible.
    optional_true(&rec, "use_space_char")?;
    scalar(&rec, "name", "CTCLabelDecode")?;
    key(&rec, "x")?;
    if !sequence(&rec, &[3, 48, 320]) || !sequence(&rec, &[8, 3, 48, 3200]) {
        return Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ));
    }
    let dict = dictionary(&rec)?;
    if dict.len() != 436 || dict.iter().any(String::is_empty) {
        return Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ));
    }
    let mut h = Sha256::new();
    h.update(b"paddle-character-dictionary-v1\0");
    for s in &dict {
        check_budget(cancel, deadline)?;
        let length = u32::try_from(s.len())
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))?;
        h.update(length.to_be_bytes());
        h.update(s.as_bytes());
    }
    let digest: [u8; 32] = h.finalize().into();
    let decoder = admit_dictionary(&dict, digest)?;
    Ok(VerifiedBundle {
        bundle_id: m.bundle_id.clone(),
        source_dictionary_sha256: Sha256Digest(hex(&digest)),
        recognizer_class_count: decoder.class_count(),
        profile_id: BUNDLE.into(),
        upstream_detector_resize_long: 960,
        runtime_detector_max_side: 960,
        recognizer_dictionary: dict,
    })
}

fn admit_dictionary(entries: &[String], digest: [u8; 32]) -> Result<CtcDecoder, ModelStoreError> {
    CtcDecoder::new_verified(entries.to_vec(), true, digest)
        .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))
}

fn decode_sha256(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        digest[index] = (high << 4) | low;
    }
    Some(digest)
}

const fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
fn key(t: &str, k: &str) -> Result<(), ModelStoreError> {
    let x = format!("{k}:");
    if t.lines().any(|l| l.trim() == x) {
        Ok(())
    } else {
        Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ))
    }
}
fn scalar(t: &str, k: &str, v: &str) -> Result<(), ModelStoreError> {
    let p = format!("{k}:");
    if t.lines()
        .any(|l| l.trim().strip_prefix(&p).is_some_and(|x| x.trim() == v))
    {
        Ok(())
    } else {
        Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ))
    }
}
fn optional_true(t: &str, key: &str) -> Result<(), ModelStoreError> {
    let prefix = format!("{key}:");
    let mut values = t
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&prefix).map(str::trim));
    match (values.next(), values.next()) {
        (None | Some("true"), None) => Ok(()),
        _ => Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        )),
    }
}
fn sequence(t: &str, w: &[u32]) -> bool {
    let v: Vec<u32> = t
        .lines()
        .filter_map(|l| l.trim().strip_prefix("- "))
        .map(|value| value.strip_prefix("- ").unwrap_or(value))
        .filter_map(|x| x.parse().ok())
        .collect();
    v.windows(w.len()).any(|x| x == w)
}
fn dictionary(t: &str) -> Result<Vec<String>, ModelStoreError> {
    let mut lines = t.lines();
    let mut indent = None;
    for l in lines.by_ref() {
        if l.trim() == "character_dict:" {
            indent = Some(spaces(l)?);
            break;
        }
    }
    let base =
        indent.ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))?;
    let mut out = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let indentation = spaces(l)?;
        let trimmed = l.trim();
        if indentation < base || !trimmed.starts_with("- ") {
            break;
        }
        let v = trimmed
            .strip_prefix("- ")
            .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))?;
        out.push(yaml_string(v)?);
        if out.len() > 436 {
            return Err(ModelStoreError::new(
                ModelStoreErrorCode::ConfigurationMismatch,
            ));
        }
    }
    Ok(out)
}
fn spaces(l: &str) -> Result<usize, ModelStoreError> {
    if l.contains('\t') {
        Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ))
    } else {
        Ok(l.len() - l.trim_start_matches(' ').len())
    }
}
fn yaml_string(v: &str) -> Result<String, ModelStoreError> {
    let v = v.trim();
    if v.starts_with('"') {
        serde_json::from_str(v)
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::ConfigurationMismatch))
    } else if v.starts_with('\'') && v.len() >= 2 && v.ends_with('\'') {
        Ok(v[1..v.len() - 1].replace("''", "'"))
    } else if v.is_empty() || v.starts_with('#') {
        Err(ModelStoreError::new(
            ModelStoreErrorCode::ConfigurationMismatch,
        ))
    } else {
        Ok(v.into())
    }
}

async fn write_marker(
    path: &Path,
    m: &ModelBundleManifest,
    v: &VerifiedBundle,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<(), ModelStoreError> {
    check_budget(cancel, deadline)?;
    let bytes = serde_json::to_vec_pretty(&marker_for(m, v))
        .map_err(|e| ModelStoreError::source(ModelStoreErrorCode::Io, e))?;
    let mut f = io_with_budget(File::create(path.join(OWNER)), cancel, deadline).await?;
    io_with_budget(f.write_all(&bytes), cancel, deadline).await?;
    io_with_budget(f.flush(), cancel, deadline).await?;
    io_with_budget(f.sync_all(), cancel, deadline).await?;
    check_budget(cancel, deadline)?;
    sync_dir(path)?;
    check_budget(cancel, deadline)
}

async fn verify_ownership(
    path: &Path,
    manifest: &ModelBundleManifest,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<(), ModelStoreError> {
    check_budget(cancel, deadline)?;
    no_link_ancestry(path).await?;
    let bytes = read_bounded(&path.join(OWNER), cancel, deadline).await?;
    let marker: Marker = serde_json::from_slice(&bytes)
        .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::UnsafeFilesystem))?;
    if marker.bundle_id == manifest.bundle_id && marker.manifest_sha256 == manifest.fingerprint() {
        Ok(())
    } else {
        Err(ModelStoreError::new(ModelStoreErrorCode::UnsafeFilesystem))
    }
}

#[allow(clippy::unnecessary_wraps)] // Windows is infallible; Unix fsync is not.
fn sync_dir(path: &Path) -> Result<(), ModelStoreError> {
    #[cfg(unix)]
    {
        return std::fs::File::open(path)
            .map_err(ioe)?
            .sync_all()
            .map_err(ioe);
    }
    #[cfg(windows)]
    {
        // All payload files and the ownership marker are individually flushed before the
        // same-volume rename. Stable Rust does not expose a portable Windows directory flush.
        let _ = path;
        Ok(())
    }
}
fn marker_for(m: &ModelBundleManifest, v: &VerifiedBundle) -> Marker {
    Marker {
        schema_version: 1,
        bundle_id: m.bundle_id.clone(),
        manifest_sha256: m.fingerprint(),
        dictionary_sha256: v.source_dictionary_sha256.as_str().into(),
        recognizer_class_count: v.recognizer_class_count,
        profile_id: v.profile_id.clone(),
        upstream_resize_long: v.upstream_detector_resize_long,
        runtime_max_side: v.runtime_detector_max_side,
        yaml_preprocessing_adopted: false,
    }
}
async fn read_bounded(
    path: &Path,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<Vec<u8>, ModelStoreError> {
    check_budget(cancel, deadline)?;
    no_link_ancestry(path).await?;
    let md = io_with_budget(fs::metadata(path), cancel, deadline).await?;
    if !md.is_file() || md.len() > MAX_YAML {
        return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
    }
    let capacity = usize::try_from(md.len())
        .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))?;
    let mut b = Vec::with_capacity(capacity);
    let mut file = io_with_budget(File::open(path), cancel, deadline)
        .await?
        .take(MAX_YAML + 1);
    let mut chunk = [0_u8; 8192];
    loop {
        check_budget(cancel, deadline)?;
        let read = io_with_budget(file.read(&mut chunk), cancel, deadline).await?;
        if read == 0 {
            break;
        }
        b.extend_from_slice(&chunk[..read]);
        if b.len() as u64 > MAX_YAML {
            return Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch));
        }
    }
    if b.len() as u64 > MAX_YAML {
        Err(ModelStoreError::new(ModelStoreErrorCode::IntegrityMismatch))
    } else {
        Ok(b)
    }
}
fn valid_url(u: &Url, test: bool) -> bool {
    let Some(h) = u.host_str() else { return false };
    let prod = u.scheme() == "https"
        && u.username().is_empty()
        && u.password().is_none()
        && u.port().is_none()
        && HOSTS.contains(&h);
    #[cfg(test)]
    let local = test && u.scheme() == "http" && matches!(h, "127.0.0.1" | "localhost" | "::1");
    #[cfg(not(test))]
    let local = {
        let _ = test;
        false
    };
    prod || local
}
fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
fn no_parent(p: &Path) -> Result<(), ModelStoreError> {
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        Err(ModelStoreError::new(ModelStoreErrorCode::UnsafeFilesystem))
    } else {
        Ok(())
    }
}
async fn no_link_ancestry(path: &Path) -> Result<(), ModelStoreError> {
    no_parent(path)?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map_err(ioe)?.join(path)
    };
    let mut ancestors: Vec<&Path> = absolute.ancestors().collect();
    ancestors.reverse();
    for ancestor in ancestors {
        match fs::symlink_metadata(ancestor).await {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || reparse(&metadata) {
                    return Err(ModelStoreError::new(ModelStoreErrorCode::UnsafeFilesystem));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(ioe(error)),
        }
    }
    Ok(())
}
#[cfg(windows)]
fn reparse(m: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    m.file_attributes() & 0x400 != 0
}
#[cfg(not(windows))]
fn reparse(_: &std::fs::Metadata) -> bool {
    false
}
fn lock_file(p: &Path) -> Result<std::fs::File, ModelStoreError> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(p)
        .map_err(ioe)
}
fn lock_is_contended(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock || matches!(error.raw_os_error(), Some(11 | 33 | 35))
}
fn remove_safe<'a>(
    p: &'a Path,
    cancel: &'a CancellationToken,
    deadline: Instant,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ModelStoreError>> + Send + 'a>> {
    Box::pin(async move {
        check_budget(cancel, deadline)?;
        no_link_ancestry(p).await?;
        let md = io_with_budget(fs::metadata(p), cancel, deadline).await?;
        if md.is_file() {
            return io_with_budget(fs::remove_file(p), cancel, deadline).await;
        }
        let mut d = io_with_budget(fs::read_dir(p), cancel, deadline).await?;
        while let Some(e) = io_with_budget(d.next_entry(), cancel, deadline).await? {
            let c = e.path();
            no_link_ancestry(&c).await?;
            if io_with_budget(fs::metadata(&c), cancel, deadline)
                .await?
                .is_dir()
            {
                remove_safe(&c, cancel, deadline).await?;
            } else {
                io_with_budget(fs::remove_file(&c), cancel, deadline).await?;
            }
        }
        io_with_budget(fs::remove_dir(p), cancel, deadline).await
    })
}
fn cancelled(c: &CancellationToken) -> Result<(), ModelStoreError> {
    if c.is_cancelled() {
        Err(ModelStoreError::new(ModelStoreErrorCode::Cancelled))
    } else {
        Ok(())
    }
}
fn operation_deadline(total: Duration) -> Result<Instant, ModelStoreError> {
    Instant::now()
        .checked_add(total)
        .ok_or_else(|| ModelStoreError::new(ModelStoreErrorCode::TimedOut))
}
fn check_budget(cancel: &CancellationToken, deadline: Instant) -> Result<(), ModelStoreError> {
    cancelled(cancel)?;
    if Instant::now() >= deadline {
        Err(ModelStoreError::new(ModelStoreErrorCode::TimedOut))
    } else {
        Ok(())
    }
}
async fn io_with_budget<T, F>(
    future: F,
    cancel: &CancellationToken,
    deadline: Instant,
) -> Result<T, ModelStoreError>
where
    F: std::future::Future<Output = io::Result<T>>,
{
    check_budget(cancel, deadline)?;
    let operation = Box::pin(timeout_at(deadline, future));
    let cancellation = Box::pin(cancel.cancelled());
    match select(cancellation, operation).await {
        Either::Left(_) => Err(ModelStoreError::new(ModelStoreErrorCode::Cancelled)),
        Either::Right((result, _)) => result
            .map_err(|_| ModelStoreError::new(ModelStoreErrorCode::TimedOut))?
            .map_err(ioe),
    }
}
fn ioe(e: io::Error) -> ModelStoreError {
    ModelStoreError::source(ModelStoreErrorCode::Io, e)
}
fn hex(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(char::from(H[usize::from(x >> 4)]));
        s.push(char::from(H[usize::from(x & 15)]));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, Bytes},
        extract::{Path as AxumPath, State},
        http::Response,
        routing::get,
    };
    use std::convert::Infallible;
    #[cfg(windows)]
    use std::process::Command;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    #[derive(Clone)]
    struct Files(Arc<BTreeMap<String, Vec<u8>>>);

    fn detector_yaml() -> Vec<u8> {
        b"Hpi:\n  x:\n    - 1\nPreProcess:\n  resize_long: 960\nPostProcess:\n  thresh: 0.3\n  box_thresh: 0.6\n  max_candidates: 1000\n  unclip_ratio: 1.5\n".to_vec()
    }

    #[test]
    fn numeric_sequence_accepts_paddle_nested_dynamic_shapes() {
        assert!(sequence(
            "x:\n- - 8\n  - 3\n  - 48\n  - 3200\n",
            &[8, 3, 48, 3200]
        ));
    }

    #[allow(clippy::format_push_string)] // The bounded test fixture favors direct readable YAML.
    fn recognizer_yaml() -> Vec<u8> {
        let mut value = String::from(
            "Hpi:\n  x:\n    - 1\n    - 3\n    - 48\n    - 160\n    - 1\n    - 3\n    - 48\n    - 320\n    - 8\n    - 3\n    - 48\n    - 3200\nPreProcess:\n  image_shape:\n    - 3\n    - 48\n    - 320\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n",
        );
        for index in 0..436_u32 {
            let scalar = char::from_u32(0xe000 + index).unwrap_or_else(|| unreachable!());
            value.push_str(&format!("  - {scalar}\n"));
        }
        value.push_str("  use_space_char: true\n");
        value.into_bytes()
    }

    fn sha(bytes: &[u8]) -> Sha256Digest {
        Sha256Digest::new(hex(&Sha256::digest(bytes))).unwrap_or_else(|_| unreachable!())
    }

    fn test_catalog(base: &Url, graph: &[u8], detector: &[u8], recognizer: &[u8]) -> ModelCatalog {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let make = |role, kind, filename: &str, endpoint: &str, bytes: &[u8]| ModelArtifact {
            role,
            kind,
            upstream_project: "PaddlePaddle/synthetic".into(),
            revision: revision.into(),
            filename: filename.into(),
            byte_length: bytes.len() as u64,
            sha256: sha(bytes),
            license: "Apache-2.0".into(),
            source_url: base
                .join(&format!("{revision}/{endpoint}"))
                .unwrap_or_else(|_| unreachable!()),
        };
        let manifest = ModelBundleManifest {
            bundle_id: BUNDLE.into(),
            schema_version: 1,
            artifacts: vec![
                make(
                    ModelRole::Detector,
                    ArtifactKind::OnnxGraph,
                    "detector.onnx",
                    "detector.onnx",
                    graph,
                ),
                make(
                    ModelRole::Detector,
                    ArtifactKind::InferenceConfig,
                    "detector-inference.yml",
                    "detector.yml",
                    detector,
                ),
                make(
                    ModelRole::EnglishRecognizer,
                    ArtifactKind::OnnxGraph,
                    "english-recognizer.onnx",
                    "recognizer.onnx",
                    graph,
                ),
                make(
                    ModelRole::EnglishRecognizer,
                    ArtifactKind::InferenceConfig,
                    "english-recognizer-inference.yml",
                    "recognizer.yml",
                    recognizer,
                ),
            ],
        };
        manifest
            .validate_policy(true)
            .unwrap_or_else(|_| unreachable!());
        let mut catalog = ModelCatalog::default();
        catalog
            .insert_checked(manifest)
            .unwrap_or_else(|_| unreachable!());
        catalog
    }

    async fn handler(
        AxumPath((_revision, file)): AxumPath<(String, String)>,
        State(files): State<Files>,
    ) -> Response<Body> {
        match files.0.get(&file) {
            Some(bytes) => Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(bytes.clone()))
                .unwrap_or_else(|_| unreachable!()),
            None => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::empty())
                .unwrap_or_else(|_| unreachable!()),
        }
    }

    async fn mock_server(files: BTreeMap<String, Vec<u8>>) -> (Url, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/{revision}/{file}", get(handler))
            .with_state(Files(Arc::new(files)));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (
            Url::parse(&format!("http://{address}/")).unwrap_or_else(|_| unreachable!()),
            task,
        )
    }

    async fn store(temp: &TempDir, catalog: ModelCatalog) -> ModelStore {
        ModelStore::open_policy(
            temp.path(),
            catalog,
            Policy {
                total: Duration::from_secs(5),
                idle: Duration::from_millis(500),
                redirects: 2,
                test_http: true,
                fail_after_quarantine: false,
            },
        )
        .await
        .unwrap_or_else(|_| unreachable!())
    }

    async fn slow_handler(State(bytes): State<Arc<Vec<u8>>>) -> Response<Body> {
        let stream = futures_util::stream::unfold(0_usize, move |index| {
            let bytes = Arc::clone(&bytes);
            async move {
                if index == bytes.len() {
                    None
                } else {
                    sleep(Duration::from_millis(30)).await;
                    Some((
                        Ok::<Bytes, Infallible>(Bytes::copy_from_slice(&bytes[index..=index])),
                        index + 1,
                    ))
                }
            }
        });
        Response::builder()
            .status(StatusCode::OK)
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| unreachable!())
    }

    async fn slow_server(bytes: Vec<u8>) -> (Url, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/{revision}/{file}", get(slow_handler))
            .with_state(Arc::new(bytes));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (
            Url::parse(&format!("http://{address}/")).unwrap_or_else(|_| unreachable!()),
            task,
        )
    }

    fn make_directory_link(link: &Path, target: &Path) {
        #[cfg(windows)]
        {
            let output = Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .output()
                .unwrap_or_else(|_| unreachable!());
            assert!(output.status.success());
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link).unwrap_or_else(|_| unreachable!());
        #[cfg(not(any(unix, windows)))]
        compile_error!("the lifecycle link regression requires symlink support");
    }

    #[test]
    fn curated_catalog_is_exact_and_redacts_sources() {
        let catalog = ModelCatalog::curated().unwrap_or_else(|_| unreachable!());
        let manifest = catalog.get(BUNDLE).unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.artifacts.len(), 4);
        assert_eq!(
            manifest
                .artifact(ModelRole::Detector, ArtifactKind::OnnxGraph)
                .unwrap_or_else(|_| unreachable!())
                .byte_length,
            4_826_518
        );
        assert_eq!(
            manifest
                .artifact(ModelRole::EnglishRecognizer, ArtifactKind::OnnxGraph)
                .unwrap_or_else(|_| unreachable!())
                .byte_length,
            7_848_423
        );
        assert!(!format!("{manifest:?}").contains("resolve/"));
    }

    #[test]
    fn config_parser_requires_profile_and_ordered_436_entry_dictionary() {
        let rec = String::from_utf8(recognizer_yaml()).unwrap_or_else(|_| unreachable!());
        let entries = dictionary(&rec).unwrap_or_else(|_| unreachable!());
        assert_eq!(entries.len(), 436);
        assert_eq!(entries.first().map(String::as_str), Some("\u{e000}"));
        assert_eq!(entries.last().map(String::as_str), Some("\u{e1b3}"));
        let mutant = String::from_utf8(detector_yaml())
            .unwrap_or_else(|_| unreachable!())
            .replace("box_thresh: 0.6", "box_thresh: 0.7");
        assert!(scalar(&mutant, "box_thresh", "0.6").is_err());
        assert!(optional_true("PostProcess:\n  name: CTCLabelDecode\n", "use_space_char").is_ok());
        assert!(optional_true("use_space_char: false\n", "use_space_char").is_err());
        assert!(
            optional_true(
                "use_space_char: true\nuse_space_char: true\n",
                "use_space_char"
            )
            .is_err()
        );

        let mut duplicate = entries.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(admit_dictionary(&duplicate, [0_u8; 32]).is_err());
        let mut multiple_scalars = entries;
        multiple_scalars[0] = "ab".into();
        assert!(admit_dictionary(&multiple_scalars, [0_u8; 32]).is_err());
    }

    #[tokio::test]
    async fn explicit_install_offline_verify_concurrency_and_lease_delete() {
        let graph = b"synthetic-graph".to_vec();
        let det = detector_yaml();
        let rec = recognizer_yaml();
        let files = BTreeMap::from([
            ("detector.onnx".into(), graph.clone()),
            ("recognizer.onnx".into(), graph.clone()),
            ("detector.yml".into(), det.clone()),
            ("recognizer.yml".into(), rec.clone()),
        ]);
        let (base, task) = mock_server(files).await;
        let catalog = test_catalog(&base, &graph, &det, &rec);
        let temp = TempDir::new().unwrap_or_else(|_| unreachable!());
        let store = Arc::new(store(&temp, catalog).await);
        let one = {
            let store = Arc::clone(&store);
            tokio::spawn(async move { store.install(BUNDLE, &CancellationToken::new()).await })
        };
        let two = {
            let store = Arc::clone(&store);
            tokio::spawn(async move { store.install(BUNDLE, &CancellationToken::new()).await })
        };
        let outcomes = [
            one.await
                .unwrap_or_else(|_| unreachable!())
                .unwrap_or_else(|_| unreachable!()),
            two.await
                .unwrap_or_else(|_| unreachable!())
                .unwrap_or_else(|_| unreachable!()),
        ];
        assert!(outcomes.contains(&InstallOutcome::Installed));
        assert!(outcomes.contains(&InstallOutcome::AlreadyVerified));
        task.abort();
        assert_eq!(store.status(BUNDLE).await, ModelStoreStatus::Verified);
        let verified = store
            .verify(BUNDLE)
            .await
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(verified.recognizer_class_count, 438);
        let decoder = verified.ctc_decoder().unwrap_or_else(|_| unreachable!());
        assert_eq!(decoder.class_count(), 438);
        assert!(!format!("{verified:?}").contains('\u{e000}'));
        let lease = store.lease(BUNDLE).await.unwrap_or_else(|_| unreachable!());
        assert_eq!(
            store
                .delete(BUNDLE)
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::InUse)
        );
        drop(lease);
        assert_eq!(
            store
                .delete(BUNDLE)
                .await
                .unwrap_or_else(|_| unreachable!()),
            DeleteOutcome::Deleted
        );
    }

    #[tokio::test]
    async fn mismatch_cancel_traversal_and_unowned_partial_fail_closed() {
        let graph = b"expected".to_vec();
        let det = detector_yaml();
        let rec = recognizer_yaml();
        let files = BTreeMap::from([
            ("detector.onnx".into(), b"oversized-or-wrong".to_vec()),
            ("recognizer.onnx".into(), graph.clone()),
            ("detector.yml".into(), det.clone()),
            ("recognizer.yml".into(), rec.clone()),
        ]);
        let (base, task) = mock_server(files).await;
        let catalog = test_catalog(&base, &graph, &det, &rec);
        let temp = TempDir::new().unwrap_or_else(|_| unreachable!());
        let store = store(&temp, catalog).await;
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::IntegrityMismatch)
        );
        assert_eq!(store.status(BUNDLE).await, ModelStoreStatus::Missing);
        task.abort();
        let cancelled = CancellationToken::new();
        assert!(cancelled.cancel());
        assert_eq!(
            store
                .install(BUNDLE, &cancelled)
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::Cancelled)
        );
        fs::create_dir(store.bundle(BUNDLE))
            .await
            .unwrap_or_else(|_| unreachable!());
        fs::write(store.bundle(BUNDLE).join("private"), b"sentinel")
            .await
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(
            store
                .delete(BUNDLE)
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::Io)
        );
        assert!(
            fs::try_exists(store.bundle(BUNDLE).join("private"))
                .await
                .unwrap_or(false)
        );
        assert!(no_parent(Path::new("safe/../escape")).is_err());
    }

    #[tokio::test]
    async fn absolute_deadline_stops_slow_drip_and_local_copy() {
        let graph = b"slow-graph".to_vec();
        let detector = detector_yaml();
        let recognizer = recognizer_yaml();
        let (base, task) = slow_server(graph.clone()).await;
        let catalog = test_catalog(&base, &graph, &detector, &recognizer);
        let temp = TempDir::new().unwrap_or_else(|_| unreachable!());
        let store = ModelStore::open_policy(
            temp.path(),
            catalog.clone(),
            Policy {
                total: Duration::from_millis(80),
                idle: Duration::from_millis(60),
                redirects: 1,
                test_http: true,
                fail_after_quarantine: false,
            },
        )
        .await
        .unwrap_or_else(|_| unreachable!());
        let started = Instant::now();
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::TimedOut)
        );
        assert!(started.elapsed() < Duration::from_millis(400));
        task.abort();

        let artifact = catalog
            .get(BUNDLE)
            .unwrap_or_else(|| unreachable!())
            .artifact(ModelRole::Detector, ArtifactKind::OnnxGraph)
            .unwrap_or_else(|_| unreachable!())
            .clone();
        let (reader, mut writer) = tokio::io::duplex(1);
        let writer_task = tokio::spawn(async move {
            for byte in graph {
                sleep(Duration::from_millis(30)).await;
                if writer.write_all(&[byte]).await.is_err() {
                    break;
                }
            }
        });
        let output = temp.path().join("slow-import.part");
        assert_eq!(
            copy_verified(
                reader,
                &output,
                &artifact,
                &CancellationToken::new(),
                Instant::now() + Duration::from_millis(80),
            )
            .await
            .as_ref()
            .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::TimedOut)
        );
        writer_task.abort();
    }

    #[tokio::test]
    async fn lease_blocks_repair_and_success_removes_quarantine_with_rollback() {
        let graph = b"synthetic-graph".to_vec();
        let detector = detector_yaml();
        let recognizer = recognizer_yaml();
        let files = BTreeMap::from([
            ("detector.onnx".into(), graph.clone()),
            ("recognizer.onnx".into(), graph.clone()),
            ("detector.yml".into(), detector.clone()),
            ("recognizer.yml".into(), recognizer.clone()),
        ]);
        let (base, task) = mock_server(files).await;
        let catalog = test_catalog(&base, &graph, &detector, &recognizer);
        let temp = TempDir::new().unwrap_or_else(|_| unreachable!());
        let store = store(&temp, catalog.clone()).await;
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .unwrap_or_else(|_| unreachable!()),
            InstallOutcome::Installed
        );
        let lease = store.lease(BUNDLE).await.unwrap_or_else(|_| unreachable!());
        fs::write(store.bundle(BUNDLE).join("detector.onnx"), b"corrupt")
            .await
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::InUse)
        );
        drop(lease);
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .unwrap_or_else(|_| unreachable!()),
            InstallOutcome::Repaired
        );
        let quarantine = temp.path().join(format!(".{BUNDLE}.quarantine"));
        assert!(!fs::try_exists(&quarantine).await.unwrap_or(true));

        fs::write(store.bundle(BUNDLE).join("detector.onnx"), b"corrupt-again")
            .await
            .unwrap_or_else(|_| unreachable!());
        let failing = ModelStore::open_policy(
            temp.path(),
            catalog,
            Policy {
                total: Duration::from_secs(5),
                idle: Duration::from_millis(500),
                redirects: 2,
                test_http: true,
                fail_after_quarantine: true,
            },
        )
        .await
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(
            failing
                .install(BUNDLE, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::Io)
        );
        assert_eq!(failing.status(BUNDLE).await, ModelStoreStatus::Corrupt);
        assert!(!fs::try_exists(&quarantine).await.unwrap_or(true));
        task.abort();
    }

    #[tokio::test]
    async fn parent_directory_links_are_rejected_for_store_staging_and_import() {
        let graph = b"graph".to_vec();
        let detector = detector_yaml();
        let recognizer = recognizer_yaml();
        let base = Url::parse("http://127.0.0.1:9/").unwrap_or_else(|_| unreachable!());
        let catalog = test_catalog(&base, &graph, &detector, &recognizer);
        let temp = TempDir::new().unwrap_or_else(|_| unreachable!());
        let real_store = temp.path().join("real-store");
        fs::create_dir(&real_store)
            .await
            .unwrap_or_else(|_| unreachable!());
        let linked_store = temp.path().join("linked-store");
        make_directory_link(&linked_store, &real_store);
        let open_error = ModelStore::open_policy(&linked_store, catalog.clone(), Policy::default())
            .await
            .err()
            .map(|error| error.code());
        assert_eq!(open_error, Some(ModelStoreErrorCode::UnsafeFilesystem));

        let store_root = temp.path().join("store");
        let store = ModelStore::open_policy(&store_root, catalog, Policy::default())
            .await
            .unwrap_or_else(|_| unreachable!());
        let real_sources = temp.path().join("real-sources");
        fs::create_dir(&real_sources)
            .await
            .unwrap_or_else(|_| unreachable!());
        for (name, bytes) in [
            ("detector.onnx", graph.as_slice()),
            ("english-recognizer.onnx", graph.as_slice()),
            ("detector-inference.yml", detector.as_slice()),
            ("english-recognizer-inference.yml", recognizer.as_slice()),
        ] {
            fs::write(real_sources.join(name), bytes)
                .await
                .unwrap_or_else(|_| unreachable!());
        }
        let linked_sources = temp.path().join("linked-sources");
        make_directory_link(&linked_sources, &real_sources);
        let sources = ImportBundle {
            detector_graph: linked_sources.join("detector.onnx"),
            detector_config: linked_sources.join("detector-inference.yml"),
            recognizer_graph: linked_sources.join("english-recognizer.onnx"),
            recognizer_config: linked_sources.join("english-recognizer-inference.yml"),
        };
        assert_eq!(
            store
                .import(BUNDLE, &sources, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::UnsafeFilesystem)
        );

        let stage_target = temp.path().join("stage-target");
        fs::create_dir(&stage_target)
            .await
            .unwrap_or_else(|_| unreachable!());
        make_directory_link(
            &store_root.join(format!(".{BUNDLE}.staging")),
            &stage_target,
        );
        assert_eq!(
            store
                .install(BUNDLE, &CancellationToken::new())
                .await
                .as_ref()
                .map_err(ModelStoreError::code),
            Err(ModelStoreErrorCode::UnsafeFilesystem)
        );
    }
}
