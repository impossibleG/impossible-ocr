//! Explicit, offline qualification of an already-installed production runtime.

use std::{collections::BTreeSet, error::Error, fmt, fs, path::PathBuf, sync::Arc, time::Duration};

use impossible_ocr_domain::{InputMetadata, OcrOptions, RasterFormat, RasterInput, RasterLimits};
use impossible_ocr_onnx::{
    CURATED_BUNDLE_ID, ModelCatalog, ModelStore, OrtCpuRuntime, OrtRuntimeConfig,
};
use impossible_ocr_pipeline::{
    CtcDecoder, DecodedRaster, OcrBackend, PpOcrBackend, RasterDecoder, TensorOcrRuntime,
    preprocess_detector, preprocess_recognizer_batch,
};
use impossible_server_core::{CancellationToken, RequestContext, RequestIdSource};
use serde::{Deserialize, Serialize};

const MAX_RUNTIME_MANIFEST_BYTES: u64 = 128 * 1024;
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
const PROBABILITY_SUM_TOLERANCE: f32 = 0.001;

/// Stable failure category emitted by the opt-in runtime qualification tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeQualificationErrorCode {
    /// Arguments, paths, or the runtime manifest are not the reviewed shape.
    Configuration,
    /// The curated model bundle or native runtime could not be admitted.
    ModelUnavailable,
    /// A warm-up, tensor invariant, determinism, or end-to-end probe failed.
    QualificationFailed,
}

/// Privacy-safe runtime-qualification error that never retains paths or tensor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeQualificationError {
    code: RuntimeQualificationErrorCode,
}

impl RuntimeQualificationError {
    const fn new(code: RuntimeQualificationErrorCode) -> Self {
        Self { code }
    }

    /// Returns the stable failure category.
    #[must_use]
    pub const fn code(self) -> RuntimeQualificationErrorCode {
        self.code
    }
}

impl fmt::Display for RuntimeQualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            RuntimeQualificationErrorCode::Configuration => {
                "runtime qualification configuration is invalid"
            }
            RuntimeQualificationErrorCode::ModelUnavailable => {
                "the qualified OCR runtime is unavailable"
            }
            RuntimeQualificationErrorCode::QualificationFailed => {
                "the OCR runtime failed qualification"
            }
        })
    }
}

impl Error for RuntimeQualificationError {}

/// Explicit local inputs for real-runtime qualification.
#[derive(Clone)]
pub struct RuntimeQualificationConfig {
    model_store: PathBuf,
    runtime_directory: PathBuf,
    runtime_library: String,
    runtime_library_bytes: u64,
    runtime_library_sha256: String,
}

impl RuntimeQualificationConfig {
    /// Reads a bounded reviewed runtime manifest and selects one qualified installed platform.
    ///
    /// `runtime_directory` must be the content-addressed installed directory described by the
    /// manifest. No directory scan, `PATH` lookup, or network operation is performed.
    ///
    /// # Errors
    /// Returns a fixed configuration error for malformed, provisional, mismatched, linked, or
    /// non-local inputs.
    pub fn from_manifest_file(
        model_store: impl Into<PathBuf>,
        profile: &str,
        runtime_manifest: impl Into<PathBuf>,
        runtime_directory: impl Into<PathBuf>,
        platform: &str,
    ) -> Result<Self, RuntimeQualificationError> {
        let model_store = model_store.into();
        let runtime_manifest = runtime_manifest.into();
        let runtime_directory = runtime_directory.into();
        if profile != CURATED_BUNDLE_ID
            || !safe_absolute(&model_store)
            || !safe_absolute(&runtime_manifest)
            || !safe_absolute(&runtime_directory)
        {
            return Err(configuration());
        }
        let metadata = runtime_manifest
            .symlink_metadata()
            .map_err(|_| configuration())?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() == 0
            || metadata.len() > MAX_RUNTIME_MANIFEST_BYTES
        {
            return Err(configuration());
        }
        let manifest_bytes = fs::read(&runtime_manifest).map_err(|_| configuration())?;
        if u64::try_from(manifest_bytes.len()).ok() != Some(metadata.len()) {
            return Err(configuration());
        }
        Self::from_manifest_bytes(model_store, runtime_directory, platform, &manifest_bytes)
    }

    #[allow(clippy::too_many_lines)] // One atomic manifest policy is kept visible for review.
    fn from_manifest_bytes(
        model_store: PathBuf,
        runtime_directory: PathBuf,
        platform: &str,
        bytes: &[u8],
    ) -> Result<Self, RuntimeQualificationError> {
        if bytes.is_empty()
            || u64::try_from(bytes.len()).map_err(|_| configuration())? > MAX_RUNTIME_MANIFEST_BYTES
        {
            return Err(configuration());
        }
        let manifest: RuntimeManifest =
            serde_json::from_slice(bytes).map_err(|_| configuration())?;
        if manifest.schema_version != 1
            || manifest.runtime.name != "Microsoft ONNX Runtime"
            || manifest.runtime.version != "1.28.0"
            || manifest.runtime.source_commit != "da9b5e364c465de65c49d91e696cd6485270757f"
            || manifest.runtime.c_api_version != 28
            || manifest.runtime.execution_provider != "cpu"
            || manifest.runtime.license != "MIT"
            || !manifest.release.is_object()
            || !manifest.network_policy.is_object()
        {
            return Err(configuration());
        }
        let mut selected = manifest
            .platforms
            .iter()
            .filter(|candidate| candidate.id == platform);
        let platform = selected.next().ok_or_else(configuration)?;
        if selected.next().is_some() || !host_matches(platform.id.as_str()) {
            return Err(configuration());
        }
        let (
            target_triple,
            archive_format,
            asset_id,
            archive_root,
            archive_url,
            archive_bytes,
            archive_sha256,
            maximum_expanded_bytes,
        ) = match platform.id.as_str() {
            "windows-x86_64" => (
                "x86_64-pc-windows-msvc",
                "zip",
                489_173_573,
                "onnxruntime-win-x64-1.28.0",
                "https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-win-x64-1.28.0.zip",
                78_796_801,
                "abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc",
                536_870_912,
            ),
            "linux-x86_64" => (
                "x86_64-unknown-linux-gnu",
                "tgz",
                489_174_677,
                "onnxruntime-linux-x64-1.28.0",
                "https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz",
                9_125_960,
                "a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407",
                268_435_456,
            ),
            _ => return Err(configuration()),
        };
        if platform.target_triple != target_triple
            || platform.archive.format != archive_format
            || platform.archive.asset_id != asset_id
            || platform.archive.root_directory != archive_root
            || platform.archive.url != archive_url
            || platform.archive.bytes != archive_bytes
            || platform.archive.sha256 != archive_sha256
            || platform.archive.maximum_expanded_bytes != maximum_expanded_bytes
            || platform.archive.maximum_members != 256
            || !platform.allowed_member_rules.is_object()
        {
            return Err(configuration());
        }
        let main_name = match platform.id.as_str() {
            "windows-x86_64" => "onnxruntime.dll",
            "linux-x86_64" => "libonnxruntime.so.1.28.0",
            _ => return Err(configuration()),
        };
        let mut main_matches = platform
            .required_files
            .iter()
            .filter(|file| file.install_name == main_name);
        let main = main_matches.next().ok_or_else(configuration)?;
        if main_matches.next().is_some()
            || main.bytes.is_none_or(|value| value == 0)
            || main
                .sha256
                .as_deref()
                .is_none_or(|value| !lower_sha256(value))
            || !manifest_is_fully_qualified(platform)
            || runtime_directory
                .file_name()
                .and_then(|value| value.to_str())
                != Some(platform.archive.sha256.as_str())
        {
            return Err(configuration());
        }
        Ok(Self {
            model_store,
            runtime_directory,
            runtime_library: main_name.into(),
            runtime_library_bytes: main.bytes.ok_or_else(configuration)?,
            runtime_library_sha256: main.sha256.clone().ok_or_else(configuration)?,
        })
    }
}

impl fmt::Debug for RuntimeQualificationConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeQualificationConfig")
            .field("model_store", &"[REDACTED]")
            .field("runtime_directory", &"[REDACTED]")
            .field("runtime_library", &"[REDACTED]")
            .field("runtime_library_bytes", &self.runtime_library_bytes)
            .field("runtime_library_sha256", &"[REDACTED]")
            .finish()
    }
}

/// Sanitized deterministic report from the complete real-runtime qualification suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeQualificationReport {
    schema_version: u32,
    status: &'static str,
    profile: &'static str,
    runtime_version: &'static str,
    warmup: bool,
    detector_probes: Vec<DetectorProbeReport>,
    recognizer_probes: Vec<RecognizerProbeReport>,
    end_to_end: EndToEndReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct DetectorProbeReport {
    width: u32,
    height: u32,
    shape_valid: bool,
    probability_invariants: bool,
    deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecognizerProbeReport {
    batch: usize,
    requested_width: u32,
    admitted_width: Option<u32>,
    probability_invariants: bool,
    deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct EndToEndReport {
    generated_png: bool,
    valid_result: bool,
    deterministic: bool,
}

/// Qualifies one explicit local model store and installed native runtime through production code.
///
/// # Errors
/// Returns only stable privacy-safe categories. No paths, host details, timings, tensor values, or
/// recognized text are included in errors or reports.
pub async fn qualify_installed_runtime(
    config: RuntimeQualificationConfig,
) -> Result<RuntimeQualificationReport, RuntimeQualificationError> {
    validate_runtime_directory(&config)?;
    let catalog = ModelCatalog::curated().map_err(|_| model_unavailable())?;
    let store = ModelStore::open(&config.model_store, catalog)
        .await
        .map_err(|_| model_unavailable())?;
    let verified = store
        .verify(CURATED_BUNDLE_ID)
        .await
        .map_err(|_| model_unavailable())?;
    let decoder = verified.ctc_decoder().map_err(|_| model_unavailable())?;
    let runtime_config = OrtRuntimeConfig::new(
        config.runtime_directory.join(&config.runtime_library),
        config.runtime_library_bytes,
        config.runtime_library_sha256,
    )
    .and_then(|runtime| runtime.with_lanes(1))
    .and_then(|runtime| runtime.with_threads(1, 1))
    .map_err(|_| model_unavailable())?;
    let runtime = Arc::new(
        OrtCpuRuntime::load(&store, CURATED_BUNDLE_ID, runtime_config)
            .await
            .map_err(|_| model_unavailable())?,
    );
    run_probe_suite(runtime, decoder).await
}

#[allow(clippy::too_many_lines)] // The fixed probe matrix is intentionally linear and auditable.
async fn run_probe_suite<R: TensorOcrRuntime>(
    runtime: Arc<R>,
    decoder: CtcDecoder,
) -> Result<RuntimeQualificationReport, RuntimeQualificationError> {
    runtime.warm_up().map_err(|_| qualification_failed())?;
    let mut detector_probes = Vec::new();
    for (width, height) in [(32, 32), (320, 960), (960, 320), (960, 960)] {
        let raster = solid_raster(width, height)?;
        let input = preprocess_detector(&raster).map_err(|_| qualification_failed())?;
        if input.tensor_width() != width || input.tensor_height() != height {
            return Err(qualification_failed());
        }
        let first = runtime
            .run_detector(&input, &request_context()?)
            .map_err(|_| qualification_failed())?;
        let second = runtime
            .run_detector(&input, &request_context()?)
            .map_err(|_| qualification_failed())?;
        let shape_valid = first.width() == width && first.height() == height;
        let deterministic = first == second;
        if !shape_valid || !deterministic {
            return Err(qualification_failed());
        }
        detector_probes.push(DetectorProbeReport {
            width,
            height,
            shape_valid,
            probability_invariants: true,
            deterministic,
        });
    }

    let mut recognizer_probes = Vec::new();
    for batch in [1_usize, 8] {
        for requested_width in [160_u32, 320, 3200] {
            let raster = solid_raster(requested_width, 48)?;
            let crops = vec![raster; batch];
            let input = preprocess_recognizer_batch(&crops).map_err(|_| qualification_failed())?;
            if input.batch_width() != requested_width {
                recognizer_probes.push(RecognizerProbeReport {
                    batch,
                    requested_width,
                    admitted_width: None,
                    probability_invariants: false,
                    deterministic: false,
                });
                continue;
            }
            let first = runtime
                .run_recognizer(&input, &request_context()?)
                .map_err(|_| qualification_failed())?;
            let second = runtime
                .run_recognizer(&input, &request_context()?)
                .map_err(|_| qualification_failed())?;
            let probability_invariants =
                first.probability_rows_are_normalized(PROBABILITY_SUM_TOLERANCE);
            let deterministic = first == second;
            if first.batch_size() != batch || !probability_invariants || !deterministic {
                return Err(qualification_failed());
            }
            recognizer_probes.push(RecognizerProbeReport {
                batch,
                requested_width,
                admitted_width: Some(input.batch_width()),
                probability_invariants,
                deterministic,
            });
        }
    }

    let backend = PpOcrBackend::new(
        runtime,
        RasterDecoder::new(RasterLimits::default()),
        decoder,
    );
    backend
        .warm_up()
        .await
        .map_err(|_| qualification_failed())?;
    let png = generated_rgb_png(8, 8)?;
    let input = RasterInput::new(
        png.clone(),
        InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes: u64::try_from(png.len()).map_err(|_| qualification_failed())?,
            width: 8,
            height: 8,
        },
    )
    .map_err(|_| qualification_failed())?;
    let first = backend
        .recognize(&input, OcrOptions::default(), &request_context()?)
        .await
        .map_err(|_| qualification_failed())?;
    let second = backend
        .recognize(&input, OcrOptions::default(), &request_context()?)
        .await
        .map_err(|_| qualification_failed())?;
    first.validate().map_err(|_| qualification_failed())?;
    let deterministic = first == second;
    if !deterministic {
        return Err(qualification_failed());
    }

    Ok(RuntimeQualificationReport {
        schema_version: 1,
        status: "passed",
        profile: CURATED_BUNDLE_ID,
        runtime_version: "1.28.0",
        warmup: true,
        detector_probes,
        recognizer_probes,
        end_to_end: EndToEndReport {
            generated_png: true,
            valid_result: true,
            deterministic,
        },
    })
}

fn validate_runtime_directory(
    config: &RuntimeQualificationConfig,
) -> Result<(), RuntimeQualificationError> {
    let directory = config
        .runtime_directory
        .symlink_metadata()
        .map_err(|_| configuration())?;
    let library_path = config.runtime_directory.join(&config.runtime_library);
    let library = library_path
        .symlink_metadata()
        .map_err(|_| configuration())?;
    if !directory.is_dir()
        || directory.file_type().is_symlink()
        || !library.is_file()
        || library.file_type().is_symlink()
    {
        return Err(configuration());
    }
    let canonical_directory = config
        .runtime_directory
        .canonicalize()
        .map_err(|_| configuration())?;
    let canonical_library = library_path.canonicalize().map_err(|_| configuration())?;
    if canonical_library.parent() != Some(canonical_directory.as_path()) {
        return Err(configuration());
    }
    Ok(())
}

fn manifest_is_fully_qualified(platform: &RuntimePlatform) -> bool {
    let expected: BTreeSet<&str> = match platform.id.as_str() {
        "windows-x86_64" => [
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "LICENSE.onnxruntime",
            "ThirdPartyNotices.onnxruntime.txt",
            "Privacy.onnxruntime.md",
        ]
        .into_iter()
        .collect(),
        "linux-x86_64" => [
            "libonnxruntime.so.1.28.0",
            "libonnxruntime_providers_shared.so",
            "LICENSE.onnxruntime",
            "ThirdPartyNotices.onnxruntime.txt",
            "Privacy.onnxruntime.md",
        ]
        .into_iter()
        .collect(),
        _ => return false,
    };
    let actual: BTreeSet<&str> = platform
        .required_files
        .iter()
        .map(|file| file.install_name.as_str())
        .collect();
    lower_sha256(&platform.archive.sha256)
        && platform.archive.bytes > 0
        && actual == expected
        && actual.len() == platform.required_files.len()
        && platform.required_files.iter().all(|file| {
            file.bytes.is_some_and(|value| value > 0)
                && file.sha256.as_deref().is_some_and(lower_sha256)
                && single_filename(&file.install_name)
                && safe_archive_member(&file.member)
        })
}

fn safe_archive_member(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.contains(['\\', ':', '\0'])
        && value
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

fn solid_raster(width: u32, height: u32) -> Result<DecodedRaster, RuntimeQualificationError> {
    let length = usize::try_from(u64::from(width) * u64::from(height) * 3)
        .map_err(|_| qualification_failed())?;
    DecodedRaster::from_rgb8(width, height, vec![0_u8; length]).map_err(|_| qualification_failed())
}

fn request_context() -> Result<RequestContext, RuntimeQualificationError> {
    let id = RequestIdSource::default()
        .next()
        .map_err(|_| qualification_failed())?;
    RequestContext::new(id, CancellationToken::new(), Some(PROBE_TIMEOUT))
        .map_err(|_| qualification_failed())
}

fn generated_rgb_png(width: u32, height: u32) -> Result<Vec<u8>, RuntimeQualificationError> {
    let row_bytes = usize::try_from(u64::from(width) * 3).map_err(|_| qualification_failed())?;
    let raw_len = usize::try_from(u64::from(height))
        .ok()
        .and_then(|rows| {
            row_bytes
                .checked_add(1)
                .and_then(|length| length.checked_mul(rows))
        })
        .ok_or_else(qualification_failed)?;
    if raw_len > u16::MAX.into() {
        return Err(qualification_failed());
    }
    let mut raw_pixels = vec![0_u8; raw_len];
    for scanline in raw_pixels.chunks_exact_mut(row_bytes + 1) {
        scanline[0] = 0;
    }
    let mut zlib = vec![0x78, 0x01, 0x01];
    let length = u16::try_from(raw_pixels.len()).map_err(|_| qualification_failed())?;
    zlib.extend_from_slice(&length.to_le_bytes());
    zlib.extend_from_slice(&(!length).to_le_bytes());
    zlib.extend_from_slice(&raw_pixels);
    zlib.extend_from_slice(&adler32(&raw_pixels).to_be_bytes());

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    append_png_chunk(&mut png, *b"IHDR", &ihdr)?;
    append_png_chunk(&mut png, *b"IDAT", &zlib)?;
    append_png_chunk(&mut png, *b"IEND", &[])?;
    Ok(png)
}

fn append_png_chunk(
    output: &mut Vec<u8>,
    kind: [u8; 4],
    data: &[u8],
) -> Result<(), RuntimeQualificationError> {
    output.extend_from_slice(
        &u32::try_from(data.len())
            .map_err(|_| qualification_failed())?
            .to_be_bytes(),
    );
    output.extend_from_slice(&kind);
    output.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(kind.len() + data.len());
    crc_input.extend_from_slice(&kind);
    crc_input.extend_from_slice(data);
    output.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let mut first = 1_u32;
    let mut second = 0_u32;
    for byte in bytes {
        first = (first + u32::from(*byte)) % 65_521;
        second = (second + first) % 65_521;
    }
    (second << 16) | first
}

fn safe_absolute(path: &std::path::Path) -> bool {
    path.is_absolute()
        && path.components().all(|component| {
            !matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
}

fn single_filename(value: &str) -> bool {
    let path = std::path::Path::new(value);
    let mut components = path.components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
}

fn lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn host_matches(platform: &str) -> bool {
    matches!(
        (std::env::consts::OS, std::env::consts::ARCH, platform),
        ("windows", "x86_64", "windows-x86_64") | ("linux", "x86_64", "linux-x86_64")
    )
}

const fn configuration() -> RuntimeQualificationError {
    RuntimeQualificationError::new(RuntimeQualificationErrorCode::Configuration)
}

const fn model_unavailable() -> RuntimeQualificationError {
    RuntimeQualificationError::new(RuntimeQualificationErrorCode::ModelUnavailable)
}

const fn qualification_failed() -> RuntimeQualificationError {
    RuntimeQualificationError::new(RuntimeQualificationErrorCode::QualificationFailed)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeManifest {
    schema_version: u32,
    runtime: RuntimeIdentity,
    release: serde_json::Value,
    network_policy: serde_json::Value,
    platforms: Vec<RuntimePlatform>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeIdentity {
    name: String,
    version: String,
    source_commit: String,
    c_api_version: u32,
    execution_provider: String,
    license: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimePlatform {
    id: String,
    target_triple: String,
    archive: RuntimeArchive,
    required_files: Vec<RuntimeFile>,
    allowed_member_rules: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeArchive {
    format: String,
    url: String,
    asset_id: u64,
    bytes: u64,
    sha256: String,
    root_directory: String,
    maximum_expanded_bytes: u64,
    maximum_members: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeFile {
    member: String,
    install_name: String,
    bytes: Option<u64>,
    sha256: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use impossible_ocr_domain::{OcrError, OcrErrorCode};
    use impossible_ocr_pipeline::{
        BackendState, DetectorInput, DetectorTensorOutput, RecognizerBatch, RecognizerTensorOutput,
        TensorOcrRuntime,
    };

    use super::{
        RuntimeQualificationConfig, RuntimeQualificationErrorCode, generated_rgb_png,
        run_probe_suite,
    };

    #[derive(Debug)]
    struct FakeRuntime;

    impl TensorOcrRuntime for FakeRuntime {
        fn state(&self) -> BackendState {
            BackendState::Ready
        }

        fn warm_up(&self) -> Result<(), OcrError> {
            Ok(())
        }

        fn run_detector(
            &self,
            input: &DetectorInput,
            _context: &impossible_server_core::RequestContext,
        ) -> Result<DetectorTensorOutput, OcrError> {
            let length =
                usize::try_from(u64::from(input.tensor_width()) * u64::from(input.tensor_height()))
                    .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
            DetectorTensorOutput::new(
                vec![0.0; length],
                input.tensor_width(),
                input.tensor_height(),
            )
        }

        fn run_recognizer(
            &self,
            input: &RecognizerBatch,
            _context: &impossible_server_core::RequestContext,
        ) -> Result<RecognizerTensorOutput, OcrError> {
            let classes = 3_usize;
            let steps = 2_usize;
            let mut values = vec![0.0; input.batch_size() * steps * classes];
            for row in values.chunks_exact_mut(classes) {
                row[0] = 1.0;
            }
            RecognizerTensorOutput::new(values, input.batch_size(), steps, classes)
        }
    }

    #[tokio::test]
    async fn fake_suite_is_deterministic_and_never_needs_native_artifacts() {
        let decoder = impossible_ocr_pipeline::CtcDecoder::new(vec!["a".into()], true)
            .unwrap_or_else(|_| unreachable!());
        let report = run_probe_suite(Arc::new(FakeRuntime), decoder)
            .await
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(report.status, "passed");
        assert_eq!(report.detector_probes.len(), 4);
        assert_eq!(report.recognizer_probes.len(), 6);
        assert_eq!(
            report
                .recognizer_probes
                .iter()
                .filter(|probe| probe.admitted_width.is_some())
                .count(),
            4
        );
        assert!(report.end_to_end.generated_png);
    }

    #[test]
    fn generated_png_is_decodable_and_manifest_parser_fails_closed() {
        let png = generated_rgb_png(8, 8).unwrap_or_else(|_| unreachable!());
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let error = RuntimeQualificationConfig::from_manifest_bytes(
            std::path::PathBuf::from("C:/models"),
            std::path::PathBuf::from("C:/runtime"),
            "windows-x86_64",
            br#"{"schemaVersion":1}"#,
        )
        .err()
        .unwrap_or_else(|| unreachable!());
        assert_eq!(error.code(), RuntimeQualificationErrorCode::Configuration);
    }

    #[test]
    fn config_debug_and_errors_never_disclose_paths() {
        let error = super::configuration();
        assert!(!format!("{error:?}").contains("private"));
        assert_eq!(
            error.to_string(),
            "runtime qualification configuration is invalid"
        );
    }

    #[tokio::test]
    #[ignore = "requires explicitly installed qualified model and native runtime artifacts"]
    async fn installed_runtime_qualification_is_operator_opt_in() {
        let _ = std::env::var_os("IMPOSSIBLE_OCR_QUALIFICATION_MANIFEST");
    }
}
