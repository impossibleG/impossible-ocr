//! Integrity-gated CPU ONNX Runtime adapter.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    ops::{Deref, DerefMut},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use impossible_ocr_domain::{OcrError, OcrErrorCode};
use impossible_ocr_pipeline::{
    BackendState, DetectorInput, DetectorTensorOutput, RecognizerBatch, RecognizerTensorOutput,
    TensorOcrRuntime,
};
use impossible_server_core::RequestContext;
use ort::{
    ep::CPU,
    logging::{LogLevel, LoggerFunction},
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::{Outlet, Tensor, TensorElementType, ValueType},
};
use sha2::{Digest, Sha256};

use crate::{
    ArtifactKind, DimensionContract, MAX_ONNX_MODEL_BYTES, ModelContract, ModelLease, ModelRole,
    ModelStore, TensorContract, admit_onnx_model, detector_contract, english_recognizer_contract,
};

const MAX_LANES: usize = 32;
const MAX_THREADS_PER_SESSION: usize = 64;
const MAX_NATIVE_LIBRARY_BYTES: u64 = 512 * 1024 * 1024;
const MONITOR_INTERVAL: Duration = Duration::from_millis(2);
const DETECTOR_WARMUP_SIDE: usize = 32;
const RECOGNIZER_WARMUP_WIDTH: usize = 320;

static ORT_INIT_LOCK: Mutex<Option<NativeRuntimeState>> = Mutex::new(None);
static NATIVE_STAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

const NATIVE_CACHE_DIRECTORY: &str = ".impossible-ocr-native";
const NATIVE_OWNER_MARKER: &str = ".impossible-ocr-owner";

/// Explicit, integrity-bound configuration for the dynamically loaded CPU runtime.
#[derive(Clone)]
pub struct OrtRuntimeConfig {
    native_library: PathBuf,
    native_library_bytes: u64,
    native_library_sha256: String,
    lanes: usize,
    intra_threads: usize,
    inter_threads: usize,
}

impl OrtRuntimeConfig {
    /// Creates a configuration which names one exact native ONNX Runtime library.
    ///
    /// The path must be absolute. The byte length and lowercase SHA-256 are verified before and
    /// after dynamic loading. Nothing is searched on `PATH` and nothing is downloaded.
    ///
    /// # Errors
    /// Returns a sanitized model-unavailable error for an invalid path, size, digest, or limit.
    pub fn new(
        native_library: impl Into<PathBuf>,
        native_library_bytes: u64,
        native_library_sha256: impl Into<String>,
    ) -> Result<Self, OcrError> {
        let config = Self {
            native_library: native_library.into(),
            native_library_bytes,
            native_library_sha256: native_library_sha256.into(),
            lanes: 1,
            intra_threads: 1,
            inter_threads: 1,
        };
        config.validate()?;
        Ok(config)
    }

    /// Sets the fixed number of independent detector/recognizer session lanes.
    ///
    /// # Errors
    /// Returns a sanitized model-unavailable error unless `lanes` is between one and 32.
    pub fn with_lanes(mut self, lanes: usize) -> Result<Self, OcrError> {
        self.lanes = lanes;
        self.validate()?;
        Ok(self)
    }

    /// Sets deterministic per-session intra-op and inter-op thread limits.
    ///
    /// # Errors
    /// Returns a sanitized model-unavailable error unless both values are between one and 64.
    pub fn with_threads(
        mut self,
        intra_threads: usize,
        inter_threads: usize,
    ) -> Result<Self, OcrError> {
        self.intra_threads = intra_threads;
        self.inter_threads = inter_threads;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), OcrError> {
        if !self.native_library.is_absolute()
            || self.native_library_bytes == 0
            || self.native_library_bytes > MAX_NATIVE_LIBRARY_BYTES
            || !is_lower_sha256(&self.native_library_sha256)
            || !(1..=MAX_LANES).contains(&self.lanes)
            || !(1..=MAX_THREADS_PER_SESSION).contains(&self.intra_threads)
            || !(1..=MAX_THREADS_PER_SESSION).contains(&self.inter_threads)
        {
            return Err(model_unavailable());
        }
        Ok(())
    }
}

impl fmt::Debug for OrtRuntimeConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OrtRuntimeConfig")
            .field("native_library", &"[REDACTED]")
            .field("native_library_bytes", &self.native_library_bytes)
            .field("native_library_sha256", &"[REDACTED]")
            .field("lanes", &self.lanes)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .finish()
    }
}

/// Qualified CPU runtime with a bounded pool of independent session pairs.
pub struct OrtCpuRuntime {
    lanes: LanePool<OrtLane>,
    recognizer_class_count: usize,
}

impl fmt::Debug for OrtCpuRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OrtCpuRuntime")
            .field("sessions", &"[REDACTED]")
            .field("recognizer_class_count", &self.recognizer_class_count)
            .finish_non_exhaustive()
    }
}

impl OrtCpuRuntime {
    /// Verifies a complete model bundle, admits both graphs, explicitly loads the exact native
    /// library, creates fixed CPU-only sessions, and warms every lane.
    ///
    /// No network operation is performed. The checked-in graph contracts are qualified against the
    /// exact pinned model lengths, digests, and measured executable graph metadata before a native
    /// library can be loaded.
    ///
    /// # Errors
    /// Returns a fixed privacy-safe OCR error for configuration, integrity, admission, load,
    /// session-metadata, or warm-up failures.
    pub async fn load(
        store: &ModelStore,
        bundle_id: &str,
        config: OrtRuntimeConfig,
    ) -> Result<Self, OcrError> {
        config.validate()?;
        let verified = store
            .verify(bundle_id)
            .await
            .map_err(|_| model_unavailable())?;
        let lease = Arc::new(
            store
                .lease(bundle_id)
                .await
                .map_err(|_| model_unavailable())?,
        );
        let detector_path = artifact_path(&lease, ModelRole::Detector)?;
        let recognizer_path = artifact_path(&lease, ModelRole::EnglishRecognizer)?;
        let detector_contract = detector_contract().map_err(|_| model_unavailable())?;
        let recognizer_contract = english_recognizer_contract().map_err(|_| model_unavailable())?;

        let detector_bytes: Arc<[u8]> = admit_file(&detector_path, &detector_contract)?.into();
        let recognizer_bytes: Arc<[u8]> =
            admit_file(&recognizer_path, &recognizer_contract)?.into();
        let native_path = initialize_native_runtime(&config)?;

        let mut lanes = Vec::new();
        lanes
            .try_reserve_exact(config.lanes)
            .map_err(|_| internal())?;
        for _ in 0..config.lanes {
            let detector = create_session(&detector_bytes, &config)?;
            let recognizer = create_session(&recognizer_bytes, &config)?;
            validate_session_contract(&detector, &detector_contract)?;
            validate_session_contract(&recognizer, &recognizer_contract)?;
            let mut lane = OrtLane {
                detector,
                recognizer,
                _lease: Arc::clone(&lease),
            };
            warm_lane(&mut lane, verified.recognizer_class_count)?;
            lanes.push(lane);
        }

        verify_native_library(&native_path, &config)?;
        Ok(Self {
            lanes: LanePool::new(lanes)?,
            recognizer_class_count: verified.recognizer_class_count,
        })
    }
}

impl TensorOcrRuntime for OrtCpuRuntime {
    fn state(&self) -> BackendState {
        BackendState::Ready
    }

    fn warm_up(&self) -> Result<(), OcrError> {
        Ok(())
    }

    fn run_detector(
        &self,
        input: &DetectorInput,
        context: &RequestContext,
    ) -> Result<DetectorTensorOutput, OcrError> {
        let mut lane = self.lanes.acquire(context)?;
        let height = usize::try_from(input.tensor_height()).map_err(|_| internal())?;
        let width = usize::try_from(input.tensor_width()).map_err(|_| internal())?;
        let tensor = catch_internal(|| {
            Tensor::from_array((
                [1_usize, 3, height, width],
                input.data().to_vec().into_boxed_slice(),
            ))
            .map_err(|_| internal())
        })?;
        let (shape, data) = run_session(&mut lane.detector, tensor, context)?;
        if shape
            != [
                1,
                1,
                i64::from(input.tensor_height()),
                i64::from(input.tensor_width()),
            ]
        {
            return Err(internal());
        }
        DetectorTensorOutput::new(data, input.tensor_width(), input.tensor_height())
    }

    fn run_recognizer(
        &self,
        input: &RecognizerBatch,
        context: &RequestContext,
    ) -> Result<RecognizerTensorOutput, OcrError> {
        let mut lane = self.lanes.acquire(context)?;
        let batch = input.batch_size();
        let width = usize::try_from(input.batch_width()).map_err(|_| internal())?;
        let tensor = catch_internal(|| {
            Tensor::from_array((
                [batch, 3, 48, width],
                input.data().to_vec().into_boxed_slice(),
            ))
            .map_err(|_| internal())
        })?;
        let (shape, data) = run_session(&mut lane.recognizer, tensor, context)?;
        let [output_batch, time_steps, classes] = shape.as_slice() else {
            return Err(internal());
        };
        let output_batch = positive_usize(*output_batch)?;
        let time_steps = positive_usize(*time_steps)?;
        let classes = positive_usize(*classes)?;
        if output_batch != batch || classes != self.recognizer_class_count {
            return Err(internal());
        }
        RecognizerTensorOutput::new(data, output_batch, time_steps, classes)
    }
}

struct OrtLane {
    detector: Session,
    recognizer: Session,
    _lease: Arc<ModelLease>,
}

#[derive(Clone, PartialEq, Eq)]
struct NativeIdentity {
    path: PathBuf,
    byte_length: u64,
    sha256: String,
}

struct NativeRuntimeState {
    identity: NativeIdentity,
    committed: bool,
    guard: File,
}

fn initialize_native_runtime(config: &OrtRuntimeConfig) -> Result<PathBuf, OcrError> {
    let prepared = prepare_owned_native_runtime(config)?;
    let identity = NativeIdentity {
        path: prepared.path.clone(),
        byte_length: config.native_library_bytes,
        sha256: config.native_library_sha256.clone(),
    };
    let mut initialized = ORT_INIT_LOCK.lock().map_err(|_| internal())?;
    if let Some(existing) = initialized.as_ref() {
        if existing.identity != identity || !existing.committed {
            return Err(model_unavailable());
        }
        return Ok(existing.identity.path.clone());
    }

    *initialized = Some(NativeRuntimeState {
        identity: identity.clone(),
        committed: false,
        guard: prepared.guard,
    });
    let state = initialized.as_ref().ok_or_else(model_unavailable)?;
    let loader_path = native_loader_path(&identity.path, state);
    let committed = catch_unwind(AssertUnwindSafe(|| {
        ort::init_from(&loader_path)
            .map(|builder| {
                builder
                    .with_name("impossible-ocr")
                    .with_telemetry(false)
                    .with_logger(silent_logger())
                    .commit()
            })
            .map_err(|_| model_unavailable())
    }))
    .map_err(|_| model_unavailable())??;
    if !committed {
        return Err(model_unavailable());
    }
    let state = initialized.as_ref().ok_or_else(model_unavailable)?;
    verify_open_native_library(&state.guard, config)?;
    if let Some(state) = initialized.as_mut() {
        state.committed = true;
    }
    Ok(identity.path)
}

struct PreparedNativeRuntime {
    path: PathBuf,
    guard: File,
}

fn prepare_owned_native_runtime(
    config: &OrtRuntimeConfig,
) -> Result<PreparedNativeRuntime, OcrError> {
    let source = config
        .native_library
        .canonicalize()
        .map_err(|_| model_unavailable())?;
    verify_native_library(&source, config)?;
    let filename = source
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| {
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
        })
        .ok_or_else(model_unavailable)?;
    let parent = source.parent().ok_or_else(model_unavailable)?;
    let cache_root = parent.join(NATIVE_CACHE_DIRECTORY);
    ensure_real_directory(&cache_root)?;
    let canonical_root = cache_root.canonicalize().map_err(|_| model_unavailable())?;
    if canonical_root != cache_root {
        return Err(model_unavailable());
    }

    let digest_directory = canonical_root.join(&config.native_library_sha256);
    let final_path = digest_directory.join(filename);
    if digest_directory.exists() {
        return open_owned_native(&digest_directory, &final_path, filename, config);
    }

    let sequence = NATIVE_STAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging = canonical_root.join(format!(
        ".{}.stage-{}-{sequence}",
        config.native_library_sha256,
        std::process::id()
    ));
    fs::create_dir(&staging).map_err(|_| model_unavailable())?;
    let staged_file = staging.join(filename);
    let result = stage_native_file(&source, &staged_file, filename, config)
        .and_then(|()| fs::rename(&staging, &digest_directory).map_err(|_| model_unavailable()));
    if result.is_err() {
        if digest_directory.exists() {
            let _ = fs::remove_dir_all(&staging);
            return open_owned_native(&digest_directory, &final_path, filename, config);
        }
        let _ = fs::remove_dir_all(&staging);
        return Err(model_unavailable());
    }
    open_owned_native(&digest_directory, &final_path, filename, config)
}

fn ensure_real_directory(path: &Path) -> Result<(), OcrError> {
    match path.symlink_metadata() {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path).map_err(|_| model_unavailable())
        }
        Err(_) | Ok(_) => Err(model_unavailable()),
    }
}

fn stage_native_file(
    source: &Path,
    destination: &Path,
    filename: &str,
    config: &OrtRuntimeConfig,
) -> Result<(), OcrError> {
    let mut input = File::open(source).map_err(|_| model_unavailable())?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|_| model_unavailable())?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let count = input.read(&mut buffer).map_err(|_| model_unavailable())?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| model_unavailable())?)
            .ok_or_else(model_unavailable)?;
        if total > config.native_library_bytes {
            return Err(model_unavailable());
        }
        output
            .write_all(&buffer[..count])
            .map_err(|_| model_unavailable())?;
        hasher.update(&buffer[..count]);
    }
    if total != config.native_library_bytes
        || format!("{:x}", hasher.finalize()) != config.native_library_sha256
    {
        return Err(model_unavailable());
    }
    output.sync_all().map_err(|_| model_unavailable())?;
    let mut permissions = output
        .metadata()
        .map_err(|_| model_unavailable())?
        .permissions();
    permissions.set_readonly(true);
    output
        .set_permissions(permissions)
        .map_err(|_| model_unavailable())?;
    drop(output);

    let marker = destination
        .parent()
        .ok_or_else(model_unavailable)?
        .join(NATIVE_OWNER_MARKER);
    let mut marker_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(marker)
        .map_err(|_| model_unavailable())?;
    write!(
        marker_file,
        "schema=1\nsha256={}\nbytes={}\nfilename={}\n",
        config.native_library_sha256, config.native_library_bytes, filename
    )
    .map_err(|_| model_unavailable())?;
    marker_file.sync_all().map_err(|_| model_unavailable())?;
    Ok(())
}

fn open_owned_native(
    directory: &Path,
    path: &Path,
    filename: &str,
    config: &OrtRuntimeConfig,
) -> Result<PreparedNativeRuntime, OcrError> {
    let metadata = directory
        .symlink_metadata()
        .map_err(|_| model_unavailable())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(model_unavailable());
    }
    let mut entries = fs::read_dir(directory)
        .map_err(|_| model_unavailable())?
        .map(|entry| {
            entry
                .map(|value| value.file_name())
                .map_err(|_| model_unavailable())
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    let mut expected = vec![
        std::ffi::OsString::from(filename),
        std::ffi::OsString::from(NATIVE_OWNER_MARKER),
    ];
    expected.sort();
    if entries != expected {
        return Err(model_unavailable());
    }
    let marker_path = directory.join(NATIVE_OWNER_MARKER);
    let marker_metadata = marker_path
        .symlink_metadata()
        .map_err(|_| model_unavailable())?;
    let file_metadata = path.symlink_metadata().map_err(|_| model_unavailable())?;
    if !marker_metadata.is_file()
        || marker_metadata.file_type().is_symlink()
        || !file_metadata.is_file()
        || file_metadata.file_type().is_symlink()
        || !native_file_is_readonly(&file_metadata)
    {
        return Err(model_unavailable());
    }
    let marker = fs::read_to_string(marker_path).map_err(|_| model_unavailable())?;
    let expected_marker = format!(
        "schema=1\nsha256={}\nbytes={}\nfilename={}\n",
        config.native_library_sha256, config.native_library_bytes, filename
    );
    if marker != expected_marker {
        return Err(model_unavailable());
    }
    let guard = open_protected_native(path)?;
    verify_open_native_library(&guard, config)?;
    Ok(PreparedNativeRuntime {
        path: path.to_path_buf(),
        guard,
    })
}

#[cfg(windows)]
fn native_file_is_readonly(metadata: &fs::Metadata) -> bool {
    metadata.permissions().readonly()
}

#[cfg(unix)]
fn native_file_is_readonly(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o222 == 0
}

#[cfg(not(any(unix, windows)))]
fn native_file_is_readonly(_metadata: &fs::Metadata) -> bool {
    false
}

#[cfg(windows)]
fn open_protected_native(path: &Path) -> Result<File, OcrError> {
    use std::os::windows::fs::OpenOptionsExt;

    // Share reads with the dynamic loader, while denying replacement, deletion, and writes for
    // the entire process lifetime.
    OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)
        .map_err(|_| model_unavailable())
}

#[cfg(not(windows))]
fn open_protected_native(path: &Path) -> Result<File, OcrError> {
    File::open(path).map_err(|_| model_unavailable())
}

#[cfg(target_os = "linux")]
fn native_loader_path(_path: &Path, state: &NativeRuntimeState) -> PathBuf {
    use std::os::fd::AsRawFd;

    PathBuf::from(format!("/proc/self/fd/{}", state.guard.as_raw_fd()))
}

#[cfg(not(target_os = "linux"))]
fn native_loader_path(path: &Path, _state: &NativeRuntimeState) -> PathBuf {
    path.to_path_buf()
}

fn verify_native_library(path: &Path, config: &OrtRuntimeConfig) -> Result<(), OcrError> {
    if !path.is_absolute() {
        return Err(model_unavailable());
    }
    let metadata = path.symlink_metadata().map_err(|_| model_unavailable())?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() != config.native_library_bytes
    {
        return Err(model_unavailable());
    }
    let digest = digest_exact_file(path, config.native_library_bytes)?;
    if digest != config.native_library_sha256 {
        return Err(model_unavailable());
    }
    Ok(())
}

fn verify_open_native_library(file: &File, config: &OrtRuntimeConfig) -> Result<(), OcrError> {
    let metadata = file.metadata().map_err(|_| model_unavailable())?;
    if !metadata.is_file() || metadata.len() != config.native_library_bytes {
        return Err(model_unavailable());
    }
    let digest = digest_open_file(file, config.native_library_bytes)?;
    if digest != config.native_library_sha256 {
        return Err(model_unavailable());
    }
    Ok(())
}

fn digest_open_file(file: &File, expected_bytes: u64) -> Result<String, OcrError> {
    let mut file = file.try_clone().map_err(|_| model_unavailable())?;
    file.seek(SeekFrom::Start(0))
        .map_err(|_| model_unavailable())?;
    digest_reader(&mut file, expected_bytes)
}

fn digest_exact_file(path: &Path, expected_bytes: u64) -> Result<String, OcrError> {
    let mut file = File::open(path).map_err(|_| model_unavailable())?;
    digest_reader(&mut file, expected_bytes)
}

fn digest_reader(reader: &mut impl Read, expected_bytes: u64) -> Result<String, OcrError> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let count = reader.read(&mut buffer).map_err(|_| model_unavailable())?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| model_unavailable())?)
            .ok_or_else(model_unavailable)?;
        if total > expected_bytes {
            return Err(model_unavailable());
        }
        hasher.update(&buffer[..count]);
    }
    if total != expected_bytes {
        return Err(model_unavailable());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn admit_file(path: &Path, contract: &ModelContract) -> Result<Vec<u8>, OcrError> {
    let metadata = path.symlink_metadata().map_err(|_| model_unavailable())?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > u64::try_from(MAX_ONNX_MODEL_BYTES).map_err(|_| internal())?
    {
        return Err(model_unavailable());
    }
    let expected = usize::try_from(metadata.len()).map_err(|_| model_unavailable())?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(expected).map_err(|_| internal())?;
    let limit = u64::try_from(MAX_ONNX_MODEL_BYTES)
        .map_err(|_| internal())?
        .checked_add(1)
        .ok_or_else(internal)?;
    File::open(path)
        .and_then(|file| file.take(limit).read_to_end(&mut bytes))
        .map_err(|_| model_unavailable())?;
    if bytes.len() != expected {
        return Err(model_unavailable());
    }
    admit_onnx_model(&bytes, contract).map_err(|_| model_unavailable())?;
    Ok(bytes)
}

fn artifact_path(lease: &ModelLease, role: ModelRole) -> Result<PathBuf, OcrError> {
    lease
        .artifact_path(role, ArtifactKind::OnnxGraph)
        .ok_or_else(model_unavailable)
}

fn create_session(model: &[u8], config: &OrtRuntimeConfig) -> Result<Session, OcrError> {
    catch_unwind(AssertUnwindSafe(|| create_session_inner(model, config)))
        .map_err(|_| model_unavailable())?
}

fn create_session_inner(model: &[u8], config: &OrtRuntimeConfig) -> Result<Session, OcrError> {
    let mut builder = Session::builder()
        .map_err(|_| model_unavailable())?
        .with_no_environment_execution_providers()
        .map_err(|_| model_unavailable())?
        .with_execution_providers([CPU::default().with_arena_allocator(true).build()])
        .map_err(|_| model_unavailable())?
        .with_parallel_execution(false)
        .map_err(|_| model_unavailable())?
        .with_intra_threads(config.intra_threads)
        .map_err(|_| model_unavailable())?
        .with_inter_threads(config.inter_threads)
        .map_err(|_| model_unavailable())?
        .with_inter_op_spinning(false)
        .map_err(|_| model_unavailable())?
        .with_intra_op_spinning(false)
        .map_err(|_| model_unavailable())?
        .with_deterministic_compute(true)
        .map_err(|_| model_unavailable())?
        .with_memory_pattern(false)
        .map_err(|_| model_unavailable())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|_| model_unavailable())?
        .with_logger(silent_logger())
        .map_err(|_| model_unavailable())?
        .with_log_level(LogLevel::Fatal)
        .map_err(|_| model_unavailable())?;
    builder
        .commit_from_memory(model)
        .map_err(|_| model_unavailable())
}

fn silent_logger() -> LoggerFunction {
    Arc::new(|_level, _category, _id, _location, _message| {})
}

fn validate_session_contract(session: &Session, contract: &ModelContract) -> Result<(), OcrError> {
    validate_outlets(session.inputs(), &contract.inputs)?;
    validate_outlets(session.outputs(), &contract.outputs)
}

fn validate_outlets(outlets: &[Outlet], contracts: &[TensorContract]) -> Result<(), OcrError> {
    if outlets.len() != contracts.len() {
        return Err(model_unavailable());
    }
    for (outlet, contract) in outlets.iter().zip(contracts) {
        let Some(expected_type) = contract.element_type else {
            return Err(model_unavailable());
        };
        let Some(expected_dimensions) = contract.dimensions.as_ref() else {
            return Err(model_unavailable());
        };
        let ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } = outlet.dtype()
        else {
            return Err(model_unavailable());
        };
        if outlet.name() != contract.name
            || tensor_element_code(*ty) != Some(expected_type)
            || shape.len() != expected_dimensions.len()
            || dimension_symbols.len() != expected_dimensions.len()
        {
            return Err(model_unavailable());
        }
        for ((actual, symbol), expected) in shape
            .iter()
            .zip(dimension_symbols.iter())
            .zip(expected_dimensions)
        {
            let matches = dimension_matches(*actual, symbol, expected);
            if !matches {
                return Err(model_unavailable());
            }
        }
    }
    Ok(())
}

fn dimension_matches(actual: i64, symbol: &str, expected: &DimensionContract) -> bool {
    match expected {
        DimensionContract::Fixed(expected) => {
            symbol.is_empty() && u64::try_from(actual).ok() == Some(*expected)
        }
        DimensionContract::Symbol(expected) => {
            (actual == -1 && symbol == expected) || (actual > 0 && symbol.is_empty())
        }
    }
}

#[allow(clippy::too_many_lines)]
fn tensor_element_code(element: TensorElementType) -> Option<i32> {
    Some(match element {
        TensorElementType::Float32 => 1,
        TensorElementType::Uint8 => 2,
        TensorElementType::Int8 => 3,
        TensorElementType::Uint16 => 4,
        TensorElementType::Int16 => 5,
        TensorElementType::Int32 => 6,
        TensorElementType::Int64 => 7,
        TensorElementType::String => 8,
        TensorElementType::Bool => 9,
        TensorElementType::Float16 => 10,
        TensorElementType::Float64 => 11,
        TensorElementType::Uint32 => 12,
        TensorElementType::Uint64 => 13,
        TensorElementType::Complex64 => 14,
        TensorElementType::Complex128 => 15,
        TensorElementType::Bfloat16 => 16,
        TensorElementType::Float8E4M3FN => 17,
        TensorElementType::Float8E4M3FNUZ => 18,
        TensorElementType::Float8E5M2 => 19,
        TensorElementType::Float8E5M2FNUZ => 20,
        TensorElementType::Uint4 => 21,
        TensorElementType::Int4 => 22,
        _ => return None,
    })
}

fn warm_lane(lane: &mut OrtLane, classes: usize) -> Result<(), OcrError> {
    catch_unwind(AssertUnwindSafe(|| warm_lane_inner(lane, classes)))
        .map_err(|_| model_unavailable())?
}

fn warm_lane_inner(lane: &mut OrtLane, classes: usize) -> Result<(), OcrError> {
    let detector = Tensor::from_array((
        [1_usize, 3, DETECTOR_WARMUP_SIDE, DETECTOR_WARMUP_SIDE],
        vec![0.0_f32; 3 * DETECTOR_WARMUP_SIDE * DETECTOR_WARMUP_SIDE].into_boxed_slice(),
    ))
    .map_err(|_| model_unavailable())?;
    validate_detector_output(run_without_context(&mut lane.detector, detector)?, 32, 32)?;

    let recognizer = Tensor::from_array((
        [1_usize, 3, 48, RECOGNIZER_WARMUP_WIDTH],
        vec![0.0_f32; 3 * 48 * RECOGNIZER_WARMUP_WIDTH].into_boxed_slice(),
    ))
    .map_err(|_| model_unavailable())?;
    let (shape, values) = run_without_context(&mut lane.recognizer, recognizer)?;
    let [batch, time, output_classes] = shape.as_slice() else {
        return Err(model_unavailable());
    };
    if *batch != 1
        || *time <= 0
        || usize::try_from(*output_classes).ok() != Some(classes)
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(model_unavailable());
    }
    Ok(())
}

fn run_without_context(
    session: &mut Session,
    tensor: Tensor<f32>,
) -> Result<(Vec<i64>, Vec<f32>), OcrError> {
    let outputs = catch_unwind(AssertUnwindSafe(|| session.run(ort::inputs![tensor])))
        .map_err(|_| model_unavailable())?
        .map_err(|_| model_unavailable())?;
    extract_single_f32_output(&outputs).map_err(|_| model_unavailable())
}

fn run_session(
    session: &mut Session,
    tensor: Tensor<f32>,
    context: &RequestContext,
) -> Result<(Vec<i64>, Vec<f32>), OcrError> {
    check_request(context)?;
    let options = Arc::new(RunOptions::new().map_err(|_| internal())?);
    let monitor = StopMonitor::spawn(context.clone(), Arc::clone(&options))?;
    let run = catch_unwind(AssertUnwindSafe(|| {
        session.run_with_options(ort::inputs![tensor], &options)
    }));
    let stop = monitor.finish().reconcile(stop_reason(context));
    if let Some(error) = stop.as_error() {
        return Err(error);
    }
    let outputs = run.map_err(|_| internal())?.map_err(|_| internal())?;
    extract_single_f32_output(&outputs)
}

fn extract_single_f32_output(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<(Vec<i64>, Vec<f32>), OcrError> {
    if outputs.len() != 1 {
        return Err(internal());
    }
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|_| internal())?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(internal());
    }
    Ok((shape.to_vec(), values.to_vec()))
}

fn validate_detector_output(
    output: (Vec<i64>, Vec<f32>),
    width: u32,
    height: u32,
) -> Result<(), OcrError> {
    let (shape, values) = output;
    if shape != [1, 1, i64::from(height), i64::from(width)]
        || values.iter().any(|value| !(0.0..=1.0).contains(value))
    {
        return Err(model_unavailable());
    }
    Ok(())
}

trait TerminateRun: Send + Sync + 'static {
    fn terminate(&self);
}

impl TerminateRun for RunOptions {
    fn terminate(&self) {
        let _ = RunOptions::terminate(self);
    }
}

struct StopMonitor {
    done: Arc<AtomicBool>,
    reason: Arc<AtomicU8>,
    thread: Option<thread::JoinHandle<()>>,
}

impl StopMonitor {
    fn spawn<T: TerminateRun>(
        context: RequestContext,
        terminator: Arc<T>,
    ) -> Result<Self, OcrError> {
        let done = Arc::new(AtomicBool::new(false));
        let reason = Arc::new(AtomicU8::new(StopReason::Running as u8));
        let done_for_thread = Arc::clone(&done);
        let reason_for_thread = Arc::clone(&reason);
        let thread = sanitize_monitor_spawn(
            thread::Builder::new()
                .name("impossible-ocr-ort-stop".into())
                .spawn(move || {
                    while !done_for_thread.load(Ordering::Acquire) {
                        let next = stop_reason(&context);
                        if next != StopReason::Running {
                            reason_for_thread.store(next as u8, Ordering::Release);
                            terminator.terminate();
                            return;
                        }
                        thread::park_timeout(MONITOR_INTERVAL);
                    }
                }),
        )?;
        Ok(Self {
            done,
            reason,
            thread: Some(thread),
        })
    }

    fn finish(mut self) -> StopReason {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        StopReason::from_u8(self.reason.load(Ordering::Acquire))
    }
}

fn sanitize_monitor_spawn<T>(result: std::io::Result<T>) -> Result<T, OcrError> {
    result.map_err(|_| internal())
}

impl Drop for StopMonitor {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum StopReason {
    Running = 0,
    Cancelled = 1,
    DeadlineExceeded = 2,
}

impl StopReason {
    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Cancelled,
            2 => Self::DeadlineExceeded,
            _ => Self::Running,
        }
    }

    const fn as_error(self) -> Option<OcrError> {
        match self {
            Self::Running => None,
            Self::Cancelled => Some(OcrError::for_code(OcrErrorCode::Cancelled)),
            Self::DeadlineExceeded => Some(OcrError::for_code(OcrErrorCode::DeadlineExceeded)),
        }
    }

    const fn reconcile(self, fresh: Self) -> Self {
        if matches!(self, Self::Cancelled) || matches!(fresh, Self::Cancelled) {
            Self::Cancelled
        } else if matches!(self, Self::DeadlineExceeded) || matches!(fresh, Self::DeadlineExceeded)
        {
            Self::DeadlineExceeded
        } else {
            Self::Running
        }
    }
}

fn stop_reason(context: &RequestContext) -> StopReason {
    if context.cancellation().is_cancelled() {
        StopReason::Cancelled
    } else if context
        .remaining()
        .is_some_and(|remaining| remaining.is_zero())
    {
        StopReason::DeadlineExceeded
    } else {
        StopReason::Running
    }
}

fn check_request(context: &RequestContext) -> Result<(), OcrError> {
    stop_reason(context).as_error().map_or(Ok(()), Err)
}

struct LanePool<T> {
    inner: Arc<LanePoolInner<T>>,
}

struct LanePoolInner<T> {
    available: Mutex<Vec<T>>,
    ready: Condvar,
}

impl<T> LanePool<T> {
    fn new(lanes: Vec<T>) -> Result<Self, OcrError> {
        if lanes.is_empty() || lanes.len() > MAX_LANES {
            return Err(model_unavailable());
        }
        Ok(Self {
            inner: Arc::new(LanePoolInner {
                available: Mutex::new(lanes),
                ready: Condvar::new(),
            }),
        })
    }

    fn acquire(&self, context: &RequestContext) -> Result<LaneGuard<T>, OcrError> {
        let mut available = self.inner.available.lock().map_err(|_| internal())?;
        loop {
            check_request(context)?;
            if let Some(lane) = available.pop() {
                return Ok(LaneGuard {
                    lane: Some(lane),
                    pool: Arc::clone(&self.inner),
                });
            }
            let wait = context.remaining().map_or(MONITOR_INTERVAL, |remaining| {
                remaining.min(MONITOR_INTERVAL)
            });
            let (next, _) = self
                .inner
                .ready
                .wait_timeout(available, wait)
                .map_err(|_| internal())?;
            available = next;
        }
    }
}

struct LaneGuard<T> {
    lane: Option<T>,
    pool: Arc<LanePoolInner<T>>,
}

impl<T> Deref for LaneGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.lane.as_ref().unwrap_or_else(|| unreachable!())
    }
}

impl<T> DerefMut for LaneGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.lane.as_mut().unwrap_or_else(|| unreachable!())
    }
}

impl<T> Drop for LaneGuard<T> {
    fn drop(&mut self) {
        let Some(lane) = self.lane.take() else {
            return;
        };
        if let Ok(mut available) = self.pool.available.lock() {
            available.push(lane);
            self.pool.ready.notify_one();
        }
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn positive_usize(value: i64) -> Result<usize, OcrError> {
    if value <= 0 {
        return Err(internal());
    }
    usize::try_from(value).map_err(|_| internal())
}

fn catch_internal<T>(operation: impl FnOnce() -> Result<T, OcrError>) -> Result<T, OcrError> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| internal())?
}

const fn model_unavailable() -> OcrError {
    OcrError::for_code(OcrErrorCode::ModelUnavailable)
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    use impossible_ocr_domain::{OcrError, OcrErrorCode};
    use impossible_server_core::{CancellationToken, RequestContext, RequestIdSource};
    use sha2::{Digest, Sha256};

    use super::{
        DimensionContract, LanePool, OrtRuntimeConfig, StopMonitor, TerminateRun, check_request,
        dimension_matches, model_unavailable, prepare_owned_native_runtime,
    };

    #[test]
    fn session_metadata_accepts_only_exact_symbols_or_fixed_runtime_refinements() {
        let expected = DimensionContract::Symbol("batch".into());
        assert!(dimension_matches(-1, "batch", &expected));
        assert!(dimension_matches(1, "", &expected));
        assert!(!dimension_matches(-1, "other", &expected));
        assert!(!dimension_matches(0, "", &expected));
        assert!(!dimension_matches(1, "batch", &expected));
    }

    fn context(timeout: Option<Duration>) -> Result<RequestContext, Box<dyn std::error::Error>> {
        Ok(RequestContext::new(
            RequestIdSource::default().next()?,
            CancellationToken::new(),
            timeout,
        )?)
    }

    #[test]
    fn config_requires_absolute_integrity_bound_library() {
        assert_eq!(
            OrtRuntimeConfig::new("runtime.dll", 1, "a".repeat(64))
                .err()
                .map(OcrError::code),
            Some(OcrErrorCode::ModelUnavailable)
        );
        assert!(OrtRuntimeConfig::new(PathBuf::from("C:/runtime.dll"), 1, "A".repeat(64)).is_err());
        assert!(
            OrtRuntimeConfig::new(PathBuf::from("C:/runtime.dll"), 1, "a".repeat(64))
                .and_then(|config| config.with_lanes(0))
                .is_err()
        );
    }

    #[test]
    fn config_debug_redacts_path_and_digest() -> Result<(), Box<dyn std::error::Error>> {
        let executable = std::env::current_exe()?;
        let secret_digest = "a".repeat(64);
        let config = OrtRuntimeConfig::new(&executable, 1, &secret_digest)?;
        let rendered = format!("{config:?}");
        assert!(!rendered.contains(executable.to_string_lossy().as_ref()));
        assert!(!rendered.contains(&secret_digest));
        assert!(rendered.contains("[REDACTED]"));
        Ok(())
    }

    #[test]
    fn native_runtime_is_promoted_to_verified_content_addressed_storage()
    -> Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let source = temporary.path().join("onnxruntime.dll");
        let original = b"verified-native-runtime";
        fs::write(&source, original)?;
        let digest = format!("{:x}", Sha256::digest(original));
        let config = OrtRuntimeConfig::new(
            source.canonicalize()?,
            u64::try_from(original.len())?,
            &digest,
        )?;
        let prepared = prepare_owned_native_runtime(&config)?;
        assert!(prepared.path.to_string_lossy().contains(&digest));
        assert_eq!(fs::read(&prepared.path)?, original);

        fs::write(&source, b"attacker-replacement")?;
        assert_eq!(fs::read(&prepared.path)?, original);
        drop(prepared);
        fs::write(&source, original)?;

        let owned_directory = temporary
            .path()
            .join(super::NATIVE_CACHE_DIRECTORY)
            .join(&digest);
        let unexpected = owned_directory.join("unexpected");
        fs::write(&unexpected, b"mutant")?;
        assert!(prepare_owned_native_runtime(&config).is_err());
        fs::remove_file(unexpected)?;

        let owned_file = owned_directory.join("onnxruntime.dll");
        make_test_file_writable(&owned_file)?;
        assert!(prepare_owned_native_runtime(&config).is_err());
        Ok(())
    }

    #[cfg(windows)]
    fn make_test_file_writable(path: &std::path::Path) -> std::io::Result<()> {
        let mut permissions = fs::metadata(path)?.permissions();
        #[allow(clippy::permissions_set_readonly_false)] // Windows readonly is a file attribute.
        permissions.set_readonly(false);
        fs::set_permissions(path, permissions)
    }

    #[cfg(unix)]
    fn make_test_file_writable(path: &std::path::Path) -> std::io::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
    }

    #[test]
    fn lane_pool_is_bounded_and_honors_deadlines() -> Result<(), Box<dyn std::error::Error>> {
        let pool = Arc::new(LanePool::new(vec![7_u8])?);
        let held = pool.acquire(&context(None)?)?;
        let waiter_pool = Arc::clone(&pool);
        let waiter = thread::spawn(move || {
            let context =
                context(Some(Duration::from_millis(20))).map_err(|_| model_unavailable())?;
            waiter_pool.acquire(&context).map(|guard| *guard)
        });
        thread::sleep(Duration::from_millis(40));
        assert_eq!(
            waiter
                .join()
                .map_err(|_| model_unavailable())?
                .err()
                .map(OcrError::code),
            Some(OcrErrorCode::DeadlineExceeded)
        );
        drop(held);
        assert_eq!(*pool.acquire(&context(None)?)?, 7);
        Ok(())
    }

    #[test]
    fn waiting_for_lane_honors_cancellation() -> Result<(), Box<dyn std::error::Error>> {
        let pool = Arc::new(LanePool::new(vec![1_u8])?);
        let _held = pool.acquire(&context(None)?)?;
        let token = CancellationToken::new();
        let request = RequestContext::new(RequestIdSource::default().next()?, token.clone(), None)?;
        let _ = token.cancel();
        assert_eq!(
            pool.acquire(&request).err().map(OcrError::code),
            Some(OcrErrorCode::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn terminated_run_does_not_release_lane_before_worker_exits()
    -> Result<(), Box<dyn std::error::Error>> {
        let pool = Arc::new(LanePool::new(vec![11_u8])?);
        let token = CancellationToken::new();
        let worker_request = RequestContext::new(
            RequestIdSource::default().next()?,
            token.clone(),
            Some(Duration::from_secs(1)),
        )?;
        let terminator = Arc::new(FakeTerminator {
            terminated: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let worker_pool = Arc::clone(&pool);
        let worker_terminator = Arc::clone(&terminator);
        let (ready_tx, ready_rx) = mpsc::sync_channel(0);
        let worker = thread::spawn(move || -> Result<(), OcrError> {
            let _lane = worker_pool.acquire(&worker_request)?;
            let monitor = StopMonitor::spawn(worker_request, worker_terminator)?;
            ready_tx.send(()).map_err(|_| model_unavailable())?;
            while monitor.reason.load(Ordering::Acquire) == 0 {
                thread::yield_now();
            }
            if monitor.reason.load(Ordering::Acquire) != 1 {
                return Err(model_unavailable());
            }
            thread::sleep(Duration::from_millis(40));
            let _ = monitor.finish();
            Ok(())
        });
        ready_rx.recv()?;
        let _ = token.cancel();
        let limit = Instant::now() + Duration::from_secs(1);
        while !terminator.terminated.load(Ordering::Acquire) && Instant::now() < limit {
            thread::yield_now();
        }
        let waiting = context(Some(Duration::from_millis(10)))?;
        assert_eq!(
            pool.acquire(&waiting).err().map(OcrError::code),
            Some(OcrErrorCode::DeadlineExceeded)
        );
        worker.join().map_err(|_| model_unavailable())??;
        assert_eq!(*pool.acquire(&context(None)?)?, 11);
        Ok(())
    }

    struct FakeTerminator {
        terminated: AtomicBool,
        calls: AtomicUsize,
    }

    impl TerminateRun for FakeTerminator {
        fn terminate(&self) {
            self.terminated.store(true, Ordering::Release);
            self.calls.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn monitor_terminates_cancelled_run_and_joins() -> Result<(), Box<dyn std::error::Error>> {
        let token = CancellationToken::new();
        let request = RequestContext::new(
            RequestIdSource::default().next()?,
            token.clone(),
            Some(Duration::from_secs(1)),
        )?;
        let terminator = Arc::new(FakeTerminator {
            terminated: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let monitor = StopMonitor::spawn(request, Arc::clone(&terminator))?;
        let _ = token.cancel();
        let limit = Instant::now() + Duration::from_secs(1);
        while !terminator.terminated.load(Ordering::Acquire) && Instant::now() < limit {
            thread::yield_now();
        }
        assert_eq!(
            monitor.finish().as_error().map(OcrError::code),
            Some(OcrErrorCode::Cancelled)
        );
        assert_eq!(terminator.calls.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[test]
    fn cancellation_after_monitor_join_wins_output_race() -> Result<(), Box<dyn std::error::Error>>
    {
        let token = CancellationToken::new();
        let request = RequestContext::new(
            RequestIdSource::default().next()?,
            token.clone(),
            Some(Duration::from_secs(1)),
        )?;
        let terminator = Arc::new(FakeTerminator {
            terminated: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let observed = StopMonitor::spawn(request.clone(), terminator)?.finish();
        assert!(observed.as_error().is_none());
        let _ = token.cancel();
        assert_eq!(
            observed
                .reconcile(super::stop_reason(&request))
                .as_error()
                .map(OcrError::code),
            Some(OcrErrorCode::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn monitor_spawn_errors_are_sanitized() {
        let error = super::sanitize_monitor_spawn::<()>(Err(std::io::Error::other(
            "private-thread-detail",
        )))
        .err();
        assert_eq!(error.map(OcrError::code), Some(OcrErrorCode::Internal));
        assert!(!format!("{error:?}").contains("private-thread-detail"));
    }

    #[test]
    fn cancelled_request_has_precedence_over_expired_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let token = CancellationToken::new();
        let request = RequestContext::new(
            RequestIdSource::default().next()?,
            token.clone(),
            Some(Duration::ZERO),
        )?;
        let _ = token.cancel();
        assert_eq!(
            check_request(&request).err().map(OcrError::code),
            Some(OcrErrorCode::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn panic_payload_is_never_exposed() {
        let error = super::catch_internal(|| -> Result<(), OcrError> {
            std::panic::resume_unwind(Box::new("private-runtime-detail"));
        })
        .err();
        assert_eq!(error.map(OcrError::code), Some(OcrErrorCode::Internal));
        assert!(!format!("{error:?}").contains("private-runtime-detail"));
    }
}
