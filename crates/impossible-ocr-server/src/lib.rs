//! Bounded HTTP, gRPC, and MCP adapters for the canonical OCR pipeline.

#[cfg(feature = "onnx-runtime")]
mod runtime_qualification;

#[cfg(feature = "onnx-runtime")]
pub use runtime_qualification::{
    RuntimeQualificationConfig, RuntimeQualificationError, RuntimeQualificationErrorCode,
    RuntimeQualificationReport, qualify_installed_runtime,
};

#[cfg(feature = "onnx-runtime")]
use std::path::{Component, Path, PathBuf};
use std::{
    future::{Future, IntoFuture},
    io,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    Extension, Json, Router,
    body::Bytes,
    extract::{
        DefaultBodyLimit, Request, State,
        rejection::{BytesRejection, JsonRejection},
    },
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    serve::Listener,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use impossible_ocr_domain::{
    Confidence, InputMetadata, OcrError, OcrErrorCode, OcrLanguage, OcrOptions, OcrResult,
    RasterFormat, RasterInput,
};
#[cfg(feature = "onnx-runtime")]
use impossible_ocr_pipeline::{CtcDecoder, PpOcrBackend, RasterDecoder};
use impossible_ocr_pipeline::{OcrBackend, OcrPipeline};
use impossible_ocr_protocol::{
    ErrorBody, ErrorEnvelope, MAX_BATCH_IMAGE_BYTES, MAX_BATCH_ITEMS, MAX_IMAGE_BYTES,
    base64_wire_len, grpc as pb, http, mcp,
};
use impossible_server_core::{
    CancellationToken, DrainOutcome, HealthRegistry, ProcessState, ReadinessReason, RequestContext,
    RequestIdSource, RequestStop, ServerLimits, ShutdownGate,
};
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{OwnedSemaphorePermit, Semaphore},
    time::Sleep,
};
use tonic::{Request as GrpcRequest, Response as GrpcResponse, Status};

const BACKEND_COMPONENT: &str = "ocr_backend";
const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Debug, Clone, Copy)]
struct WireLimits {
    image: usize,
    batch_images: usize,
    raw: usize,
    json_single: usize,
    json_batch: usize,
    mcp: usize,
    grpc: usize,
}

impl WireLimits {
    fn new(limits: ServerLimits) -> Self {
        let image = limits.max_request_bytes().min(MAX_IMAGE_BYTES);
        let batch_images = image.min(MAX_BATCH_IMAGE_BYTES);
        let base64 = base64_wire_len(image).unwrap_or(usize::MAX);
        let item_overhead = serde_json::to_vec(&http::OcrRequest {
            image_base64: String::new(),
            metadata: InputMetadata {
                format: RasterFormat::Jpeg,
                encoded_bytes: u64::try_from(image).unwrap_or(u64::MAX),
                width: impossible_ocr_domain::MAX_DIMENSION,
                height: impossible_ocr_domain::MAX_DIMENSION,
            },
            options: OcrOptions {
                language: OcrLanguage::English,
                include_words: false,
                minimum_confidence: Confidence::ZERO,
            },
        })
        .map_or(512, |bytes| bytes.len());
        let batch_shell = serde_json::to_vec(&http::BatchRequest { items: Vec::new() })
            .map_or(16, |bytes| bytes.len());
        let json_single = base64.saturating_add(item_overhead);
        let split_padding = 4_usize.saturating_mul(MAX_BATCH_ITEMS.saturating_sub(1));
        let json_batch = base64
            .saturating_add(split_padding)
            .saturating_add(item_overhead.saturating_mul(MAX_BATCH_ITEMS))
            .saturating_add(batch_shell);
        let mcp_shell = serde_json::to_vec(&json!({
            "jsonrpc":"2.0","id":"", "method":"tools/call",
            "params":{"name":mcp::OCR_BATCH,"arguments":{}}
        }))
        .map_or(256, |bytes| bytes.len());
        let mcp = json_batch.saturating_add(mcp_shell);
        let proto_item = pb::RecognizeRequest {
            image: Vec::new(),
            metadata: Some(pb::InputMetadata {
                format: pb::RasterFormat::Jpeg as i32,
                encoded_bytes: u64::try_from(image).unwrap_or(u64::MAX),
                width: impossible_ocr_domain::MAX_DIMENSION,
                height: impossible_ocr_domain::MAX_DIMENSION,
            }),
            options: Some(pb::OcrOptions {
                include_words: false,
                minimum_confidence: 1.0,
                language: "english".to_owned(),
            }),
        }
        .encoded_len();
        let grpc = batch_images
            .saturating_add(proto_item.saturating_mul(MAX_BATCH_ITEMS))
            .saturating_add(10_usize.saturating_mul(MAX_BATCH_ITEMS));
        Self {
            image,
            batch_images,
            raw: image,
            json_single,
            json_batch,
            mcp,
            grpc,
        }
    }
}

#[derive(Debug)]
struct Metrics {
    active: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
}

impl Metrics {
    const fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct AppState<B> {
    pipeline: OcrPipeline<B>,
    limits: ServerLimits,
    health: HealthRegistry,
    gate: ShutdownGate,
    shutdown: CancellationToken,
    ids: RequestIdSource,
    permits: Arc<Semaphore>,
    queued: AtomicUsize,
    metrics: Metrics,
    wire: WireLimits,
    warmed: AtomicBool,
    model_ids: Vec<&'static str>,
}

/// Explicit offline startup configuration for the curated CPU ONNX backend.
#[cfg(feature = "onnx-runtime")]
#[derive(Clone)]
pub struct OrtServerConfig {
    model_store: PathBuf,
    runtime_directory: PathBuf,
    runtime_library_name: PathBuf,
    runtime: impossible_ocr_onnx::OrtRuntimeConfig,
    lanes: usize,
    intra_threads: usize,
    inter_threads: usize,
    cpu_budget: usize,
}

#[cfg(feature = "onnx-runtime")]
impl OrtServerConfig {
    /// Creates an explicit store and native-runtime configuration.
    ///
    /// `runtime_library_name` must be one filename beneath the absolute runtime directory. The
    /// profile must equal the curated profile; arbitrary model identifiers are rejected.
    ///
    /// # Errors
    /// Returns a sanitized model-unavailable error for incomplete, relative, traversing, or
    /// otherwise invalid configuration.
    pub fn new(
        model_store: impl Into<PathBuf>,
        profile: &str,
        runtime_directory: impl Into<PathBuf>,
        runtime_library_name: impl Into<PathBuf>,
        runtime_library_bytes: u64,
        runtime_library_sha256: impl Into<String>,
        cpu_budget: usize,
    ) -> Result<Self, OcrError> {
        let model_store = model_store.into();
        let runtime_directory = runtime_directory.into();
        let runtime_library_name = runtime_library_name.into();
        if profile != impossible_ocr_onnx::CURATED_BUNDLE_ID
            || !safe_absolute(&model_store)
            || !safe_absolute(&runtime_directory)
            || !single_filename(&runtime_library_name)
            || !(1..=256).contains(&cpu_budget)
        {
            return Err(model_unavailable());
        }
        let runtime = impossible_ocr_onnx::OrtRuntimeConfig::new(
            runtime_directory.join(&runtime_library_name),
            runtime_library_bytes,
            runtime_library_sha256,
        )?;
        Ok(Self {
            model_store,
            runtime_directory,
            runtime_library_name,
            runtime,
            lanes: 1,
            intra_threads: 1,
            inter_threads: 1,
            cpu_budget,
        })
    }

    /// Sets fixed session-lane and per-session thread limits.
    ///
    /// # Errors
    /// Returns a sanitized model-unavailable error if the limits exceed the explicit CPU budget
    /// or the runtime's fixed safety caps.
    pub fn with_runtime_limits(
        mut self,
        lanes: usize,
        intra_threads: usize,
        inter_threads: usize,
    ) -> Result<Self, OcrError> {
        let threads_per_lane = intra_threads.max(inter_threads);
        if lanes == 0
            || threads_per_lane == 0
            || lanes
                .checked_mul(threads_per_lane)
                .is_none_or(|threads| threads > self.cpu_budget)
        {
            return Err(model_unavailable());
        }
        self.runtime = self
            .runtime
            .with_lanes(lanes)?
            .with_threads(intra_threads, inter_threads)?;
        self.lanes = lanes;
        self.intra_threads = intra_threads;
        self.inter_threads = inter_threads;
        Ok(self)
    }

    fn validate_for(&self, limits: ServerLimits) -> Result<(), OcrError> {
        if self.lanes > limits.max_concurrent_requests()
            || self
                .lanes
                .checked_mul(self.intra_threads.max(self.inter_threads))
                .is_none_or(|threads| threads > self.cpu_budget)
        {
            return Err(model_unavailable());
        }
        Ok(())
    }
}

#[cfg(feature = "onnx-runtime")]
impl std::fmt::Debug for OrtServerConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OrtServerConfig")
            .field("model_store", &"[REDACTED]")
            .field("runtime_directory", &"[REDACTED]")
            .field("runtime_library_name", &"[REDACTED]")
            .field("runtime", &self.runtime)
            .field("lanes", &self.lanes)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field("cpu_budget", &self.cpu_budget)
            .finish()
    }
}

#[derive(Debug, Clone)]
struct Admission {
    id: u64,
    context: RequestContext,
}

#[derive(Debug)]
struct DeadlineListener {
    inner: TcpListener,
    lifetime: Duration,
}

impl Listener for DeadlineListener {
    type Io = DeadlineStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.inner.accept().await {
                Ok((stream, address)) => {
                    return (
                        DeadlineStream {
                            inner: stream,
                            deadline: Box::pin(tokio::time::sleep(self.lifetime)),
                        },
                        address,
                    );
                }
                Err(_) => tokio::task::yield_now().await,
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

#[derive(Debug)]
struct DeadlineStream {
    inner: TcpStream,
    deadline: Pin<Box<Sleep>>,
}

impl AsyncRead for DeadlineStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.deadline.as_mut().poll(cx).is_ready() {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "HTTP connection read deadline exceeded",
            )))
        } else {
            Pin::new(&mut self.inner).poll_read(cx, buffer)
        }
    }
}

impl AsyncWrite for DeadlineStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// OCR transport host backed by one canonical pipeline implementation.
#[derive(Debug)]
pub struct OcrServer<B> {
    state: Arc<AppState<B>>,
}

impl<B> Clone for OcrServer<B> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl OcrServer<impossible_ocr_onnx::UnavailableOnnxBackend> {
    /// Creates the production shell with an unavailable backend and false readiness.
    #[must_use]
    pub fn new(limits: ServerLimits) -> Self {
        Self::with_backend(limits, impossible_ocr_onnx::UnavailableOnnxBackend)
    }
}

/// Builds the curated production backend entirely from an already-installed local model store and
/// an explicitly identified native ONNX Runtime library.
///
/// This function never scans the machine, downloads models, or searches `PATH`. The returned
/// server is ready only after bundle verification, exact graph admission, session construction,
/// and warm-up all succeed.
///
/// # Errors
/// Returns a sanitized model-unavailable error for invalid resource budgets, unsafe paths, missing
/// or corrupt local artifacts, provisional graph contracts, native runtime failures, or warm-up
/// failures.
#[cfg(feature = "onnx-runtime")]
pub async fn build_ort_server(
    limits: ServerLimits,
    config: OrtServerConfig,
) -> Result<OcrServer<PpOcrBackend<impossible_ocr_onnx::OrtCpuRuntime>>, OcrError> {
    config.validate_for(limits)?;
    verify_runtime_location(&config).await?;
    let catalog = impossible_ocr_onnx::ModelCatalog::curated().map_err(|_| model_unavailable())?;
    let store = impossible_ocr_onnx::ModelStore::open(&config.model_store, catalog)
        .await
        .map_err(|_| model_unavailable())?;
    let verified = store
        .verify(impossible_ocr_onnx::CURATED_BUNDLE_ID)
        .await
        .map_err(|_| model_unavailable())?;
    if verified.bundle_id != impossible_ocr_onnx::CURATED_BUNDLE_ID
        || verified.profile_id != impossible_ocr_onnx::CURATED_BUNDLE_ID
        || verified.runtime_detector_max_side != 960
    {
        return Err(model_unavailable());
    }
    let decoder: CtcDecoder = verified.ctc_decoder()?;
    let runtime = impossible_ocr_onnx::OrtCpuRuntime::load(
        &store,
        impossible_ocr_onnx::CURATED_BUNDLE_ID,
        config.runtime,
    )
    .await?;
    let backend = PpOcrBackend::new(
        Arc::new(runtime),
        RasterDecoder::new(impossible_ocr_domain::RasterLimits::default()),
        decoder,
    );
    let server =
        OcrServer::with_unwarmed_backend(limits, backend, impossible_ocr_onnx::CURATED_BUNDLE_ID);
    server.warm_up().await?;
    Ok(server)
}

impl<B: OcrBackend> OcrServer<B> {
    /// Creates a transport host around an explicit backend.
    #[must_use]
    pub fn with_backend(limits: ServerLimits, backend: B) -> Self {
        let warmed = backend.state() == impossible_ocr_pipeline::BackendState::Ready;
        Self::with_backend_state(limits, backend, warmed, Vec::new())
    }

    /// Creates a false-ready server around a backend that must pass explicit warm-up before use.
    #[must_use]
    pub fn with_unwarmed_backend(limits: ServerLimits, backend: B, model_id: &'static str) -> Self {
        Self::with_backend_state(limits, backend, false, vec![model_id])
    }

    fn with_backend_state(
        limits: ServerLimits,
        backend: B,
        warmed: bool,
        model_ids: Vec<&'static str>,
    ) -> Self {
        let wire = WireLimits::new(limits);
        let pipeline = OcrPipeline::new(backend);
        let health = HealthRegistry::new();
        health.register_component(BACKEND_COMPONENT);
        let _ = health.set_component_ready(BACKEND_COMPONENT, warmed && pipeline.is_ready());
        health.set_process(ProcessState::Running);
        Self {
            state: Arc::new(AppState {
                pipeline,
                limits,
                health,
                gate: ShutdownGate::new(),
                shutdown: CancellationToken::new(),
                ids: RequestIdSource::default(),
                permits: Arc::new(Semaphore::new(limits.max_concurrent_requests())),
                queued: AtomicUsize::new(0),
                metrics: Metrics::new(),
                wire,
                warmed: AtomicBool::new(warmed),
                model_ids,
            }),
        }
    }

    /// Warms the backend and enables readiness only after a successful ready-state recheck.
    ///
    /// # Errors
    /// Leaves readiness false and returns a sanitized backend error if warm-up fails or the backend
    /// does not enter the ready state.
    pub async fn warm_up(&self) -> Result<(), OcrError> {
        self.state.warmed.store(false, Ordering::Release);
        refresh_backend_health(&self.state);
        self.state.pipeline.warm_up().await?;
        if !self.state.pipeline.is_ready() {
            return Err(model_unavailable());
        }
        self.state.warmed.store(true, Ordering::Release);
        refresh_backend_health(&self.state);
        Ok(())
    }

    /// Returns the bounded HTTP and Streamable HTTP MCP router.
    pub fn router(&self) -> Router {
        let state = Arc::clone(&self.state);
        let admitted = Router::new()
            .route(
                http::OCR_ROUTE,
                post(recognize_json::<B>).layer(DefaultBodyLimit::max(self.state.wire.json_single)),
            )
            .route(
                http::OCR_RAW_ROUTE,
                post(recognize_raw::<B>).layer(DefaultBodyLimit::max(self.state.wire.raw)),
            )
            .route(
                http::OCR_BATCH_ROUTE,
                post(recognize_batch::<B>).layer(DefaultBodyLimit::max(self.state.wire.json_batch)),
            )
            .route(
                http::MCP_ROUTE,
                post(mcp_http::<B>).layer(DefaultBodyLimit::max(self.state.wire.mcp)),
            )
            .route_layer(middleware::from_fn_with_state(
                Arc::clone(&state),
                admit_http::<B>,
            ));
        Router::new()
            .route("/health/live", get(live::<B>))
            .route("/health/ready", get(ready::<B>))
            .route("/metrics", get(metrics::<B>))
            .route(http::STATUS_ROUTE, get(status::<B>))
            .route(http::CAPABILITIES_ROUTE, get(capabilities::<B>))
            .route(http::MODELS_ROUTE, get(models::<B>))
            .merge(admitted)
            .method_not_allowed_fallback(method_not_allowed::<B>)
            .fallback(not_found::<B>)
            .with_state(state)
    }

    /// Returns the generated unary gRPC service using the same pipeline state.
    #[must_use]
    pub fn grpc_service(&self) -> pb::ocr_service_server::OcrServiceServer<GrpcService<B>> {
        pb::ocr_service_server::OcrServiceServer::new(GrpcService {
            state: Arc::clone(&self.state),
        })
        .max_decoding_message_size(self.state.wire.grpc)
        .max_encoding_message_size(self.state.wire.grpc)
    }

    /// Serves HTTP on a loopback listener until cancellation and bounded drain complete.
    ///
    /// # Errors
    /// Rejects non-loopback listeners and returns listener, serving, or forced-drain failures.
    pub async fn serve(&self, listener: TcpListener, stop: CancellationToken) -> io::Result<()> {
        ensure_loopback(&listener)?;
        let stop_for_graceful = stop.clone();
        let listener = DeadlineListener {
            inner: listener,
            lifetime: self.state.limits.request_timeout(),
        };
        let server = axum::serve(listener, self.router())
            .with_graceful_shutdown(async move {
                stop_for_graceful.cancelled().await;
            })
            .into_future();
        tokio::pin!(server);
        let mut drain_budget = self.state.limits.shutdown_timeout();
        tokio::select! {
            result = &mut server => result?,
            () = stop.cancelled() => {
                let started = Instant::now();
                self.begin_shutdown();
                if tokio::time::timeout(self.state.limits.shutdown_timeout(), &mut server).await.is_err() {
                    self.state.gate.stop_now();
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "HTTP shutdown timed out"));
                }
                drain_budget = drain_budget.saturating_sub(started.elapsed());
            }
        }
        self.finish_drain(drain_budget).await
    }

    /// Serves gRPC on a loopback listener until cancellation and bounded drain complete.
    ///
    /// # Errors
    /// Rejects non-loopback listeners and returns transport or forced-drain failures.
    pub async fn serve_grpc(
        &self,
        listener: TcpListener,
        stop: CancellationToken,
    ) -> io::Result<()> {
        ensure_loopback(&listener)?;
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let stop_for_graceful = stop.clone();
        let server = tonic::transport::Server::builder()
            .concurrency_limit_per_connection(self.state.limits.max_concurrent_requests())
            .load_shed(true)
            .timeout(self.state.limits.request_timeout())
            .add_service(self.grpc_service())
            .serve_with_incoming_shutdown(incoming, async move {
                stop_for_graceful.cancelled().await;
            });
        tokio::pin!(server);
        let mut drain_budget = self.state.limits.shutdown_timeout();
        tokio::select! {
            result = &mut server => result.map_err(io::Error::other)?,
            () = stop.cancelled() => {
                let started = Instant::now();
                self.begin_shutdown();
                if let Ok(result) = tokio::time::timeout(self.state.limits.shutdown_timeout(), &mut server).await {
                    result.map_err(io::Error::other)?;
                } else {
                    self.state.gate.stop_now();
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "gRPC shutdown timed out"));
                }
                drain_budget = drain_budget.saturating_sub(started.elapsed());
            }
        }
        self.finish_drain(drain_budget).await
    }

    /// Serves newline-delimited MCP JSON-RPC over stdio with the same strict tool contracts.
    ///
    /// # Errors
    /// Returns sanitized standard-I/O failures.
    pub async fn serve_mcp_stdio(&self) -> io::Result<()> {
        self.serve_mcp_io(BufReader::new(tokio::io::stdin()), tokio::io::stdout())
            .await
    }

    /// Serves MCP stdio until EOF or an external shutdown signal, then performs bounded drain.
    ///
    /// # Errors
    /// Returns standard-I/O or forced-drain failures.
    pub async fn serve_mcp_stdio_until(&self, stop: CancellationToken) -> io::Result<()> {
        tokio::select! {
            result = self.serve_mcp_stdio() => result,
            () = stop.cancelled() => {
                self.begin_shutdown();
                self.finish_drain(self.state.limits.shutdown_timeout()).await
            }
        }
    }

    async fn serve_mcp_io<R, W>(&self, mut input: R, mut stdout: W) -> io::Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut line = Vec::new();
        while let Some(too_large) =
            read_bounded_line(&mut input, &mut line, self.state.wire.mcp).await?
        {
            let response = if too_large {
                Some(mcp_error(
                    Value::Null,
                    -32_600,
                    "request too large",
                    next_id(&self.state),
                ))
            } else {
                match serde_json::from_slice::<Value>(&line) {
                    Ok(value) => handle_mcp(Arc::clone(&self.state), value, None).await,
                    Err(_) => Some(mcp_error(
                        Value::Null,
                        -32_700,
                        "parse error",
                        next_id(&self.state),
                    )),
                }
            };
            if let Some(response) = response {
                let mut bytes = serde_json::to_vec(&response).map_err(io::Error::other)?;
                bytes.push(b'\n');
                stdout.write_all(&bytes).await?;
                stdout.flush().await?;
            }
        }
        Ok(())
    }

    fn begin_shutdown(&self) {
        self.state.warmed.store(false, Ordering::Release);
        self.state.health.set_process(ProcessState::Draining);
        let _ = self.state.shutdown.cancel();
    }

    async fn finish_drain(&self, timeout: Duration) -> io::Result<()> {
        self.begin_shutdown();
        let outcome = self.state.gate.drain(timeout).await;
        self.state.health.set_process(ProcessState::Stopped);
        if outcome == DrainOutcome::Forced {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "OCR drain timed out",
            ))
        } else {
            Ok(())
        }
    }
}

async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    output: &mut Vec<u8>,
    maximum: usize,
) -> io::Result<Option<bool>> {
    output.clear();
    let mut too_large = false;
    let mut observed = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(observed.then_some(too_large));
        }
        observed = true;
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |index| index);
        if !too_large {
            if output.len().saturating_add(take) > maximum {
                too_large = true;
                output.clear();
            } else {
                output.extend_from_slice(&available[..take]);
            }
        }
        let consumed = take.saturating_add(usize::from(newline.is_some()));
        reader.consume(consumed);
        if newline.is_some() {
            if output.last() == Some(&b'\r') {
                output.pop();
            }
            return Ok(Some(too_large));
        }
    }
}

fn ensure_loopback(listener: &TcpListener) -> io::Result<()> {
    if listener.local_addr()?.ip().is_loopback() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "public binding is disabled",
        ))
    }
}

#[cfg(feature = "onnx-runtime")]
async fn verify_runtime_location(config: &OrtServerConfig) -> Result<(), OcrError> {
    let directory_metadata = tokio::fs::symlink_metadata(&config.runtime_directory)
        .await
        .map_err(|_| model_unavailable())?;
    let library_path = config.runtime_directory.join(&config.runtime_library_name);
    let library_metadata = tokio::fs::symlink_metadata(&library_path)
        .await
        .map_err(|_| model_unavailable())?;
    if !directory_metadata.is_dir()
        || directory_metadata.file_type().is_symlink()
        || !library_metadata.is_file()
        || library_metadata.file_type().is_symlink()
    {
        return Err(model_unavailable());
    }
    let canonical_directory = tokio::fs::canonicalize(&config.runtime_directory)
        .await
        .map_err(|_| model_unavailable())?;
    let canonical_library = tokio::fs::canonicalize(&library_path)
        .await
        .map_err(|_| model_unavailable())?;
    if canonical_library.parent() != Some(canonical_directory.as_path()) {
        return Err(model_unavailable());
    }
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn safe_absolute(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|component| !matches!(component, Component::ParentDir | Component::CurDir))
}

#[cfg(feature = "onnx-runtime")]
fn single_filename(path: &Path) -> bool {
    let mut components = path.components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

async fn admit_http<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
    mut request: Request,
    next: Next,
) -> Response {
    if request.uri().path() == http::MCP_ROUTE && state.shutdown.is_cancelled() {
        return next.run(request).await;
    }
    let (id, context) = match new_context(&state) {
        Ok(value) => value,
        Err(error) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, error.code(), 0),
    };
    let Some(_work) = state.gate.try_enter() else {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, OcrErrorCode::Cancelled, id);
    };
    let _permit = match acquire(&state, &context).await {
        Ok(permit) => permit,
        Err(error) => return error_response(status_for(error.code()), error.code(), id),
    };
    state.metrics.active.fetch_add(1, Ordering::Relaxed);
    let _active = ActiveGuard(&state.metrics);
    request.extensions_mut().insert(Admission {
        id,
        context: context.clone(),
    });
    tokio::select! {
        biased;
        stop = context.stopped() => {
            state.metrics.failed.fetch_add(1, Ordering::Relaxed);
            let error = stop_error(stop);
            error_response(status_for(error.code()), error.code(), id)
        }
        response = next.run(request) => response,
    }
}

#[derive(Debug, Serialize)]
struct HealthBody {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

async fn live<B: OcrBackend>(State(state): State<Arc<AppState<B>>>) -> Response {
    let snapshot = state.health.snapshot();
    let status_code = if snapshot.live {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status_code,
        Json(HealthBody {
            status: if snapshot.live { "live" } else { "stopped" },
            reason: None,
        }),
    )
        .into_response()
}

async fn ready<B: OcrBackend>(State(state): State<Arc<AppState<B>>>) -> Response {
    refresh_backend_health(&state);
    let snapshot = state.health.snapshot();
    let status_code = if snapshot.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status_code,
        Json(HealthBody {
            status: if snapshot.ready { "ready" } else { "not_ready" },
            reason: snapshot.reason.map(ReadinessReason::as_str),
        }),
    )
        .into_response()
}

async fn status<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
) -> Json<http::StatusResponse> {
    refresh_backend_health(&state);
    let snapshot = state.health.snapshot();
    Json(http::StatusResponse {
        live: snapshot.live,
        ready: snapshot.ready,
        reason: snapshot.reason.map(ReadinessReason::as_str),
    })
}

async fn capabilities<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
) -> Json<http::CapabilitiesResponse> {
    Json(capabilities_response(&state))
}

fn capabilities_response<B>(state: &AppState<B>) -> http::CapabilitiesResponse {
    http::CapabilitiesResponse {
        max_batch_items: MAX_BATCH_ITEMS,
        max_image_bytes: state.wire.image,
        max_batch_image_bytes: state.wire.batch_images,
        max_raw_wire_bytes: state.wire.raw,
        max_json_wire_bytes: state.wire.json_single.max(state.wire.json_batch),
        max_mcp_wire_bytes: state.wire.mcp,
        max_grpc_wire_bytes: state.wire.grpc,
        ..http::CapabilitiesResponse::default()
    }
}

async fn models<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
) -> Json<http::ModelsResponse> {
    Json(models_response(&state))
}

fn models_response<B: OcrBackend>(state: &AppState<B>) -> http::ModelsResponse {
    refresh_backend_health(state);
    let ready = backend_ready(state);
    http::ModelsResponse {
        state: if ready { "ready" } else { "unavailable" },
        models: if ready {
            state.model_ids.clone()
        } else {
            Vec::new()
        },
    }
}

async fn metrics<B: OcrBackend>(State(state): State<Arc<AppState<B>>>) -> impl IntoResponse {
    refresh_backend_health(&state);
    let ready = u8::from(state.health.snapshot().ready);
    let body = format!(
        "# TYPE impossible_ocr_ready gauge\nimpossible_ocr_ready {ready}\n\
         # TYPE impossible_ocr_active_requests gauge\nimpossible_ocr_active_requests {}\n\
         # TYPE impossible_ocr_completed_requests counter\nimpossible_ocr_completed_requests {}\n\
         # TYPE impossible_ocr_failed_requests counter\nimpossible_ocr_failed_requests {}\n",
        state.metrics.active.load(Ordering::Relaxed),
        state.metrics.completed.load(Ordering::Relaxed),
        state.metrics.failed.load(Ordering::Relaxed),
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}

fn refresh_backend_health<B: OcrBackend>(state: &AppState<B>) {
    let _ = state
        .health
        .set_component_ready(BACKEND_COMPONENT, backend_ready(state));
}

fn backend_ready<B: OcrBackend>(state: &AppState<B>) -> bool {
    state.warmed.load(Ordering::Acquire) && state.pipeline.is_ready()
}

async fn recognize_json<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
    Extension(admission): Extension<Admission>,
    request: Result<Json<http::OcrRequest>, JsonRejection>,
) -> Response {
    let id = admission.id;
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => {
            let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(status, OcrErrorCode::InvalidRequest, id);
        }
    };
    let input = match decode_json_request(&request, state.wire.image) {
        Ok(input) => input,
        Err(error) => return error_response(status_for(error.code()), error.code(), id),
    };
    match run_admitted_inputs(&state, &admission.context, vec![(input, request.options)]).await {
        Ok(mut results) => success_response(
            StatusCode::OK,
            id,
            http::OcrResponse {
                result: results.remove(0),
            },
        ),
        Err(error) => error_response(status_for(error.code()), error.code(), id),
    }
}

async fn recognize_raw<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
    Extension(admission): Extension<Admission>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let id = admission.id;
    let body = match body {
        Ok(body) => body,
        Err(rejection) => {
            let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(status, OcrErrorCode::InvalidRequest, id);
        }
    };
    if body.len() > state.wire.image {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            OcrErrorCode::InvalidRequest,
            id,
        );
    }
    let (metadata, options) = match raw_metadata(&headers, body.len()) {
        Ok(value) => value,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error.code(), id),
    };
    let input = match RasterInput::new(body.to_vec(), metadata).and_then(validate_raster_magic) {
        Ok(input) => input,
        Err(error) => return error_response(status_for(error.code()), error.code(), id),
    };
    match run_admitted_inputs(&state, &admission.context, vec![(input, options)]).await {
        Ok(mut results) => success_response(
            StatusCode::OK,
            id,
            http::OcrResponse {
                result: results.remove(0),
            },
        ),
        Err(error) => error_response(status_for(error.code()), error.code(), id),
    }
}

async fn recognize_batch<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
    Extension(admission): Extension<Admission>,
    request: Result<Json<http::BatchRequest>, JsonRejection>,
) -> Response {
    let id = admission.id;
    let request = match request {
        Ok(Json(request)) if request.has_valid_cardinality() => request,
        Ok(_) => return error_response(StatusCode::BAD_REQUEST, OcrErrorCode::InvalidRequest, id),
        Err(rejection) => {
            let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(status, OcrErrorCode::InvalidRequest, id);
        }
    };
    let mut inputs = Vec::with_capacity(request.items.len());
    let mut total = 0_usize;
    for item in &request.items {
        let input = match decode_json_request(item, state.wire.image) {
            Ok(input) => input,
            Err(error) => return error_response(status_for(error.code()), error.code(), id),
        };
        total = match total.checked_add(input.bytes().len()) {
            Some(total) if total <= state.wire.batch_images => total,
            _ => {
                return error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    OcrErrorCode::InvalidRequest,
                    id,
                );
            }
        };
        inputs.push((input, item.options));
    }
    match run_admitted_inputs(&state, &admission.context, inputs).await {
        Ok(results) => success_response(StatusCode::OK, id, http::BatchResponse { results }),
        Err(error) => error_response(status_for(error.code()), error.code(), id),
    }
}

fn decode_json_request(
    request: &http::OcrRequest,
    maximum: usize,
) -> Result<RasterInput, OcrError> {
    let padding = request
        .image_base64
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'=')
        .take(2)
        .count();
    let estimated = request
        .image_base64
        .len()
        .checked_div(4)
        .and_then(|groups| groups.checked_mul(3))
        .and_then(|bytes| bytes.checked_sub(padding))
        .ok_or_else(|| OcrError::for_code(OcrErrorCode::InvalidRequest))?;
    if estimated > maximum {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    let bytes = STANDARD
        .decode(&request.image_base64)
        .map_err(|_| OcrError::for_code(OcrErrorCode::InvalidRequest))?;
    if bytes.len() > maximum {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    RasterInput::new(bytes, request.metadata).and_then(validate_raster_magic)
}

fn validate_raster_magic(input: RasterInput) -> Result<RasterInput, OcrError> {
    let bytes = input.bytes();
    let matches = match input.metadata().format {
        RasterFormat::Png => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        RasterFormat::Jpeg => bytes.starts_with(&[0xff, 0xd8, 0xff]),
    };
    if matches {
        Ok(input)
    } else {
        Err(OcrError::for_code(OcrErrorCode::InvalidImage))
    }
}

fn raw_metadata(
    headers: &HeaderMap,
    length: usize,
) -> Result<(InputMetadata, OcrOptions), OcrError> {
    let format = match header_text(headers, "x-image-format")? {
        "png" => RasterFormat::Png,
        "jpeg" => RasterFormat::Jpeg,
        _ => return Err(OcrError::for_code(OcrErrorCode::InvalidRequest)),
    };
    let expected_content_type = match format {
        RasterFormat::Png => "image/png",
        RasterFormat::Jpeg => "image/jpeg",
    };
    if header_text(headers, "content-type")? != expected_content_type {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    let width = parse_header::<u32>(headers, "x-image-width")?;
    let height = parse_header::<u32>(headers, "x-image-height")?;
    let include_words = match headers.get("x-include-words") {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|text| text.parse::<bool>().ok())
            .ok_or_else(|| OcrError::for_code(OcrErrorCode::InvalidRequest))?,
        None => true,
    };
    let minimum_confidence = match headers.get("x-minimum-confidence") {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|text| text.parse::<f32>().ok())
            .ok_or_else(|| OcrError::for_code(OcrErrorCode::InvalidRequest))?,
        None => 0.0,
    };
    Ok((
        InputMetadata {
            format,
            encoded_bytes: u64::try_from(length)
                .map_err(|_| OcrError::for_code(OcrErrorCode::InvalidRequest))?,
            width,
            height,
        },
        OcrOptions {
            language: OcrLanguage::English,
            include_words,
            minimum_confidence: Confidence::new(minimum_confidence)?,
        },
    ))
}

fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Result<&'a str, OcrError> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| OcrError::for_code(OcrErrorCode::InvalidRequest))
}

fn parse_header<T: std::str::FromStr>(headers: &HeaderMap, name: &str) -> Result<T, OcrError> {
    header_text(headers, name)?
        .parse()
        .map_err(|_| OcrError::for_code(OcrErrorCode::InvalidRequest))
}

struct ActiveGuard<'a>(&'a Metrics);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}

struct QueueGuard<'a>(&'a AtomicUsize);

impl Drop for QueueGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn acquire<B: OcrBackend>(
    state: &AppState<B>,
    context: &RequestContext,
) -> Result<OwnedSemaphorePermit, OcrError> {
    if let Ok(permit) = Arc::clone(&state.permits).try_acquire_owned() {
        return Ok(permit);
    }
    let queued = state.queued.fetch_add(1, Ordering::AcqRel);
    if queued >= state.limits.queue_capacity() {
        state.queued.fetch_sub(1, Ordering::AcqRel);
        return Err(OcrError::for_code(OcrErrorCode::Overloaded));
    }
    let _queue_guard = QueueGuard(&state.queued);
    tokio::select! {
        biased;
        stop = context.stopped() => Err(stop_error(stop)),
        permit = Arc::clone(&state.permits).acquire_owned() => {
            permit.map_err(|_| OcrError::for_code(OcrErrorCode::Cancelled))
        }
    }
}

async fn run_inputs<B: OcrBackend>(
    state: &Arc<AppState<B>>,
    context: &RequestContext,
    inputs: Vec<(RasterInput, OcrOptions)>,
) -> Result<Vec<OcrResult>, OcrError> {
    let _work = state
        .gate
        .try_enter()
        .ok_or_else(|| OcrError::for_code(OcrErrorCode::Cancelled))?;
    let _permit = acquire(state, context).await?;
    state.metrics.active.fetch_add(1, Ordering::Relaxed);
    let _active = ActiveGuard(&state.metrics);
    run_admitted_inputs(state, context, inputs).await
}

async fn run_admitted_inputs<B: OcrBackend>(
    state: &Arc<AppState<B>>,
    context: &RequestContext,
    inputs: Vec<(RasterInput, OcrOptions)>,
) -> Result<Vec<OcrResult>, OcrError> {
    if !backend_ready(state) {
        return Err(model_unavailable());
    }
    let mut results = Vec::with_capacity(inputs.len());
    for (input, options) in &inputs {
        let result = tokio::select! {
            biased;
            stop = context.stopped() => Err(stop_error(stop)),
            result = state.pipeline.recognize(input, *options, context) => result,
        };
        match result {
            Ok(result) => results.push(result),
            Err(error) => {
                state.metrics.failed.fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        }
    }
    state.metrics.completed.fetch_add(1, Ordering::Relaxed);
    Ok(results)
}

fn stop_error(stop: RequestStop) -> OcrError {
    OcrError::for_code(match stop {
        RequestStop::Cancelled => OcrErrorCode::Cancelled,
        RequestStop::DeadlineExceeded => OcrErrorCode::DeadlineExceeded,
    })
}

const fn model_unavailable() -> OcrError {
    OcrError::for_code(OcrErrorCode::ModelUnavailable)
}

fn next_id<B>(state: &AppState<B>) -> u64 {
    state
        .ids
        .next()
        .map_or(0, impossible_server_core::RequestId::get)
}

fn new_context<B>(state: &AppState<B>) -> Result<(u64, RequestContext), OcrError> {
    let request_id = state
        .ids
        .next()
        .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
    let id = request_id.get();
    let context = RequestContext::new(
        request_id,
        state.shutdown.clone(),
        Some(state.limits.request_timeout()),
    )
    .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
    Ok((id, context))
}

fn success_response<T: Serialize>(status: StatusCode, id: u64, body: T) -> Response {
    let mut response = (status, Json(body)).into_response();
    insert_request_id(response.headers_mut(), id);
    response
}

fn error_response(status: StatusCode, code: OcrErrorCode, id: u64) -> Response {
    let error = OcrError::for_code(code);
    let mut response = (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody::from(error),
            request_id: id,
        }),
    )
        .into_response();
    insert_request_id(response.headers_mut(), id);
    response
}

fn insert_request_id(headers: &mut HeaderMap, id: u64) {
    if let Ok(value) = HeaderValue::from_str(&id.to_string()) {
        headers.insert(REQUEST_ID_HEADER, value);
    }
}

const fn status_for(code: OcrErrorCode) -> StatusCode {
    match code {
        OcrErrorCode::InvalidRequest | OcrErrorCode::InvalidImage => StatusCode::BAD_REQUEST,
        OcrErrorCode::ModelUnavailable | OcrErrorCode::Cancelled => StatusCode::SERVICE_UNAVAILABLE,
        OcrErrorCode::Overloaded => StatusCode::TOO_MANY_REQUESTS,
        OcrErrorCode::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

async fn not_found<B: OcrBackend>(State(state): State<Arc<AppState<B>>>) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        OcrErrorCode::InvalidRequest,
        next_id(&state),
    )
}

async fn method_not_allowed<B: OcrBackend>(State(state): State<Arc<AppState<B>>>) -> Response {
    error_response(
        StatusCode::METHOD_NOT_ALLOWED,
        OcrErrorCode::InvalidRequest,
        next_id(&state),
    )
}

/// Generated gRPC adapter using the canonical OCR pipeline.
#[derive(Debug)]
pub struct GrpcService<B> {
    state: Arc<AppState<B>>,
}

#[tonic::async_trait]
impl<B: OcrBackend> pb::ocr_service_server::OcrService for GrpcService<B> {
    async fn recognize(
        &self,
        request: GrpcRequest<pb::RecognizeRequest>,
    ) -> Result<GrpcResponse<pb::RecognizeResponse>, Status> {
        let (id, context) = new_context(&self.state).map_err(status_from_error)?;
        let (input, options) = proto_request(request.into_inner(), self.state.wire.image)
            .map_err(|error| status_from_error_with_id(error, id))?;
        let mut results = run_inputs(&self.state, &context, vec![(input, options)])
            .await
            .map_err(|error| status_from_error_with_id(error, id))?;
        Ok(grpc_response_with_id(
            pb::RecognizeResponse {
                result: Some(result_to_proto(results.remove(0))),
                request_id: id,
            },
            id,
        ))
    }

    async fn recognize_batch(
        &self,
        request: GrpcRequest<pb::RecognizeBatchRequest>,
    ) -> Result<GrpcResponse<pb::RecognizeBatchResponse>, Status> {
        let (id, context) = new_context(&self.state).map_err(status_from_error)?;
        let request = request.into_inner();
        if request.items.is_empty() || request.items.len() > MAX_BATCH_ITEMS {
            return Err(status_from_error_with_id(
                OcrError::for_code(OcrErrorCode::InvalidRequest),
                id,
            ));
        }
        let mut inputs = Vec::with_capacity(request.items.len());
        let mut total = 0_usize;
        for item in request.items {
            let (input, options) = proto_request(item, self.state.wire.image)
                .map_err(|error| status_from_error_with_id(error, id))?;
            total = total
                .checked_add(input.bytes().len())
                .filter(|total| *total <= self.state.wire.batch_images)
                .ok_or_else(|| {
                    status_from_error_with_id(OcrError::for_code(OcrErrorCode::InvalidRequest), id)
                })?;
            inputs.push((input, options));
        }
        let results = run_inputs(&self.state, &context, inputs)
            .await
            .map_err(|error| status_from_error_with_id(error, id))?;
        Ok(grpc_response_with_id(
            pb::RecognizeBatchResponse {
                results: results.into_iter().map(result_to_proto).collect(),
                request_id: id,
            },
            id,
        ))
    }

    async fn get_capabilities(
        &self,
        _request: GrpcRequest<pb::GetCapabilitiesRequest>,
    ) -> Result<GrpcResponse<pb::CapabilitiesResponse>, Status> {
        let id = next_id(&self.state);
        let value = capabilities_response(&self.state);
        Ok(grpc_response_with_id(
            pb::CapabilitiesResponse {
                api_version: value.api_version.to_owned(),
                raster_formats: value.raster_formats.map(str::to_owned).to_vec(),
                max_batch_items: u32::try_from(value.max_batch_items).unwrap_or(u32::MAX),
                supports_pdf: value.supports_pdf,
                supports_jobs: value.supports_jobs,
                supports_websocket: value.supports_websocket,
                max_image_bytes: u64::try_from(value.max_image_bytes).unwrap_or(u64::MAX),
                max_batch_image_bytes: u64::try_from(value.max_batch_image_bytes)
                    .unwrap_or(u64::MAX),
                max_grpc_wire_bytes: u64::try_from(value.max_grpc_wire_bytes).unwrap_or(u64::MAX),
            },
            id,
        ))
    }
}

fn proto_request(
    request: pb::RecognizeRequest,
    maximum: usize,
) -> Result<(RasterInput, OcrOptions), OcrError> {
    let metadata = request
        .metadata
        .ok_or_else(|| OcrError::for_code(OcrErrorCode::InvalidRequest))?;
    let format = match pb::RasterFormat::try_from(metadata.format).ok() {
        Some(pb::RasterFormat::Png) => RasterFormat::Png,
        Some(pb::RasterFormat::Jpeg) => RasterFormat::Jpeg,
        Some(pb::RasterFormat::Unspecified) | None => {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
    };
    let options = request.options.unwrap_or(pb::OcrOptions {
        include_words: true,
        minimum_confidence: 0.0,
        language: "english".to_owned(),
    });
    if options.language != "english" {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    if request.image.len() > maximum {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    let input = RasterInput::new(
        request.image,
        InputMetadata {
            format,
            encoded_bytes: metadata.encoded_bytes,
            width: metadata.width,
            height: metadata.height,
        },
    )
    .and_then(validate_raster_magic)?;
    Ok((
        input,
        OcrOptions {
            language: OcrLanguage::English,
            include_words: options.include_words,
            minimum_confidence: Confidence::new(options.minimum_confidence)?,
        },
    ))
}

fn status_from_error(error: OcrError) -> Status {
    match error.code() {
        OcrErrorCode::InvalidRequest | OcrErrorCode::InvalidImage => {
            Status::invalid_argument(error.code().as_str())
        }
        OcrErrorCode::ModelUnavailable => Status::unavailable(error.code().as_str()),
        OcrErrorCode::Overloaded => Status::resource_exhausted(error.code().as_str()),
        OcrErrorCode::Cancelled => Status::cancelled(error.code().as_str()),
        OcrErrorCode::DeadlineExceeded => Status::deadline_exceeded(error.code().as_str()),
        OcrErrorCode::Internal => Status::internal(error.code().as_str()),
        _ => Status::internal("internal"),
    }
}

fn status_from_error_with_id(error: OcrError, id: u64) -> Status {
    let mut status = status_from_error(error);
    insert_grpc_request_id(status.metadata_mut(), id);
    status
}

fn grpc_response_with_id<T>(message: T, id: u64) -> GrpcResponse<T> {
    let mut response = GrpcResponse::new(message);
    insert_grpc_request_id(response.metadata_mut(), id);
    response
}

fn insert_grpc_request_id(metadata: &mut tonic::metadata::MetadataMap, id: u64) {
    if let Ok(value) = id.to_string().parse() {
        metadata.insert(REQUEST_ID_HEADER, value);
    }
}

fn result_to_proto(result: OcrResult) -> pb::OcrResult {
    pb::OcrResult {
        pages: result
            .pages
            .into_iter()
            .map(|page| pb::Page {
                index: page.index,
                width: page.width,
                height: page.height,
                blocks: page
                    .blocks
                    .into_iter()
                    .map(|block| pb::Block {
                        polygon: Some(polygon_to_proto(block.polygon)),
                        lines: block
                            .lines
                            .into_iter()
                            .map(|line| pb::Line {
                                text: line.text,
                                confidence: line.confidence.get(),
                                polygon: Some(polygon_to_proto(line.polygon)),
                                words: line
                                    .words
                                    .into_iter()
                                    .map(|word| pb::Word {
                                        text: word.text,
                                        confidence: word.confidence.get(),
                                        polygon: Some(polygon_to_proto(word.polygon)),
                                    })
                                    .collect(),
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn polygon_to_proto(polygon: impossible_ocr_domain::Polygon) -> pb::Polygon {
    pb::Polygon {
        points: polygon
            .points
            .into_iter()
            .map(|point| pb::Point {
                x: point.x,
                y: point.y,
            })
            .collect(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpRequest {
    jsonrpc: String,
    #[serde(default, rename = "id")]
    _id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpCallParams {
    name: String,
    #[serde(default)]
    arguments: Value,
}

async fn mcp_http<B: OcrBackend>(
    State(state): State<Arc<AppState<B>>>,
    admission: Option<Extension<Admission>>,
    request: Result<Json<Value>, JsonRejection>,
) -> Response {
    let admission = admission.map(|Extension(value)| value);
    let response = match request {
        Ok(Json(value)) => handle_mcp(state, value, admission.clone()).await,
        Err(_) => Some(mcp_error(
            Value::Null,
            -32_700,
            "parse error",
            admission.as_ref().map_or(0, |value| value.id),
        )),
    };
    match response {
        Some(response) => (StatusCode::OK, Json(response)).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

#[derive(Debug, Clone)]
struct McpExecution {
    admission: Admission,
    pre_admitted: bool,
}

async fn handle_mcp<B: OcrBackend>(
    state: Arc<AppState<B>>,
    value: Value,
    admission: Option<Admission>,
) -> Option<Value> {
    let execution = match admission {
        Some(admission) => McpExecution {
            admission,
            pre_admitted: true,
        },
        None => match new_context(&state) {
            Ok((id, context)) => McpExecution {
                admission: Admission { id, context },
                pre_admitted: false,
            },
            Err(error) => {
                return Some(mcp_error(Value::Null, -32_003, error.code().as_str(), 0));
            }
        },
    };
    let id = match value.as_object().and_then(|object| object.get("id")) {
        None => None,
        Some(Value::Null) => Some(Value::Null),
        Some(value @ (Value::String(_) | Value::Number(_))) => Some(value.clone()),
        Some(_) => {
            return Some(mcp_error(
                Value::Null,
                -32_600,
                "invalid request",
                execution.admission.id,
            ));
        }
    };
    let request = match serde_json::from_value::<McpRequest>(value) {
        Ok(request) if request.jsonrpc == "2.0" => request,
        _ => {
            return Some(mcp_error(
                Value::Null,
                -32_600,
                "invalid request",
                execution.admission.id,
            ));
        }
    };
    let correlation = id.clone().unwrap_or(Value::Null);
    let response = match request.method.as_str() {
        "notifications/initialized" if id.is_none() => None,
        "notifications/initialized" => Some(mcp_error(
            correlation,
            -32_600,
            "invalid request",
            execution.admission.id,
        )),
        "initialize" => Some(json!({
            "jsonrpc": "2.0", "id": correlation,
            "result": {"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"impossible-ocr","version":"0.1.0"}}
        })),
        "tools/list" => Some(json!({
            "jsonrpc": "2.0", "id": correlation,
            "result": {"tools":[
                {"name":mcp::OCR_RECOGNIZE,"description":"Recognize one PNG or JPEG raster","inputSchema":schema_value(mcp::RECOGNIZE_SCHEMA)},
                {"name":mcp::OCR_BATCH,"description":"Recognize a bounded raster batch","inputSchema":schema_value(mcp::BATCH_SCHEMA)},
                {"name":mcp::OCR_CAPABILITIES,"description":"Return OCR capabilities","inputSchema":schema_value(mcp::EMPTY_SCHEMA)},
                {"name":mcp::OCR_MODELS,"description":"Return aggregate model status","inputSchema":schema_value(mcp::EMPTY_SCHEMA)}
            ]}
        })),
        "tools/call" => Some(mcp_call(state, correlation, request.params, &execution).await),
        _ => Some(mcp_error(
            correlation,
            -32_601,
            "method not found",
            execution.admission.id,
        )),
    };
    if id.is_none() { None } else { response }
}

fn schema_value(schema: &str) -> Value {
    serde_json::from_str(schema).unwrap_or_else(|_| json!({"type":"object"}))
}

async fn mcp_call<B: OcrBackend>(
    state: Arc<AppState<B>>,
    id: Value,
    params: Value,
    execution: &McpExecution,
) -> Value {
    let Ok(call) = serde_json::from_value::<McpCallParams>(params) else {
        return mcp_error(id, -32_602, "invalid params", 0);
    };
    let arguments = call.arguments;
    match call.name.as_str() {
        mcp::OCR_CAPABILITIES if is_empty_object(&arguments) => {
            mcp_success(id, json!(capabilities_response(&state)))
        }
        mcp::OCR_MODELS if is_empty_object(&arguments) => {
            mcp_success(id, json!(models_response(&state)))
        }
        mcp::OCR_RECOGNIZE => {
            let Ok(request) = serde_json::from_value::<http::OcrRequest>(arguments) else {
                return mcp_error(id, -32_602, "invalid params", 0);
            };
            let input = match decode_json_request(&request, state.wire.image) {
                Ok(input) => input,
                Err(error) => return mcp_ocr_error(id, error, execution.admission.id),
            };
            let result = if execution.pre_admitted {
                run_admitted_inputs(
                    &state,
                    &execution.admission.context,
                    vec![(input, request.options)],
                )
                .await
            } else {
                run_inputs(
                    &state,
                    &execution.admission.context,
                    vec![(input, request.options)],
                )
                .await
            };
            match result {
                Ok(mut results) => mcp_success(
                    id,
                    json!({"result": results.remove(0), "request_id": execution.admission.id}),
                ),
                Err(error) => mcp_ocr_error(id, error, execution.admission.id),
            }
        }
        mcp::OCR_BATCH => {
            let request = match serde_json::from_value::<http::BatchRequest>(arguments) {
                Ok(request) if request.has_valid_cardinality() => request,
                _ => return mcp_error(id, -32_602, "invalid params", 0),
            };
            let mut inputs = Vec::with_capacity(request.items.len());
            let mut total = 0_usize;
            for item in request.items {
                let input = match decode_json_request(&item, state.wire.image) {
                    Ok(input) => input,
                    Err(error) => return mcp_ocr_error(id, error, execution.admission.id),
                };
                total = match total.checked_add(input.bytes().len()) {
                    Some(total) if total <= state.wire.batch_images => total,
                    _ => {
                        return mcp_ocr_error(
                            id,
                            OcrError::for_code(OcrErrorCode::InvalidRequest),
                            execution.admission.id,
                        );
                    }
                };
                inputs.push((input, item.options));
            }
            let result = if execution.pre_admitted {
                run_admitted_inputs(&state, &execution.admission.context, inputs).await
            } else {
                run_inputs(&state, &execution.admission.context, inputs).await
            };
            match result {
                Ok(results) => mcp_success(
                    id,
                    json!({"results":results,"request_id":execution.admission.id}),
                ),
                Err(error) => mcp_ocr_error(id, error, execution.admission.id),
            }
        }
        _ => mcp_error(id, -32_602, "unknown tool", 0),
    }
}

fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(serde_json::Map::is_empty)
}

#[allow(clippy::needless_pass_by_value)] // The helper owns one complete JSON-RPC response boundary.
fn mcp_success(id: Value, structured: Value) -> Value {
    json!({
        "jsonrpc":"2.0", "id":id,
        "result":{"content":[{"type":"text","text":"OCR operation completed"}],"structuredContent":structured,"isError":false}
    })
}

fn mcp_ocr_error(id: Value, error: OcrError, request_id: u64) -> Value {
    mcp_error(id, -32_003, error.code().as_str(), request_id)
}

#[allow(clippy::needless_pass_by_value)] // The helper owns one complete JSON-RPC response boundary.
fn mcp_error(id: Value, code: i32, message: &'static str, request_id: u64) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message,"data":{"request_id":request_id}}})
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use impossible_ocr_domain::{OcrBlock, OcrLine, OcrPage, OcrWord, Point, Polygon};
    use impossible_ocr_pipeline::{BackendFuture, BackendState, OcrBackend};
    use impossible_server_core::{CancellationToken, ServerLimits};
    use impossible_server_testkit::reserve_loopback_listener;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tower::ServiceExt;

    use super::*;

    #[derive(Debug)]
    struct FakeBackend;

    impl OcrBackend for FakeBackend {
        fn state(&self) -> BackendState {
            BackendState::Ready
        }
        fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
            Box::pin(async { Ok(()) })
        }
        fn recognize<'a>(
            &'a self,
            input: &'a RasterInput,
            options: OcrOptions,
            _context: &'a RequestContext,
        ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
            Box::pin(async move { Ok(fake_result(input.metadata(), options.include_words)) })
        }
    }

    #[derive(Debug)]
    struct FailingWarmupBackend;

    impl OcrBackend for FailingWarmupBackend {
        fn state(&self) -> BackendState {
            BackendState::Ready
        }

        fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
            Box::pin(async { Err(model_unavailable()) })
        }

        fn recognize<'a>(
            &'a self,
            _input: &'a RasterInput,
            _options: OcrOptions,
            _context: &'a RequestContext,
        ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
            Box::pin(async { Err(model_unavailable()) })
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SlowBackend(Duration);

    impl OcrBackend for SlowBackend {
        fn state(&self) -> BackendState {
            BackendState::Ready
        }

        fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
            Box::pin(async { Ok(()) })
        }

        fn recognize<'a>(
            &'a self,
            input: &'a RasterInput,
            options: OcrOptions,
            _context: &'a RequestContext,
        ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
            Box::pin(async move {
                tokio::time::sleep(self.0).await;
                Ok(fake_result(input.metadata(), options.include_words))
            })
        }
    }

    fn polygon(width: u32, height: u32) -> Polygon {
        let right = f32::from(u8::try_from(width.min(8)).unwrap_or(8));
        let bottom = f32::from(u8::try_from(height.min(8)).unwrap_or(8));
        Polygon {
            points: [
                Point { x: 0.0, y: 0.0 },
                Point { x: right, y: 0.0 },
                Point {
                    x: right,
                    y: bottom,
                },
                Point { x: 0.0, y: bottom },
            ],
        }
    }

    fn fake_result(metadata: InputMetadata, include_words: bool) -> OcrResult {
        let polygon = polygon(metadata.width, metadata.height);
        let confidence = Confidence::new(1.0).unwrap_or(Confidence::ZERO);
        let words = if include_words {
            vec![OcrWord {
                text: "TEST".to_owned(),
                confidence,
                polygon,
            }]
        } else {
            Vec::new()
        };
        OcrResult {
            pages: vec![OcrPage {
                index: 0,
                width: metadata.width,
                height: metadata.height,
                blocks: vec![OcrBlock {
                    polygon,
                    lines: vec![OcrLine {
                        text: "TEST".to_owned(),
                        confidence,
                        polygon,
                        words,
                    }],
                }],
            }],
        }
    }

    fn request_json() -> Value {
        json!({"image_base64":STANDARD.encode(b"\x89PNG\r\n\x1a\n"),"metadata":{"format":"png","encoded_bytes":8,"width":8,"height":8}})
    }

    fn test_server() -> OcrServer<FakeBackend> {
        OcrServer::with_backend(
            ServerLimits::new(4096, 1, 1, Duration::from_secs(1), Duration::from_secs(1))
                .unwrap_or_default(),
            FakeBackend,
        )
    }

    fn json_http_request() -> Result<Request<Body>, Box<dyn std::error::Error>> {
        Ok(Request::post(http::OCR_ROUTE)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&request_json())?))?)
    }

    #[tokio::test]
    async fn control_plane_reflects_backend_readiness() -> Result<(), Box<dyn std::error::Error>> {
        for (path, expected) in [
            ("/health/live", 200),
            ("/health/ready", 200),
            ("/metrics", 200),
            (http::CAPABILITIES_ROUTE, 200),
            (http::MODELS_ROUTE, 200),
        ] {
            let response = test_server()
                .router()
                .oneshot(Request::builder().uri(path).body(Body::empty())?)
                .await?;
            assert_eq!(response.status().as_u16(), expected);
        }
        assert_eq!(
            OcrServer::new(ServerLimits::default())
                .router()
                .oneshot(
                    Request::builder()
                        .uri("/health/ready")
                        .body(Body::empty())?
                )
                .await?
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        Ok(())
    }

    #[tokio::test]
    async fn explicit_warmup_gates_readiness_and_model_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = OcrServer::with_unwarmed_backend(
            ServerLimits::default(),
            FakeBackend,
            impossible_ocr_onnx::CURATED_BUNDLE_ID,
        );
        let before = server
            .router()
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(before.status(), StatusCode::SERVICE_UNAVAILABLE);

        server.warm_up().await?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .uri(http::MODELS_ROUTE)
                    .body(Body::empty())?,
            )
            .await?;
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
        assert_eq!(body["state"], "ready");
        assert_eq!(
            body["models"],
            json!([impossible_ocr_onnx::CURATED_BUNDLE_ID])
        );
        Ok(())
    }

    #[tokio::test]
    async fn model_status_is_identical_over_http_mcp_http_and_mcp_stdio()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = OcrServer::with_unwarmed_backend(
            ServerLimits::default(),
            FakeBackend,
            impossible_ocr_onnx::CURATED_BUNDLE_ID,
        );
        assert_model_transport_parity(&server, json!({"state":"unavailable","models":[]})).await?;
        server.warm_up().await?;
        assert_model_transport_parity(
            &server,
            json!({"state":"ready","models":[impossible_ocr_onnx::CURATED_BUNDLE_ID]}),
        )
        .await?;
        server.begin_shutdown();
        assert_model_transport_parity(&server, json!({"state":"unavailable","models":[]})).await?;
        Ok(())
    }

    async fn assert_model_transport_parity<B: OcrBackend>(
        server: &OcrServer<B>,
        expected: Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let http_response = server
            .router()
            .oneshot(
                Request::builder()
                    .uri(http::MODELS_ROUTE)
                    .body(Body::empty())?,
            )
            .await?;
        let http_body: Value =
            serde_json::from_slice(&http_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(http_body, expected);

        let request = json!({
            "jsonrpc":"2.0",
            "id":1,
            "method":"tools/call",
            "params":{"name":mcp::OCR_MODELS,"arguments":{}}
        });
        let mcp_response = server
            .router()
            .oneshot(
                Request::post(http::MCP_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request)?))?,
            )
            .await?;
        let mcp_body: Value =
            serde_json::from_slice(&mcp_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(mcp_body["result"]["structuredContent"], expected);

        let (mut client, server_side) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(server_side);
        let serving = server.clone();
        let task =
            tokio::spawn(async move { serving.serve_mcp_io(BufReader::new(read), write).await });
        client.write_all(&serde_json::to_vec(&request)?).await?;
        client.write_all(b"\n").await?;
        client.shutdown().await?;
        let mut output = Vec::new();
        client.read_to_end(&mut output).await?;
        task.await??;
        let stdio_body: Value = serde_json::from_slice(&output)?;
        assert_eq!(stdio_body["result"]["structuredContent"], expected);
        Ok(())
    }

    #[tokio::test]
    async fn failed_warmup_never_becomes_ready() -> Result<(), Box<dyn std::error::Error>> {
        let server = OcrServer::with_unwarmed_backend(
            ServerLimits::default(),
            FailingWarmupBackend,
            impossible_ocr_onnx::CURATED_BUNDLE_ID,
        );
        assert!(server.warm_up().await.is_err());
        let readiness = server
            .router()
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(readiness.status(), StatusCode::SERVICE_UNAVAILABLE);
        let models = server
            .router()
            .oneshot(
                Request::builder()
                    .uri(http::MODELS_ROUTE)
                    .body(Body::empty())?,
            )
            .await?;
        let body: Value = serde_json::from_slice(&models.into_body().collect().await?.to_bytes())?;
        assert_eq!(body, json!({"state":"unavailable","models":[]}));
        Ok(())
    }

    #[cfg(feature = "onnx-runtime")]
    #[tokio::test]
    async fn ort_startup_config_is_bounded_redacted_and_location_confined()
    -> Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let runtime_directory = temporary.path().join("runtime");
        let model_store = temporary.path().join("models");
        tokio::fs::create_dir(&runtime_directory).await?;
        tokio::fs::write(runtime_directory.join("runtime.bin"), b"runtime").await?;
        let digest = "a".repeat(64);
        let config = OrtServerConfig::new(
            &model_store,
            impossible_ocr_onnx::CURATED_BUNDLE_ID,
            &runtime_directory,
            "runtime.bin",
            7,
            &digest,
            4,
        )?
        .with_runtime_limits(2, 2, 1)?;
        config.validate_for(ServerLimits::default())?;
        verify_runtime_location(&config).await?;
        let rendered = format!("{config:?}");
        assert!(!rendered.contains(temporary.path().to_string_lossy().as_ref()));
        assert!(!rendered.contains(&digest));
        assert!(rendered.contains("[REDACTED]"));
        assert!(config.clone().with_runtime_limits(3, 2, 1).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn http_raw_json_batch_and_mcp_have_semantic_parity()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = test_server();
        let json_response = server
            .router()
            .oneshot(
                Request::post(http::OCR_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request_json())?))?,
            )
            .await?;
        assert_eq!(json_response.status(), StatusCode::OK);
        let request_id = json_response
            .headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .ok_or("missing request id")?
            .to_owned();
        let json_body: Value =
            serde_json::from_slice(&json_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(
            json_body["request_id"]
                .as_u64()
                .map(|value| value.to_string()),
            None
        );
        assert!(!request_id.is_empty());
        let raw_response = server
            .router()
            .oneshot(
                Request::post(http::OCR_RAW_ROUTE)
                    .header("content-type", "image/png")
                    .header("x-image-format", "png")
                    .header("x-image-width", "8")
                    .header("x-image-height", "8")
                    .body(Body::from(b"\x89PNG\r\n\x1a\n".to_vec()))?,
            )
            .await?;
        let raw_body: Value =
            serde_json::from_slice(&raw_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(json_body["result"], raw_body["result"]);
        let batch = json!({"items":[request_json(),request_json()]});
        let batch_response = server
            .router()
            .oneshot(
                Request::post(http::OCR_BATCH_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&batch)?))?,
            )
            .await?;
        let batch_body: Value =
            serde_json::from_slice(&batch_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(batch_body["results"][0], json_body["result"]);
        let mcp = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ocr_recognize","arguments":request_json()}});
        let mcp_response = server
            .router()
            .oneshot(
                Request::post(http::MCP_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&mcp)?))?,
            )
            .await?;
        let mcp_body: Value =
            serde_json::from_slice(&mcp_response.into_body().collect().await?.to_bytes())?;
        assert_eq!(
            mcp_body["result"]["structuredContent"]["result"],
            json_body["result"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn malformed_oversized_and_poisoned_batches_fail_before_backend()
    -> Result<(), Box<dyn std::error::Error>> {
        let bad = json!({"items":[request_json(),{"image_base64":"not-base64","metadata":{"format":"png","encoded_bytes":1,"width":1,"height":1}}]});
        let response = test_server()
            .router()
            .oneshot(
                Request::post(http::OCR_BATCH_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&bad)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let too_many = json!({"items":vec![request_json();MAX_BATCH_ITEMS+1]});
        let response = test_server()
            .router()
            .oneshot(
                Request::post(http::OCR_BATCH_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&too_many)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let response = test_server()
            .router()
            .oneshot(
                Request::post(http::OCR_RAW_ROUTE)
                    .header("content-type", "image/png")
                    .header("x-image-format", "png")
                    .header("x-image-width", "8")
                    .header("x-image-height", "8")
                    .body(Body::from(vec![0_u8; 5000]))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
        assert_eq!(body["error"]["code"], "invalid_request");

        let response = test_server()
            .router()
            .oneshot(Request::get(http::OCR_ROUTE).body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
        assert_eq!(body["error"]["code"], "invalid_request");
        Ok(())
    }

    #[tokio::test]
    async fn grpc_maps_results_and_errors_stably() -> Result<(), Box<dyn std::error::Error>> {
        use pb::ocr_service_server::OcrService;
        let service = GrpcService {
            state: test_server().state,
        };
        let response = service
            .recognize(GrpcRequest::new(pb::RecognizeRequest {
                image: b"\x89PNG\r\n\x1a\n".to_vec(),
                metadata: Some(pb::InputMetadata {
                    format: pb::RasterFormat::Png as i32,
                    encoded_bytes: 8,
                    width: 8,
                    height: 8,
                }),
                options: None,
            }))
            .await?;
        assert!(response.metadata().get(REQUEST_ID_HEADER).is_some());
        let response = response.into_inner();
        assert_eq!(
            response
                .result
                .and_then(|result| result.pages.into_iter().next())
                .map(|page| page.width),
            Some(8)
        );
        let error = service
            .recognize_batch(GrpcRequest::new(pb::RecognizeBatchRequest {
                items: Vec::new(),
            }))
            .await;
        let Err(error) = error else {
            return Err("empty batch unexpectedly succeeded".into());
        };
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.metadata().get(REQUEST_ID_HEADER).is_some());

        let capabilities = service
            .get_capabilities(GrpcRequest::new(pb::GetCapabilitiesRequest {}))
            .await?;
        assert!(capabilities.metadata().get(REQUEST_ID_HEADER).is_some());
        Ok(())
    }

    #[tokio::test]
    async fn real_grpc_listener_serves_and_stops_cleanly() -> Result<(), Box<dyn std::error::Error>>
    {
        let listener = reserve_loopback_listener().await?;
        let address = listener.local_addr()?;
        let token = CancellationToken::new();
        let stop = token.clone();
        let server = test_server();
        let serving = server.clone();
        let join = tokio::spawn(async move { serving.serve_grpc(listener, token).await });
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))?
            .connect()
            .await?;
        let mut client = pb::ocr_service_client::OcrServiceClient::new(channel);
        let response = client
            .recognize(pb::RecognizeRequest {
                image: b"\x89PNG\r\n\x1a\n".to_vec(),
                metadata: Some(pb::InputMetadata {
                    format: pb::RasterFormat::Png as i32,
                    encoded_bytes: 8,
                    width: 8,
                    height: 8,
                }),
                options: None,
            })
            .await?
            .into_inner();
        assert_eq!(
            response
                .result
                .and_then(|result| result.pages.into_iter().next())
                .map(|page| page.width),
            Some(8)
        );
        let _ = stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), join).await???;
        Ok(())
    }

    #[tokio::test]
    async fn listener_starts_and_shutdown_is_bounded() -> Result<(), Box<dyn std::error::Error>> {
        let listener = reserve_loopback_listener().await?;
        let token = CancellationToken::new();
        let stop = token.clone();
        let server = test_server();
        let join = tokio::spawn(async move { server.serve(listener, token).await });
        let _ = stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), join).await???;
        Ok(())
    }

    #[tokio::test]
    async fn deadline_and_shutdown_cancel_active_backend_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits =
            ServerLimits::new(4096, 1, 1, Duration::from_millis(5), Duration::from_secs(1))?;
        let deadline_server = OcrServer::with_backend(limits, SlowBackend(Duration::from_secs(1)));
        let response = deadline_server
            .router()
            .oneshot(json_http_request()?)
            .await?;
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);

        let limits = ServerLimits::new(4096, 1, 1, Duration::from_secs(5), Duration::from_secs(1))?;
        let shutdown_server = OcrServer::with_backend(limits, SlowBackend(Duration::from_secs(5)));
        let state = Arc::clone(&shutdown_server.state);
        let join = tokio::spawn(shutdown_server.router().oneshot(json_http_request()?));
        while state.metrics.active.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let _ = state.shutdown.cancel();
        let response = join.await??;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        Ok(())
    }

    #[tokio::test]
    async fn queue_saturation_is_bounded_and_handler_drop_releases_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = ServerLimits::new(4096, 1, 1, Duration::from_secs(2), Duration::from_secs(1))?;
        let server = OcrServer::with_backend(limits, SlowBackend(Duration::from_millis(100)));
        let first = tokio::spawn(server.router().oneshot(json_http_request()?));
        while server.state.metrics.active.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let second = tokio::spawn(server.router().oneshot(json_http_request()?));
        while server.state.queued.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let third = server.router().oneshot(json_http_request()?).await?;
        assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(first.await??.status(), StatusCode::OK);
        assert_eq!(second.await??.status(), StatusCode::OK);

        let dropped = tokio::spawn(server.router().oneshot(json_http_request()?));
        while server.state.metrics.active.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        dropped.abort();
        let _ = dropped.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while server.state.metrics.active.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn real_listener_bounds_slow_headers_and_bodies() -> Result<(), Box<dyn std::error::Error>>
    {
        let limits = ServerLimits::new(
            4096,
            1,
            1,
            Duration::from_millis(50),
            Duration::from_secs(1),
        )?;
        let server = OcrServer::with_backend(limits, FakeBackend);
        let listener = reserve_loopback_listener().await?;
        let address = listener.local_addr()?;
        let stop = CancellationToken::new();
        let serving = server.clone();
        let server_stop = stop.clone();
        let join = tokio::spawn(async move { serving.serve(listener, server_stop).await });

        let mut slow_header = TcpStream::connect(address).await?;
        slow_header
            .write_all(b"POST /v1/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Len")
            .await?;
        let mut byte = [0_u8; 1];
        assert!(
            tokio::time::timeout(Duration::from_millis(250), slow_header.read(&mut byte))
                .await
                .is_ok()
        );

        let mut slow_body = TcpStream::connect(address).await?;
        slow_body
            .write_all(
                b"POST /v1/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{",
            )
            .await?;
        let mut response = Vec::new();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(250),
                slow_body.read_to_end(&mut response)
            )
            .await
            .is_ok()
        );
        let _ = stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), join).await???;
        Ok(())
    }

    #[tokio::test]
    async fn admission_saturates_before_body_extraction() -> Result<(), Box<dyn std::error::Error>>
    {
        let limits = ServerLimits::new(4096, 1, 1, Duration::from_secs(2), Duration::from_secs(1))?;
        let server = OcrServer::with_backend(limits, FakeBackend);
        let listener = reserve_loopback_listener().await?;
        let address = listener.local_addr()?;
        let stop = CancellationToken::new();
        let serving = server.clone();
        let server_stop = stop.clone();
        let join = tokio::spawn(async move { serving.serve(listener, server_stop).await });
        let partial = b"POST /v1/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{";
        let mut first = TcpStream::connect(address).await?;
        first.write_all(partial).await?;
        while server.state.metrics.active.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let mut second = TcpStream::connect(address).await?;
        second.write_all(partial).await?;
        while server.state.queued.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let mut third = TcpStream::connect(address).await?;
        third
            .write_all(b"POST /v1/ocr HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            .await?;
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_millis(500), third.read_to_end(&mut response))
            .await??;
        assert!(String::from_utf8_lossy(&response).contains("429 Too Many Requests"));
        drop(first);
        drop(second);
        let _ = stop.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(2), join).await?;
        Ok(())
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)] // Keeps one exact cross-transport boundary scenario together.
    async fn configured_image_boundary_matches_http_grpc_and_batch_total()
    -> Result<(), Box<dyn std::error::Error>> {
        use pb::ocr_service_server::OcrService;
        let limits = ServerLimits::new(8, 1, 1, Duration::from_secs(1), Duration::from_secs(1))?;
        let server = OcrServer::with_backend(limits, FakeBackend);
        let raw = |bytes: Vec<u8>| {
            Request::post(http::OCR_RAW_ROUTE)
                .header("content-type", "image/png")
                .header("x-image-format", "png")
                .header("x-image-width", "8")
                .header("x-image-height", "8")
                .body(Body::from(bytes))
        };
        let response = server
            .router()
            .oneshot(raw(b"\x89PNG\r\n\x1a\n".to_vec())?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let response = server
            .router()
            .oneshot(raw(b"\x89PNG\r\n\x1a\nX".to_vec())?)
            .await?;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let exact = request_json();
        let response = server
            .router()
            .oneshot(
                Request::post(http::OCR_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&exact)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let over = json!({"image_base64":STANDARD.encode(b"\x89PNG\r\n\x1a\nX"),"metadata":{"format":"png","encoded_bytes":9,"width":8,"height":8}});
        let response = server
            .router()
            .oneshot(
                Request::post(http::OCR_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&over)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let batch = json!({"items":[exact.clone(),exact]});
        let response = server
            .router()
            .oneshot(
                Request::post(http::OCR_BATCH_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&batch)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let service = GrpcService {
            state: Arc::clone(&server.state),
        };
        let exact_proto = pb::RecognizeRequest {
            image: b"\x89PNG\r\n\x1a\n".to_vec(),
            metadata: Some(pb::InputMetadata {
                format: pb::RasterFormat::Png as i32,
                encoded_bytes: 8,
                width: 8,
                height: 8,
            }),
            options: None,
        };
        assert!(
            service
                .recognize(GrpcRequest::new(exact_proto.clone()))
                .await
                .is_ok()
        );
        let mut over_proto = exact_proto.clone();
        over_proto.image.push(b'X');
        if let Some(metadata) = over_proto.metadata.as_mut() {
            metadata.encoded_bytes = 9;
        }
        assert_eq!(
            service
                .recognize(GrpcRequest::new(over_proto))
                .await
                .err()
                .map(|status| status.code()),
            Some(tonic::Code::InvalidArgument)
        );
        assert_eq!(
            service
                .recognize_batch(GrpcRequest::new(pb::RecognizeBatchRequest {
                    items: vec![exact_proto.clone(), exact_proto.clone()],
                }))
                .await
                .err()
                .map(|status| status.code()),
            Some(tonic::Code::InvalidArgument)
        );

        let listener = reserve_loopback_listener().await?;
        let address = listener.local_addr()?;
        let stop = CancellationToken::new();
        let serving = server.clone();
        let server_stop = stop.clone();
        let join = tokio::spawn(async move { serving.serve_grpc(listener, server_stop).await });
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))?
            .connect()
            .await?;
        let mut client = pb::ocr_service_client::OcrServiceClient::new(channel);
        assert!(client.recognize(exact_proto.clone()).await.is_ok());
        let mut over_proto = exact_proto;
        over_proto.image.push(b'X');
        if let Some(metadata) = over_proto.metadata.as_mut() {
            metadata.encoded_bytes = 9;
        }
        assert_eq!(
            client
                .recognize(over_proto)
                .await
                .err()
                .map(|status| status.code()),
            Some(tonic::Code::InvalidArgument)
        );
        let _ = stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), join).await???;
        Ok(())
    }

    #[tokio::test]
    async fn mcp_notifications_and_explicit_null_follow_json_rpc_lifecycle()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = test_server();
        let notification = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        let response = server
            .router()
            .oneshot(
                Request::post(http::MCP_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&notification)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        for explicit_id in [json!(17), json!("request-17"), Value::Null] {
            let initialized = json!({
                "jsonrpc":"2.0",
                "id": explicit_id,
                "method":"notifications/initialized"
            });
            let response = server
                .router()
                .oneshot(
                    Request::post(http::MCP_ROUTE)
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&initialized)?))?,
                )
                .await?;
            let body: Value =
                serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
            assert_eq!(body["error"]["code"], -32_600);
            assert_eq!(body["id"], explicit_id);
        }
        let explicit_null = json!({"jsonrpc":"2.0","id":null,"method":"tools/list"});
        let response = server
            .router()
            .oneshot(
                Request::post(http::MCP_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&explicit_null)?))?,
            )
            .await?;
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
        assert!(body.get("result").is_some());
        assert_eq!(body["id"], Value::Null);
        let invalid_id = json!({"jsonrpc":"2.0","id":true,"method":"tools/list"});
        let response = server
            .router()
            .oneshot(
                Request::post(http::MCP_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&invalid_id)?))?,
            )
            .await?;
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await?.to_bytes())?;
        assert_eq!(body["error"]["code"], -32_600);
        assert!(
            body["error"]["data"]["request_id"]
                .as_u64()
                .is_some_and(|id| id > 0)
        );

        let (mut client, server_side) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(server_side);
        let serving = server.clone();
        let task =
            tokio::spawn(async move { serving.serve_mcp_io(BufReader::new(read), write).await });
        client
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"method\":\"tools/list\"}\n{\"jsonrpc\":\"2.0\",\"id\":17,\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":\"request-17\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":null,\"method\":\"notifications/initialized\"}\n",
            )
            .await?;
        client.shutdown().await?;
        let mut output = Vec::new();
        client.read_to_end(&mut output).await?;
        task.await??;
        let output_text = String::from_utf8(output)?;
        let responses = output_text
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(responses.len(), 3);
        for (response, expected_id) in
            responses
                .iter()
                .zip([json!(17), json!("request-17"), Value::Null])
        {
            assert_eq!(response["error"]["code"], -32_600);
            assert_eq!(response["id"], expected_id);
        }
        Ok(())
    }
}
