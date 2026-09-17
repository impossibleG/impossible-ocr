//! Command-line entry point for the Impossible OCR server shell.

#[cfg(feature = "onnx-runtime")]
use std::path::PathBuf;
use std::{net::SocketAddr, time::Duration};

use clap::Parser;
use impossible_ocr_pipeline::OcrBackend;
use impossible_ocr_server::OcrServer;
#[cfg(feature = "onnx-runtime")]
use impossible_ocr_server::{OrtServerConfig, build_ort_server};
use impossible_server_core::{CancellationToken, ServerLimits};
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(
    name = "impossible-ocr",
    version,
    about = "Local-first raster OCR server"
)]
struct Args {
    /// Loopback bind address. Public binding is intentionally not available in v0.1.
    #[arg(long, env = "IMPOSSIBLE_OCR_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    /// Loopback gRPC bind address.
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_GRPC_BIND",
        default_value = "127.0.0.1:50051"
    )]
    grpc_bind: SocketAddr,
    /// Run newline-delimited MCP on stdio instead of opening network listeners.
    #[arg(long, env = "IMPOSSIBLE_OCR_MCP_STDIO", default_value_t = false)]
    mcp_stdio: bool,
    /// Maximum encoded PNG/JPEG bytes in one image after base64/protobuf decoding.
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_MAX_REQUEST_BYTES",
        default_value_t = 33_554_432
    )]
    max_request_bytes: usize,
    /// Maximum queued requests.
    #[arg(long, env = "IMPOSSIBLE_OCR_QUEUE_CAPACITY", default_value_t = 32)]
    queue_capacity: usize,
    /// Maximum concurrently executing OCR requests.
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_MAX_CONCURRENT_REQUESTS",
        default_value_t = 2
    )]
    max_concurrent_requests: usize,
    /// Total request deadline in milliseconds.
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_REQUEST_TIMEOUT_MS",
        default_value_t = 30_000
    )]
    request_timeout_ms: u64,
    /// Total graceful shutdown deadline in milliseconds.
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_SHUTDOWN_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    shutdown_timeout_ms: u64,
    /// Absolute directory containing the already-installed curated model bundle.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_MODEL_STORE")]
    model_store: Option<PathBuf>,
    /// Exact curated model profile. Arbitrary model identifiers are rejected.
    #[cfg(feature = "onnx-runtime")]
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_MODEL_PROFILE",
        default_value = "paddlex-ocr-3.7-max960"
    )]
    model_profile: String,
    /// Absolute directory containing the native ONNX Runtime library.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_DIRECTORY")]
    runtime_directory: Option<PathBuf>,
    /// Native ONNX Runtime library filename beneath `runtime-directory`.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_LIBRARY")]
    runtime_library: Option<PathBuf>,
    /// Exact native ONNX Runtime library byte length.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_LIBRARY_BYTES")]
    runtime_library_bytes: Option<u64>,
    /// Lowercase SHA-256 of the native ONNX Runtime library.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_LIBRARY_SHA256")]
    runtime_library_sha256: Option<String>,
    /// Fixed number of detector/recognizer session lanes.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_LANES", default_value_t = 1)]
    runtime_lanes: usize,
    /// Intra-op CPU threads per session.
    #[cfg(feature = "onnx-runtime")]
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_RUNTIME_INTRA_THREADS",
        default_value_t = 1
    )]
    runtime_intra_threads: usize,
    /// Inter-op CPU threads per session. Sequential execution remains enforced.
    #[cfg(feature = "onnx-runtime")]
    #[arg(
        long,
        env = "IMPOSSIBLE_OCR_RUNTIME_INTER_THREADS",
        default_value_t = 1
    )]
    runtime_inter_threads: usize,
    /// Total CPU-thread budget across all session lanes.
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, env = "IMPOSSIBLE_OCR_RUNTIME_CPU_BUDGET", default_value_t = 2)]
    runtime_cpu_budget: usize,
}

#[cfg(feature = "onnx-runtime")]
impl Args {
    fn ort_config(&self) -> Result<Option<OrtServerConfig>, &'static str> {
        let supplied = [
            self.model_store.is_some(),
            self.runtime_directory.is_some(),
            self.runtime_library.is_some(),
            self.runtime_library_bytes.is_some(),
            self.runtime_library_sha256.is_some(),
        ];
        if supplied.iter().all(|value| !value) {
            if self.model_profile != impossible_ocr_onnx::CURATED_BUNDLE_ID
                || self.runtime_lanes != 1
                || self.runtime_intra_threads != 1
                || self.runtime_inter_threads != 1
                || self.runtime_cpu_budget != 2
            {
                return Err(
                    "local ONNX backend options require the complete artifact configuration",
                );
            }
            return Ok(None);
        }
        if supplied.iter().any(|value| !value) {
            return Err("the local ONNX backend configuration is incomplete");
        }
        let config = OrtServerConfig::new(
            self.model_store
                .clone()
                .ok_or("the local ONNX backend configuration is incomplete")?,
            &self.model_profile,
            self.runtime_directory
                .clone()
                .ok_or("the local ONNX backend configuration is incomplete")?,
            self.runtime_library
                .clone()
                .ok_or("the local ONNX backend configuration is incomplete")?,
            self.runtime_library_bytes
                .ok_or("the local ONNX backend configuration is incomplete")?,
            self.runtime_library_sha256
                .clone()
                .ok_or("the local ONNX backend configuration is incomplete")?,
            self.runtime_cpu_budget,
        )
        .map_err(|_| "the local ONNX backend configuration is invalid")?
        .with_runtime_limits(
            self.runtime_lanes,
            self.runtime_intra_threads,
            self.runtime_inter_threads,
        )
        .map_err(|_| "the local ONNX backend resource limits are invalid")?;
        Ok(Some(config))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.bind.ip().is_loopback() || !args.grpc_bind.ip().is_loopback() {
        return Err("v0.1 only permits loopback binding".into());
    }
    let limits = ServerLimits::new(
        args.max_request_bytes,
        args.queue_capacity,
        args.max_concurrent_requests,
        Duration::from_millis(args.request_timeout_ms),
        Duration::from_millis(args.shutdown_timeout_ms),
    )?;
    #[cfg(feature = "onnx-runtime")]
    if let Some(config) = args.ort_config()? {
        let server = build_ort_server(limits, config).await?;
        return run_server(&args, server).await;
    }
    run_server(&args, OcrServer::new(limits)).await
}

async fn run_server<B: OcrBackend>(
    args: &Args,
    server: OcrServer<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = signal.cancel();
        }
    });
    if args.mcp_stdio {
        server.serve_mcp_stdio_until(cancellation).await?;
        return Ok(());
    }
    if args.bind == args.grpc_bind {
        return Err("HTTP and gRPC bind addresses must differ".into());
    }
    let http_listener = TcpListener::bind(args.bind).await?;
    let grpc_listener = TcpListener::bind(args.grpc_bind).await?;
    let grpc_server = server.clone();
    let grpc_stop = cancellation.clone();
    tokio::try_join!(
        server.serve(http_listener, cancellation),
        grpc_server.serve_grpc(grpc_listener, grpc_stop),
    )?;
    Ok(())
}

#[cfg(all(test, feature = "onnx-runtime"))]
mod tests {
    use clap::Parser;

    use super::Args;

    fn absolute_fixture(name: &str) -> std::path::PathBuf {
        std::env::current_exe()
            .unwrap_or_else(|_| unreachable!())
            .parent()
            .unwrap_or_else(|| unreachable!())
            .join(name)
    }

    #[test]
    fn local_runtime_arguments_are_atomic_and_budgeted() {
        let partial = Args::try_parse_from([
            "impossible-ocr",
            "--model-store",
            absolute_fixture("models").to_string_lossy().as_ref(),
        ])
        .unwrap_or_else(|_| unreachable!());
        assert!(partial.ort_config().is_err());

        let complete = Args::try_parse_from([
            "impossible-ocr",
            "--model-store",
            absolute_fixture("models").to_string_lossy().as_ref(),
            "--runtime-directory",
            absolute_fixture("runtime").to_string_lossy().as_ref(),
            "--runtime-library",
            "runtime.bin",
            "--runtime-library-bytes",
            "1",
            "--runtime-library-sha256",
            &"a".repeat(64),
            "--runtime-lanes",
            "2",
            "--runtime-intra-threads",
            "2",
            "--runtime-cpu-budget",
            "3",
        ])
        .unwrap_or_else(|_| unreachable!());
        assert!(complete.ort_config().is_err());

        for arguments in [
            vec!["impossible-ocr", "--model-profile", "unreviewed"],
            vec!["impossible-ocr", "--runtime-lanes", "2"],
            vec!["impossible-ocr", "--runtime-lanes", "0"],
            vec!["impossible-ocr", "--runtime-cpu-budget", "0"],
        ] {
            let args = Args::try_parse_from(arguments).unwrap_or_else(|_| unreachable!());
            assert!(args.ort_config().is_err());
        }
    }
}
