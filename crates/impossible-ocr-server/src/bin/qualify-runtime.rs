//! Opt-in, offline real-runtime qualification command.

#[cfg(feature = "onnx-runtime")]
use std::path::PathBuf;

#[cfg(feature = "onnx-runtime")]
use clap::Parser;
#[cfg(feature = "onnx-runtime")]
use impossible_ocr_server::{
    RuntimeQualificationConfig, RuntimeQualificationErrorCode, qualify_installed_runtime,
};
#[cfg(feature = "onnx-runtime")]
use serde::Serialize;

#[cfg(feature = "onnx-runtime")]
#[derive(Debug, Parser)]
#[command(
    name = "qualify-runtime",
    about = "Offline qualification of explicitly installed OCR artifacts"
)]
struct Args {
    /// Absolute path to the already-installed verified model store.
    #[arg(long)]
    model_store: PathBuf,
    /// Exact curated model profile.
    #[arg(long, default_value = "paddlex-ocr-3.7-max960")]
    profile: String,
    /// Absolute path to a reviewed, fully qualified runtime manifest.
    #[arg(long)]
    runtime_manifest: PathBuf,
    /// Absolute path to the manifest's installed content-addressed runtime directory.
    #[arg(long)]
    runtime_directory: PathBuf,
    /// Exact reviewed platform identifier.
    #[arg(long)]
    platform: String,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Serialize)]
struct FailureReport {
    schema_version: u32,
    status: &'static str,
    error: RuntimeQualificationErrorCode,
}

#[cfg(feature = "onnx-runtime")]
#[tokio::main]
async fn main() {
    let Ok(args) = Args::try_parse() else {
        return emit_failure(RuntimeQualificationErrorCode::Configuration, 2);
    };
    let config = match RuntimeQualificationConfig::from_manifest_file(
        args.model_store,
        &args.profile,
        args.runtime_manifest,
        args.runtime_directory,
        &args.platform,
    ) {
        Ok(config) => config,
        Err(error) => return emit_failure(error.code(), 1),
    };
    match qualify_installed_runtime(config).await {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => println!("{json}"),
            Err(_) => emit_failure(RuntimeQualificationErrorCode::QualificationFailed, 1),
        },
        Err(error) => emit_failure(error.code(), 1),
    }
}

#[cfg(feature = "onnx-runtime")]
fn emit_failure(code: RuntimeQualificationErrorCode, exit_code: i32) {
    println!("{}", failure_json(code));
    std::process::exit(exit_code);
}

#[cfg(feature = "onnx-runtime")]
fn failure_json(code: RuntimeQualificationErrorCode) -> String {
    let report = FailureReport {
        schema_version: 1,
        status: "failed",
        error: code,
    };
    serde_json::to_string(&report).unwrap_or_else(|_| {
        r#"{"schema_version":1,"status":"failed","error":"qualification_failed"}"#.into()
    })
}

#[cfg(not(feature = "onnx-runtime"))]
fn main() {
    println!("{{\"schema_version\":1,\"status\":\"failed\",\"error\":\"feature_disabled\"}}");
    std::process::exit(2);
}

#[cfg(all(test, feature = "onnx-runtime"))]
mod tests {
    use impossible_ocr_server::RuntimeQualificationErrorCode;

    #[test]
    fn failure_output_is_exact_json_without_argument_echo() {
        let output = super::failure_json(RuntimeQualificationErrorCode::Configuration);
        assert_eq!(
            output,
            r#"{"schema_version":1,"status":"failed","error":"configuration"}"#
        );
        assert!(!output.contains("private"));
    }
}
