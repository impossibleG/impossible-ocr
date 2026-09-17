//! Explicit offline import for the curated OCR model bundle.

use std::path::PathBuf;

use clap::Parser;
use impossible_ocr_onnx::{
    CURATED_BUNDLE_ID, ImportBundle, InstallOutcome, ModelCatalog, ModelStore, ModelStoreError,
    ModelStoreErrorCode,
};
use impossible_server_core::CancellationToken;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "import-model-bundle",
    about = "Import the pinned OCR bundle from four already-downloaded local files"
)]
struct Args {
    /// Absolute path to the destination model store.
    #[arg(long)]
    model_store: PathBuf,
    /// Exact curated model profile.
    #[arg(long, default_value = CURATED_BUNDLE_ID)]
    profile: String,
    /// Pinned detector ONNX file.
    #[arg(long)]
    detector_graph: PathBuf,
    /// Pinned detector inference.yml file.
    #[arg(long)]
    detector_config: PathBuf,
    /// Pinned English recognizer ONNX file.
    #[arg(long)]
    recognizer_graph: PathBuf,
    /// Pinned English recognizer inference.yml file.
    #[arg(long)]
    recognizer_config: PathBuf,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImportErrorCode {
    Configuration,
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
    ModelOperationFailed,
}

#[derive(Serialize)]
struct FailureReport {
    schema_version: u32,
    status: &'static str,
    error: ImportErrorCode,
}

#[derive(Serialize)]
struct SuccessReport {
    schema_version: u32,
    status: &'static str,
    model: &'static str,
    outcome: &'static str,
}

#[tokio::main]
async fn main() {
    let Ok(args) = Args::try_parse() else {
        return emit_failure(ImportErrorCode::Configuration, 2);
    };
    match import(args).await {
        Ok(outcome) => println!("{}", success_json(outcome)),
        Err(code) => emit_failure(code, 1),
    }
}

async fn import(args: Args) -> Result<InstallOutcome, ImportErrorCode> {
    if args.profile != CURATED_BUNDLE_ID {
        return Err(ImportErrorCode::Configuration);
    }
    let catalog = ModelCatalog::curated().map_err(|_| ImportErrorCode::Configuration)?;
    let store = ModelStore::open(args.model_store, catalog)
        .await
        .map_err(|error| model_error(&error))?;
    let source = ImportBundle {
        detector_graph: args.detector_graph,
        detector_config: args.detector_config,
        recognizer_graph: args.recognizer_graph,
        recognizer_config: args.recognizer_config,
    };
    let outcome = store
        .import(CURATED_BUNDLE_ID, &source, &CancellationToken::new())
        .await
        .map_err(|error| model_error(&error))?;
    store
        .verify(CURATED_BUNDLE_ID)
        .await
        .map_err(|error| model_error(&error))?;
    Ok(outcome)
}

fn model_error(error: &ModelStoreError) -> ImportErrorCode {
    match error.code() {
        ModelStoreErrorCode::InvalidManifest => ImportErrorCode::InvalidManifest,
        ModelStoreErrorCode::UnsafeFilesystem => ImportErrorCode::UnsafeFilesystem,
        ModelStoreErrorCode::UnapprovedSource => ImportErrorCode::UnapprovedSource,
        ModelStoreErrorCode::IntegrityMismatch => ImportErrorCode::IntegrityMismatch,
        ModelStoreErrorCode::ConfigurationMismatch => ImportErrorCode::ConfigurationMismatch,
        ModelStoreErrorCode::Cancelled => ImportErrorCode::Cancelled,
        ModelStoreErrorCode::TimedOut => ImportErrorCode::TimedOut,
        ModelStoreErrorCode::InUse => ImportErrorCode::InUse,
        ModelStoreErrorCode::NotInstalled => ImportErrorCode::NotInstalled,
        ModelStoreErrorCode::Io => ImportErrorCode::Io,
        _ => ImportErrorCode::ModelOperationFailed,
    }
}

fn emit_failure(code: ImportErrorCode, exit_code: i32) {
    println!("{}", failure_json(code));
    std::process::exit(exit_code);
}

fn failure_json(code: ImportErrorCode) -> String {
    let report = FailureReport {
        schema_version: 1,
        status: "failed",
        error: code,
    };
    serde_json::to_string(&report).unwrap_or_else(|_| {
        r#"{"schema_version":1,"status":"failed","error":"model_operation_failed"}"#.into()
    })
}

fn success_json(outcome: InstallOutcome) -> String {
    let outcome = match outcome {
        InstallOutcome::Installed => "installed",
        InstallOutcome::Repaired => "repaired",
        InstallOutcome::AlreadyVerified => "already_verified",
    };
    serde_json::to_string(&SuccessReport {
        schema_version: 1,
        status: "ok",
        model: CURATED_BUNDLE_ID,
        outcome,
    })
    .unwrap_or_else(|_| {
        r#"{"schema_version":1,"status":"failed","error":"model_operation_failed"}"#.into()
    })
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use impossible_ocr_onnx::{CURATED_BUNDLE_ID, InstallOutcome};

    use super::{Args, ImportErrorCode, failure_json, success_json};

    #[test]
    fn complete_arguments_select_only_the_curated_profile() {
        let args = Args::try_parse_from([
            "import-model-bundle",
            "--model-store",
            "models",
            "--detector-graph",
            "detector.onnx",
            "--detector-config",
            "detector.yml",
            "--recognizer-graph",
            "recognizer.onnx",
            "--recognizer-config",
            "recognizer.yml",
        ])
        .unwrap_or_else(|_| unreachable!());
        assert_eq!(args.profile, CURATED_BUNDLE_ID);
    }

    #[test]
    fn missing_artifact_argument_is_rejected() {
        assert!(Args::try_parse_from(["import-model-bundle", "--model-store", "models"]).is_err());
    }

    #[test]
    fn reports_are_deterministic_and_never_echo_paths() {
        assert_eq!(
            success_json(InstallOutcome::Installed),
            r#"{"schema_version":1,"status":"ok","model":"paddlex-ocr-3.7-max960","outcome":"installed"}"#
        );
        let failure = failure_json(ImportErrorCode::IntegrityMismatch);
        assert_eq!(
            failure,
            r#"{"schema_version":1,"status":"failed","error":"integrity_mismatch"}"#
        );
        assert!(!failure.contains("private"));
    }
}
