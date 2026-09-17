//! Verified, offline-first model lifecycle for the Impossible OCR ONNX backend.
//!
//! Merely constructing the backend or model store performs no network access. Model
//! installation is an explicit operation and inference runtime admission remains separate.

mod graph_contract;
mod lifecycle;
mod qualification;
#[cfg(feature = "onnx-runtime")]
mod runtime;

pub use graph_contract::{
    ContractRole, DimensionContract, MAX_ONNX_MODEL_BYTES, ModelContract, ModelContractError,
    ModelContractErrorCode, OnnxGraphSummary, OperatorContract, OpsetContract, QualificationStatus,
    TensorContract, admit_onnx_model, detector_contract, english_recognizer_contract,
    inspect_onnx_model, parse_model_contract,
};
pub use lifecycle::{
    ArtifactKind, CURATED_BUNDLE_ID, DeleteOutcome, ImportBundle, InstallOutcome, ModelArtifact,
    ModelBundleManifest, ModelCatalog, ModelLease, ModelRole, ModelStore, ModelStoreError,
    ModelStoreErrorCode, ModelStoreStatus, Sha256Digest, VerifiedBundle,
};
pub use qualification::{
    QualificationError, QualificationErrorCode, qualify_curated_graph, render_qualified_contract,
};
#[cfg(feature = "onnx-runtime")]
pub use runtime::{OrtCpuRuntime, OrtRuntimeConfig};

use impossible_ocr_domain::{OcrError, OcrErrorCode, OcrOptions, OcrResult, RasterInput};
use impossible_ocr_pipeline::{BackendFuture, BackendState, OcrBackend};
use impossible_server_core::RequestContext;

/// Placeholder proving fail-closed behavior before an ONNX runtime is admitted.
#[derive(Debug, Default)]
pub struct UnavailableOnnxBackend;

impl OcrBackend for UnavailableOnnxBackend {
    fn state(&self) -> BackendState {
        BackendState::Missing
    }

    fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
        Box::pin(async { Err(OcrError::for_code(OcrErrorCode::ModelUnavailable)) })
    }

    fn recognize<'a>(
        &'a self,
        _input: &'a RasterInput,
        _options: OcrOptions,
        _context: &'a RequestContext,
    ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
        Box::pin(async { Err(OcrError::for_code(OcrErrorCode::ModelUnavailable)) })
    }
}

#[cfg(feature = "onnx-runtime")]
/// Pinned `ort` API version used by the future session adapter.
///
/// Merely enabling this feature does not download or load a native ONNX Runtime library.
pub const ORT_CRATE_VERSION: &str = "2.0.0-rc.13";

#[cfg(test)]
mod tests {
    use super::UnavailableOnnxBackend;
    use impossible_ocr_pipeline::{BackendState, OcrBackend};

    #[test]
    fn unavailable_backend_is_fail_closed() {
        assert_eq!(UnavailableOnnxBackend.state(), BackendState::Missing);
    }
}
