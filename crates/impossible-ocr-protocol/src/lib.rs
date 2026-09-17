//! Stable protocol contracts shared by HTTP, gRPC, and MCP adapters.

use impossible_ocr_domain::{InputMetadata, OcrError, OcrOptions, OcrResult};
use serde::{Deserialize, Serialize};

/// Maximum number of raster items in one synchronous batch.
pub const MAX_BATCH_ITEMS: usize = 8;
/// Maximum encoded PNG or JPEG bytes in one image.
pub const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;
/// Maximum aggregate encoded image bytes in one batch.
pub const MAX_BATCH_IMAGE_BYTES: usize = MAX_IMAGE_BYTES;
/// Product-level maximum base64 characters for one maximum-size image.
pub const MAX_BASE64_IMAGE_BYTES: usize = 44_739_244;
/// Product-level maximum JSON/base64 body, including bounded batch metadata overhead.
pub const MAX_JSON_WIRE_BYTES: usize = MAX_BASE64_IMAGE_BYTES + 16 * 1024;
/// Product-level maximum MCP JSON-RPC body.
pub const MAX_MCP_WIRE_BYTES: usize = MAX_JSON_WIRE_BYTES + 1024;
/// Product-level maximum protobuf body, including bounded framing and metadata.
pub const MAX_GRPC_WIRE_BYTES: usize = MAX_IMAGE_BYTES + 4 * 1024;

/// Returns the canonical padded base64 length, or `None` on arithmetic overflow.
#[must_use]
pub const fn base64_wire_len(decoded: usize) -> Option<usize> {
    match decoded.checked_add(2) {
        Some(value) => match value.checked_div(3) {
            Some(groups) => groups.checked_mul(4),
            None => None,
        },
        None => None,
    }
}

/// Versioned HTTP boundary.
pub mod http {
    use super::{
        InputMetadata, MAX_BATCH_IMAGE_BYTES, MAX_BATCH_ITEMS, MAX_GRPC_WIRE_BYTES,
        MAX_IMAGE_BYTES, MAX_JSON_WIRE_BYTES, MAX_MCP_WIRE_BYTES, OcrOptions, OcrResult,
    };
    use serde::{Deserialize, Serialize};

    /// Canonical JSON/base64 synchronous OCR route.
    pub const OCR_ROUTE: &str = "/v1/ocr";
    /// Native raw-raster synchronous OCR route.
    pub const OCR_RAW_ROUTE: &str = "/v1/ocr:raw";
    /// Bounded synchronous batch route.
    pub const OCR_BATCH_ROUTE: &str = "/v1/ocr:batch";
    /// Capability discovery route.
    pub const CAPABILITIES_ROUTE: &str = "/v1/capabilities";
    /// Installed-model status route.
    pub const MODELS_ROUTE: &str = "/v1/models";
    /// Aggregate process/backend status route.
    pub const STATUS_ROUTE: &str = "/v1/status";
    /// Streamable HTTP MCP JSON-RPC route.
    pub const MCP_ROUTE: &str = "/mcp";

    /// JSON request carrying one base64-encoded raster.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct OcrRequest {
        /// Base64-encoded PNG or JPEG bytes.
        pub image_base64: String,
        /// Claimed input metadata, validated again after decoding.
        pub metadata: InputMetadata,
        /// OCR behavior.
        #[serde(default)]
        pub options: OcrOptions,
    }

    /// Successful synchronous response.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct OcrResponse {
        /// Canonical result.
        pub result: OcrResult,
    }

    /// Bounded all-or-nothing batch request.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct BatchRequest {
        /// Between one and [`MAX_BATCH_ITEMS`] requests.
        pub items: Vec<OcrRequest>,
    }

    impl BatchRequest {
        /// Validates batch cardinality before decoding any item.
        #[must_use]
        pub fn has_valid_cardinality(&self) -> bool {
            !self.items.is_empty() && self.items.len() <= MAX_BATCH_ITEMS
        }
    }

    /// Successful batch response in input order.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct BatchResponse {
        /// One result for every request item.
        pub results: Vec<OcrResult>,
    }

    /// Stable advertised transport and product capabilities.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CapabilitiesResponse {
        /// Protocol contract version.
        pub api_version: &'static str,
        /// Accepted encoded raster formats.
        pub raster_formats: [&'static str; 2],
        /// Maximum items per synchronous batch.
        pub max_batch_items: usize,
        /// Maximum encoded PNG/JPEG bytes in one image after base64/protobuf decoding.
        pub max_image_bytes: usize,
        /// Maximum aggregate encoded image bytes across a batch.
        pub max_batch_image_bytes: usize,
        /// Maximum raw HTTP request body bytes.
        pub max_raw_wire_bytes: usize,
        /// Maximum JSON/base64 HTTP request body bytes.
        pub max_json_wire_bytes: usize,
        /// Maximum MCP JSON-RPC request bytes.
        pub max_mcp_wire_bytes: usize,
        /// Maximum accepted gRPC protobuf message bytes.
        pub max_grpc_wire_bytes: usize,
        /// Whether PDFs are accepted.
        pub supports_pdf: bool,
        /// Whether asynchronous jobs are accepted.
        pub supports_jobs: bool,
        /// Whether `WebSockets` are exposed.
        pub supports_websocket: bool,
    }

    impl Default for CapabilitiesResponse {
        fn default() -> Self {
            Self {
                api_version: "v1",
                raster_formats: ["png", "jpeg"],
                max_batch_items: MAX_BATCH_ITEMS,
                max_image_bytes: MAX_IMAGE_BYTES,
                max_batch_image_bytes: MAX_BATCH_IMAGE_BYTES,
                max_raw_wire_bytes: MAX_IMAGE_BYTES,
                max_json_wire_bytes: MAX_JSON_WIRE_BYTES,
                max_mcp_wire_bytes: MAX_MCP_WIRE_BYTES,
                max_grpc_wire_bytes: MAX_GRPC_WIRE_BYTES,
                supports_pdf: false,
                supports_jobs: false,
                supports_websocket: false,
            }
        }
    }

    /// Privacy-safe backend model status.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ModelsResponse {
        /// Stable aggregate lifecycle state.
        pub state: &'static str,
        /// Curated bundle identifiers. Empty when no verified bundle is ready.
        pub models: Vec<&'static str>,
    }

    /// Aggregate process/backend status.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct StatusResponse {
        /// Whether the process is live.
        pub live: bool,
        /// Whether OCR work can be served.
        pub ready: bool,
        /// Stable aggregate reason when not ready.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reason: Option<&'static str>,
    }
}

/// Generated protobuf messages, service, and client for `impossible.ocr.v1`.
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod grpc {
    tonic::include_proto!("impossible.ocr.v1");
}

/// MCP tool names and strict JSON schemas.
pub mod mcp {
    /// Single-image OCR tool.
    pub const OCR_RECOGNIZE: &str = "ocr_recognize";
    /// Bounded batch OCR tool.
    pub const OCR_BATCH: &str = "ocr_batch";
    /// Capability discovery tool.
    pub const OCR_CAPABILITIES: &str = "ocr_capabilities";
    /// Model status tool.
    pub const OCR_MODELS: &str = "ocr_models";
    /// Tool names in stable discovery order.
    pub const TOOL_NAMES: [&str; 4] = [OCR_RECOGNIZE, OCR_BATCH, OCR_CAPABILITIES, OCR_MODELS];
    /// Strict schema for a single OCR request.
    pub const RECOGNIZE_SCHEMA: &str = include_str!("../schemas/ocr-recognize.schema.json");
    /// Strict schema for a bounded batch request.
    pub const BATCH_SCHEMA: &str = include_str!("../schemas/ocr-batch.schema.json");
    /// Strict empty-object schema for discovery calls.
    pub const EMPTY_SCHEMA: &str =
        r#"{"type":"object","properties":{},"additionalProperties":false}"#;
}

/// Public error body shared by JSON-based adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    /// Stable machine-readable code.
    pub code: String,
    /// Static privacy-safe description.
    pub message: String,
}

impl From<OcrError> for ErrorBody {
    fn from(error: OcrError) -> Self {
        Self {
            code: error.code().as_str().to_owned(),
            message: error.message().to_owned(),
        }
    }
}

/// JSON error envelope with a process-local correlation identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    /// Public error.
    pub error: ErrorBody,
    /// Numeric request identifier also returned in the `x-request-id` header.
    pub request_id: u64,
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_BASE64_IMAGE_BYTES, MAX_BATCH_ITEMS, MAX_IMAGE_BYTES, base64_wire_len, http, mcp,
    };

    #[test]
    fn schemas_are_strict_and_capabilities_refuse_unimplemented_surfaces() {
        for schema in [mcp::RECOGNIZE_SCHEMA, mcp::BATCH_SCHEMA, mcp::EMPTY_SCHEMA] {
            let value: serde_json::Value = serde_json::from_str(schema)
                .unwrap_or_else(|_| unreachable!("checked-in schema must be valid JSON"));
            assert_eq!(value["additionalProperties"], false);
        }
        let capabilities = http::CapabilitiesResponse::default();
        assert_eq!(capabilities.max_batch_items, MAX_BATCH_ITEMS);
        assert_eq!(capabilities.max_image_bytes, MAX_IMAGE_BYTES);
        assert_eq!(
            base64_wire_len(MAX_IMAGE_BYTES),
            Some(MAX_BASE64_IMAGE_BYTES)
        );
        assert!(capabilities.max_json_wire_bytes >= MAX_BASE64_IMAGE_BYTES);
        assert!(capabilities.max_mcp_wire_bytes >= capabilities.max_json_wire_bytes);
        assert!(capabilities.max_grpc_wire_bytes >= MAX_IMAGE_BYTES);
        assert_eq!(base64_wire_len(1), Some(4));
        assert_eq!(base64_wire_len(3), Some(4));
        assert_eq!(base64_wire_len(4), Some(8));
        assert!(!capabilities.supports_pdf);
        assert!(!capabilities.supports_jobs);
        assert!(!capabilities.supports_websocket);
    }

    #[test]
    fn batch_cardinality_is_bounded() {
        let empty = http::BatchRequest { items: Vec::new() };
        assert!(!empty.has_valid_cardinality());
    }
}
