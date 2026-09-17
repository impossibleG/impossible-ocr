//! Explicit, runtime-independent qualification of curated ONNX graph artifacts.

use std::{error::Error, fmt, fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

use crate::{
    ArtifactKind, CURATED_BUNDLE_ID, ContractRole, ModelCatalog, ModelContract,
    ModelContractErrorCode, ModelRole, QualificationStatus, admit_onnx_model, inspect_onnx_model,
    parse_model_contract,
};

/// Stable category for a graph-qualification failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualificationErrorCode {
    /// The curated catalog is internally inconsistent.
    InvalidManifest,
    /// The explicit path did not resolve to a regular file.
    NotRegularFile,
    /// The artifact could not be read completely.
    Io,
    /// The artifact byte length differs from the reviewed manifest.
    ArtifactSizeMismatch,
    /// The artifact digest differs from the reviewed manifest.
    ArtifactDigestMismatch,
    /// The bounded graph parser rejected the artifact.
    Graph(ModelContractErrorCode),
    /// Deterministic contract serialization failed.
    Serialization,
}

/// Privacy-safe graph-qualification error that never retains an artifact path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationError {
    code: QualificationErrorCode,
}

impl QualificationError {
    const fn new(code: QualificationErrorCode) -> Self {
        Self { code }
    }

    /// Returns the stable failure category without paths or tensor values.
    #[must_use]
    pub const fn code(&self) -> QualificationErrorCode {
        self.code
    }
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            QualificationErrorCode::InvalidManifest => "curated model manifest is invalid",
            QualificationErrorCode::NotRegularFile => "artifact is not a regular file",
            QualificationErrorCode::Io => "artifact could not be read",
            QualificationErrorCode::ArtifactSizeMismatch => {
                "artifact size does not match the curated manifest"
            }
            QualificationErrorCode::ArtifactDigestMismatch => {
                "artifact digest does not match the curated manifest"
            }
            QualificationErrorCode::Graph(_) => "artifact graph failed bounded qualification",
            QualificationErrorCode::Serialization => "qualified contract could not be serialized",
        })
    }
}

impl Error for QualificationError {}

#[derive(Clone, Copy)]
struct ExpectedGraph<'a> {
    role: ModelRole,
    byte_length: u64,
    sha256: &'a str,
    contract_id: &'static str,
}

/// Qualifies one explicitly selected curated ONNX artifact without loading a native runtime.
///
/// The file must match the catalog's exact byte length and SHA-256 before the bounded graph parser
/// is invoked. The returned contract is deterministic and contains no local path or host data.
///
/// # Errors
///
/// Returns a privacy-safe category when the catalog is invalid, the file is unavailable or not a
/// regular file, integrity verification fails, or bounded graph inspection rejects the artifact.
pub fn qualify_curated_graph(
    artifact_path: &Path,
    role: ModelRole,
) -> Result<ModelContract, QualificationError> {
    let catalog = ModelCatalog::curated()
        .map_err(|_| QualificationError::new(QualificationErrorCode::InvalidManifest))?;
    let bundle = catalog
        .get(CURATED_BUNDLE_ID)
        .ok_or_else(|| QualificationError::new(QualificationErrorCode::InvalidManifest))?;
    let artifact = bundle
        .artifacts
        .iter()
        .find(|artifact| artifact.role == role && artifact.kind == ArtifactKind::OnnxGraph)
        .ok_or_else(|| QualificationError::new(QualificationErrorCode::InvalidManifest))?;
    let contract_id = match role {
        ModelRole::Detector => "paddlepaddle-pp-ocrv5-mobile-detector-e6f4fa85",
        ModelRole::EnglishRecognizer => "paddlepaddle-pp-ocrv5-english-mobile-recognizer-3fafbc3b",
    };
    qualify_file(
        artifact_path,
        ExpectedGraph {
            role,
            byte_length: artifact.byte_length,
            sha256: artifact.sha256.as_str(),
            contract_id,
        },
    )
}

/// Serializes a qualified contract as canonical pretty JSON terminated by one newline.
///
/// # Errors
///
/// Returns a privacy-safe serialization category if the in-memory document cannot be encoded.
pub fn render_qualified_contract(contract: &ModelContract) -> Result<String, QualificationError> {
    let mut rendered = serde_json::to_string_pretty(contract)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Serialization))?;
    rendered.push('\n');
    Ok(rendered)
}

fn qualify_file(
    artifact_path: &Path,
    expected: ExpectedGraph<'_>,
) -> Result<ModelContract, QualificationError> {
    let mut file = File::open(artifact_path)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Io))?;
    let metadata = file
        .metadata()
        .map_err(|_| QualificationError::new(QualificationErrorCode::Io))?;
    if !metadata.is_file() {
        return Err(QualificationError::new(
            QualificationErrorCode::NotRegularFile,
        ));
    }
    if metadata.len() != expected.byte_length {
        return Err(QualificationError::new(
            QualificationErrorCode::ArtifactSizeMismatch,
        ));
    }
    let length = usize::try_from(expected.byte_length)
        .map_err(|_| QualificationError::new(QualificationErrorCode::ArtifactSizeMismatch))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Io))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Io))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Io))?
        != 0
    {
        return Err(QualificationError::new(
            QualificationErrorCode::ArtifactSizeMismatch,
        ));
    }
    qualify_bytes(&bytes, expected)
}

fn qualify_bytes(
    bytes: &[u8],
    expected: ExpectedGraph<'_>,
) -> Result<ModelContract, QualificationError> {
    if u64::try_from(bytes.len()).ok() != Some(expected.byte_length) {
        return Err(QualificationError::new(
            QualificationErrorCode::ArtifactSizeMismatch,
        ));
    }
    let digest = format!("{:x}", Sha256::digest(bytes));
    if digest != expected.sha256 {
        return Err(QualificationError::new(
            QualificationErrorCode::ArtifactDigestMismatch,
        ));
    }
    let summary = inspect_onnx_model(bytes)
        .map_err(|error| QualificationError::new(QualificationErrorCode::Graph(error.code())))?;
    let role = match expected.role {
        ModelRole::Detector => ContractRole::Detector,
        ModelRole::EnglishRecognizer => ContractRole::EnglishRecognizer,
    };
    let contract = ModelContract {
        schema_version: 1,
        contract_id: expected.contract_id.to_owned(),
        role,
        qualification: QualificationStatus::Qualified,
        artifact_sha256: digest,
        artifact_byte_length: expected.byte_length,
        ir_version: Some(summary.ir_version),
        opsets: summary.opsets,
        inputs: summary.inputs,
        outputs: summary.outputs,
        operators: summary.operators,
    };
    let serialized = serde_json::to_vec(&contract)
        .map_err(|_| QualificationError::new(QualificationErrorCode::Serialization))?;
    let validated = parse_model_contract(&serialized)
        .map_err(|error| QualificationError::new(QualificationErrorCode::Graph(error.code())))?;
    admit_onnx_model(bytes, &validated)
        .map_err(|error| QualificationError::new(QualificationErrorCode::Graph(error.code())))?;
    Ok(validated)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use prost::Message;
    use sha2::{Digest, Sha256};

    use super::{
        ExpectedGraph, QualificationErrorCode, qualify_bytes, qualify_curated_graph,
        render_qualified_contract,
    };
    use crate::{
        ArtifactKind, CURATED_BUNDLE_ID, ContractRole, DimensionContract, ModelCatalog,
        ModelContract, ModelRole, QualificationStatus, detector_contract,
        english_recognizer_contract,
    };

    #[derive(Clone, PartialEq, Message)]
    struct Model {
        #[prost(int64, tag = "1")]
        ir_version: i64,
        #[prost(message, optional, tag = "7")]
        graph: Option<Graph>,
        #[prost(message, repeated, tag = "8")]
        opsets: Vec<Opset>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct Opset {
        #[prost(string, tag = "1")]
        domain: String,
        #[prost(int64, tag = "2")]
        version: i64,
    }

    #[derive(Clone, PartialEq, Message)]
    struct Graph {
        #[prost(message, repeated, tag = "1")]
        nodes: Vec<Node>,
        #[prost(message, repeated, tag = "11")]
        inputs: Vec<ValueInfo>,
        #[prost(message, repeated, tag = "12")]
        outputs: Vec<ValueInfo>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct Node {
        #[prost(string, repeated, tag = "1")]
        inputs: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        outputs: Vec<String>,
        #[prost(string, tag = "4")]
        op_type: String,
    }

    #[derive(Clone, PartialEq, Message)]
    struct ValueInfo {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(message, optional, tag = "2")]
        value_type: Option<ValueType>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct ValueType {
        #[prost(message, optional, tag = "1")]
        tensor: Option<TensorType>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TensorType {
        #[prost(int32, tag = "1")]
        element_type: i32,
        #[prost(message, optional, tag = "2")]
        shape: Option<TensorShape>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TensorShape {
        #[prost(message, repeated, tag = "1")]
        dimensions: Vec<Dimension>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct Dimension {
        #[prost(int64, optional, tag = "1")]
        value: Option<i64>,
    }

    fn tensor(name: &str, dimensions: &[i64]) -> ValueInfo {
        ValueInfo {
            name: name.to_owned(),
            value_type: Some(ValueType {
                tensor: Some(TensorType {
                    element_type: 1,
                    shape: Some(TensorShape {
                        dimensions: dimensions
                            .iter()
                            .map(|value| Dimension {
                                value: Some(*value),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn synthetic_model() -> Vec<u8> {
        Model {
            ir_version: 8,
            graph: Some(Graph {
                nodes: vec![Node {
                    inputs: vec!["x".to_owned()],
                    outputs: vec!["y".to_owned()],
                    op_type: "Relu".to_owned(),
                }],
                inputs: vec![tensor("x", &[1, 3, 32, 32])],
                outputs: vec![tensor("y", &[1, 3, 32, 32])],
            }),
            opsets: vec![Opset {
                domain: String::new(),
                version: 17,
            }],
        }
        .encode_to_vec()
    }

    #[test]
    fn qualification_is_deterministic_and_round_trips_through_admission() {
        let bytes = synthetic_model();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let expected = ExpectedGraph {
            role: ModelRole::Detector,
            byte_length: bytes.len() as u64,
            sha256: &digest,
            contract_id: "synthetic-detector",
        };
        let contract = qualify_bytes(&bytes, expected).unwrap_or_else(|_| unreachable!());
        assert_eq!(contract.qualification, QualificationStatus::Qualified);
        assert_eq!(
            contract.inputs[0].dimensions,
            Some(vec![
                DimensionContract::Fixed(1),
                DimensionContract::Fixed(3),
                DimensionContract::Fixed(32),
                DimensionContract::Fixed(32),
            ])
        );
        let first = render_qualified_contract(&contract).unwrap_or_else(|_| unreachable!());
        let second = render_qualified_contract(&contract).unwrap_or_else(|_| unreachable!());
        assert_eq!(first, second);
        assert!(first.ends_with('\n'));
        assert!(!first.contains("artifact_path"));
    }

    #[test]
    fn integrity_failures_precede_graph_parsing() {
        let bytes = synthetic_model();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut expected = ExpectedGraph {
            role: ModelRole::Detector,
            byte_length: bytes.len() as u64 + 1,
            sha256: &digest,
            contract_id: "synthetic-detector",
        };
        assert_eq!(
            qualify_bytes(&bytes, expected).map_err(|error| error.code()),
            Err(QualificationErrorCode::ArtifactSizeMismatch)
        );
        expected.byte_length = bytes.len() as u64;
        expected.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
        assert_eq!(
            qualify_bytes(&bytes, expected).map_err(|error| error.code()),
            Err(QualificationErrorCode::ArtifactDigestMismatch)
        );
    }

    #[test]
    fn file_errors_do_not_disclose_explicit_paths() {
        let path = Path::new("private-host-path-that-must-not-leak.onnx");
        let error = qualify_curated_graph(path, ModelRole::Detector)
            .err()
            .unwrap_or_else(|| unreachable!());
        let debug = format!("{error:?}");
        let display = error.to_string();
        assert_eq!(error.code(), QualificationErrorCode::Io);
        assert!(!debug.contains("private-host-path"));
        assert!(!display.contains("private-host-path"));
    }

    fn operator_counts(contract: &ModelContract) -> Vec<(&str, u64)> {
        assert!(
            contract
                .operators
                .iter()
                .all(|operator| operator.domain.is_empty())
        );
        contract
            .operators
            .iter()
            .map(|operator| (operator.op_type.as_str(), operator.count))
            .collect()
    }

    #[test]
    #[allow(clippy::too_many_lines)] // Keeping both complete measured goldens together exposes drift.
    fn curated_contracts_are_cross_bound_to_catalog_and_measured_goldens() {
        let catalog = ModelCatalog::curated().unwrap_or_else(|_| unreachable!());
        let bundle = catalog
            .get(CURATED_BUNDLE_ID)
            .unwrap_or_else(|| unreachable!());

        let detector_artifact = bundle
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.role == ModelRole::Detector && artifact.kind == ArtifactKind::OnnxGraph
            })
            .unwrap_or_else(|| unreachable!());
        let detector = detector_contract().unwrap_or_else(|_| unreachable!());
        assert_eq!(
            detector_artifact.revision,
            "e6f4fa85f00e168c862bc462aebca69eef9b3d3d"
        );
        assert_eq!(
            detector.contract_id,
            "paddlepaddle-pp-ocrv5-mobile-detector-e6f4fa85"
        );
        assert_eq!(detector.role, ContractRole::Detector);
        assert_eq!(detector.qualification, QualificationStatus::Qualified);
        assert_eq!(detector.artifact_sha256, detector_artifact.sha256.as_str());
        assert_eq!(detector.artifact_byte_length, detector_artifact.byte_length);
        assert_eq!(detector.artifact_byte_length, 4_826_518);
        assert_eq!(detector.ir_version, Some(6));
        assert_eq!(
            detector
                .opsets
                .iter()
                .map(|opset| (opset.domain.as_str(), opset.version))
                .collect::<Vec<_>>(),
            vec![("", 11)]
        );
        assert_eq!(
            detector.inputs,
            vec![crate::TensorContract {
                name: "x".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Symbol("DynamicDimension.0".to_owned()),
                    DimensionContract::Fixed(3),
                    DimensionContract::Symbol("DynamicDimension.1".to_owned()),
                    DimensionContract::Symbol("DynamicDimension.2".to_owned()),
                ]),
            }]
        );
        assert_eq!(
            detector.outputs,
            vec![crate::TensorContract {
                name: "fetch_name_0".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Symbol("ConvTranspose_521_o0__d0".to_owned()),
                    DimensionContract::Symbol("ConvTranspose_521_o0__d1".to_owned()),
                    DimensionContract::Symbol("ConvTranspose_521_o0__d2".to_owned()),
                    DimensionContract::Symbol("ConvTranspose_521_o0__d3".to_owned()),
                ]),
            }]
        );
        assert_eq!(
            operator_counts(&detector),
            vec![
                ("Add", 117),
                ("BatchNormalization", 3),
                ("Concat", 1),
                ("Conv", 62),
                ("ConvTranspose", 2),
                ("GlobalAveragePool", 10),
                ("HardSigmoid", 34),
                ("Identity", 192),
                ("Mul", 86),
                ("Relu", 12),
                ("Resize", 6),
                ("Sigmoid", 1),
            ]
        );

        let recognizer_artifact = bundle
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.role == ModelRole::EnglishRecognizer
                    && artifact.kind == ArtifactKind::OnnxGraph
            })
            .unwrap_or_else(|| unreachable!());
        let recognizer = english_recognizer_contract().unwrap_or_else(|_| unreachable!());
        assert_eq!(
            recognizer_artifact.revision,
            "3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b"
        );
        assert_eq!(
            recognizer.contract_id,
            "paddlepaddle-pp-ocrv5-english-mobile-recognizer-3fafbc3b"
        );
        assert_eq!(recognizer.role, ContractRole::EnglishRecognizer);
        assert_eq!(recognizer.qualification, QualificationStatus::Qualified);
        assert_eq!(
            recognizer.artifact_sha256,
            recognizer_artifact.sha256.as_str()
        );
        assert_eq!(
            recognizer.artifact_byte_length,
            recognizer_artifact.byte_length
        );
        assert_eq!(recognizer.artifact_byte_length, 7_848_423);
        assert_eq!(recognizer.ir_version, Some(3));
        assert_eq!(
            recognizer
                .opsets
                .iter()
                .map(|opset| (opset.domain.as_str(), opset.version))
                .collect::<Vec<_>>(),
            vec![("", 7)]
        );
        assert_eq!(
            recognizer.inputs,
            vec![crate::TensorContract {
                name: "x".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Symbol("DynamicDimension.0".to_owned()),
                    DimensionContract::Fixed(3),
                    DimensionContract::Fixed(48),
                    DimensionContract::Symbol("DynamicDimension.1".to_owned()),
                ]),
            }]
        );
        assert_eq!(
            recognizer.outputs,
            vec![crate::TensorContract {
                name: "fetch_name_0".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Symbol("DynamicDimension.2".to_owned()),
                    DimensionContract::Symbol("DynamicDimension.3".to_owned()),
                    DimensionContract::Fixed(438),
                ]),
            }]
        );
        assert_eq!(
            operator_counts(&recognizer),
            vec![
                ("Add", 114),
                ("AveragePool", 1),
                ("BatchNormalization", 6),
                ("Concat", 3),
                ("Constant", 330),
                ("Conv", 38),
                ("Div", 5),
                ("GlobalAveragePool", 2),
                ("HardSigmoid", 30),
                ("Identity", 245),
                ("MatMul", 13),
                ("Mul", 100),
                ("Pow", 5),
                ("ReduceMean", 10),
                ("Relu", 2),
                ("Reshape", 52),
                ("Shape", 7),
                ("Sigmoid", 7),
                ("Slice", 13),
                ("Softmax", 3),
                ("Sqrt", 5),
                ("Squeeze", 9),
                ("Sub", 5),
                ("Transpose", 9),
                ("Unsqueeze", 5),
            ]
        );
    }
}
