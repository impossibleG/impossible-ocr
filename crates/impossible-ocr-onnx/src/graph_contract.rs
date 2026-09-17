//! Bounded, fail-closed admission for reviewed ONNX inference graphs.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum serialized ONNX graph size accepted by the admission parser.
pub const MAX_ONNX_MODEL_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROTO_FIELDS: usize = 250_000;
const MAX_PROTO_DEPTH: usize = 32;
const MAX_TEXT_BYTES: usize = 4 * 1024;
const MAX_CONTRACT_BYTES: usize = 64 * 1024;
const MAX_GRAPH_IO: usize = 128;
const MAX_OPSETS: usize = 64;
const MAX_GRAPH_NODES: usize = 8_192;
const MAX_GRAPH_ATTRIBUTES: usize = 65_536;
const MAX_GRAPH_TENSORS: usize = 16_384;
const MAX_GRAPH_VALUES: usize = 65_536;
const MAX_TENSOR_RANK: usize = 16;

/// Stable category for an ONNX model-contract admission failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelContractErrorCode {
    /// The serialized model exceeds the fixed admission cap.
    ModelTooLarge,
    /// The protobuf wire representation or graph structure is malformed.
    MalformedModel,
    /// A fixed structural or recursion budget was exceeded.
    LimitExceeded,
    /// The model uses a deliberately unsupported ONNX feature.
    UnsupportedModel,
    /// The contract has not completed real-model qualification.
    ProvisionalContract,
    /// The qualified graph differs from its reviewed contract.
    ContractMismatch,
    /// The contract document itself is invalid.
    InvalidContract,
}

/// Privacy-safe ONNX model-contract admission error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelContractError {
    code: ModelContractErrorCode,
}

impl ModelContractError {
    const fn new(code: ModelContractErrorCode) -> Self {
        Self { code }
    }

    /// Returns the stable failure category without model names, paths, or tensor values.
    #[must_use]
    pub const fn code(&self) -> ModelContractErrorCode {
        self.code
    }
}

impl fmt::Display for ModelContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.code {
            ModelContractErrorCode::ModelTooLarge => "ONNX model exceeds the admission limit",
            ModelContractErrorCode::MalformedModel => "ONNX model is malformed",
            ModelContractErrorCode::LimitExceeded => "ONNX model exceeds a structural limit",
            ModelContractErrorCode::UnsupportedModel => "ONNX model uses an unsupported feature",
            ModelContractErrorCode::ProvisionalContract => {
                "ONNX model contract has not completed qualification"
            }
            ModelContractErrorCode::ContractMismatch => {
                "ONNX model does not match its qualified contract"
            }
            ModelContractErrorCode::InvalidContract => "ONNX model contract is invalid",
        })
    }
}

impl Error for ModelContractError {}

/// Qualification state of a checked-in graph contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    /// Artifact provenance is known, but graph metadata has not been measured.
    Provisional,
    /// Every executable graph property has been measured and reviewed.
    Qualified,
}

/// Curated model role covered by a graph contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractRole {
    /// Text detector graph.
    Detector,
    /// English text recognizer graph.
    EnglishRecognizer,
}

/// One exact tensor-dimension constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DimensionContract {
    /// Fixed non-negative tensor dimension.
    Fixed(u64),
    /// Exact symbolic tensor dimension name.
    Symbol(String),
}

/// Exact graph input or output tensor contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorContract {
    /// Exact graph value name.
    pub name: String,
    /// Exact ONNX `TensorProto.DataType` number; absent only while provisional.
    pub element_type: Option<i32>,
    /// Exact ordered dimensions; absent only while provisional.
    pub dimensions: Option<Vec<DimensionContract>>,
}

/// Exact ONNX operator-set import.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpsetContract {
    /// Empty string denotes the standard ONNX domain.
    pub domain: String,
    /// Exact positive opset version.
    pub version: i64,
}

/// Exact occurrence count of an operator in the recursively admitted graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorContract {
    /// Empty string denotes the standard ONNX domain.
    pub domain: String,
    /// Exact operator type.
    pub op_type: String,
    /// Exact number of occurrences.
    pub count: u64,
}

/// Versioned, reviewable contract for one immutable ONNX artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelContract {
    /// Contract document schema version. Version 1 is currently supported.
    pub schema_version: u32,
    /// Human-stable contract identifier.
    pub contract_id: String,
    /// Curated model role.
    pub role: ContractRole,
    /// Whether real-model qualification has populated every graph property.
    pub qualification: QualificationStatus,
    /// Immutable lowercase SHA-256 of the serialized ONNX artifact.
    pub artifact_sha256: String,
    /// Exact byte length of the serialized ONNX artifact.
    pub artifact_byte_length: u64,
    /// Exact positive ONNX IR version; absent only while provisional.
    pub ir_version: Option<i64>,
    /// Exact imported operator sets.
    pub opsets: Vec<OpsetContract>,
    /// Exact ordered graph inputs.
    pub inputs: Vec<TensorContract>,
    /// Exact ordered graph outputs.
    pub outputs: Vec<TensorContract>,
    /// Exact recursive operator multiset.
    pub operators: Vec<OperatorContract>,
}

/// Parsed executable properties used for deterministic contract comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxGraphSummary {
    /// ONNX IR version.
    pub ir_version: i64,
    /// Imported operator sets in canonical order.
    pub opsets: Vec<OpsetContract>,
    /// Ordered graph inputs.
    pub inputs: Vec<TensorContract>,
    /// Ordered graph outputs.
    pub outputs: Vec<TensorContract>,
    /// Recursive operator multiset in canonical order.
    pub operators: Vec<OperatorContract>,
}

/// Parses a contract document and enforces all schema-level invariants.
///
/// # Errors
///
/// Returns a privacy-safe error when JSON is malformed, contains unknown fields, or violates the
/// qualified/provisional invariants.
pub fn parse_model_contract(json: &[u8]) -> Result<ModelContract, ModelContractError> {
    if json.is_empty() || json.len() > MAX_CONTRACT_BYTES {
        return Err(ModelContractError::new(
            ModelContractErrorCode::InvalidContract,
        ));
    }
    let contract: ModelContract = serde_json::from_slice(json)
        .map_err(|_| ModelContractError::new(ModelContractErrorCode::InvalidContract))?;
    validate_contract_document(&contract)?;
    Ok(contract)
}

/// Parses and recursively validates a bounded ONNX protobuf without loading a native runtime.
///
/// # Errors
///
/// Returns a stable, privacy-safe category for oversized, malformed, structurally excessive, or
/// deliberately unsupported graphs.
pub fn inspect_onnx_model(bytes: &[u8]) -> Result<OnnxGraphSummary, ModelContractError> {
    if bytes.is_empty() {
        return Err(ModelContractError::new(
            ModelContractErrorCode::MalformedModel,
        ));
    }
    if bytes.len() > MAX_ONNX_MODEL_BYTES {
        return Err(ModelContractError::new(
            ModelContractErrorCode::ModelTooLarge,
        ));
    }
    let mut preflight = PreflightBudget::default();
    scan_message(bytes, MessageSchema::Model, 0, &mut preflight)?;
    let model = ModelProto::decode(bytes)
        .map_err(|_| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
    summarize_model(&model)
}

/// Admits an ONNX graph only when its bytes and every executable property match a qualified
/// contract exactly.
///
/// # Errors
///
/// Returns a stable, privacy-safe category when the contract is provisional or invalid, the
/// artifact length or digest differs, or graph parsing/contract comparison fails.
pub fn admit_onnx_model(
    bytes: &[u8],
    contract: &ModelContract,
) -> Result<OnnxGraphSummary, ModelContractError> {
    validate_contract_document(contract)?;
    if contract.qualification != QualificationStatus::Qualified {
        return Err(ModelContractError::new(
            ModelContractErrorCode::ProvisionalContract,
        ));
    }
    if u64::try_from(bytes.len()).ok() != Some(contract.artifact_byte_length) {
        return Err(ModelContractError::new(
            ModelContractErrorCode::ContractMismatch,
        ));
    }
    let actual_digest = format!("{:x}", Sha256::digest(bytes));
    if actual_digest != contract.artifact_sha256 {
        return Err(ModelContractError::new(
            ModelContractErrorCode::ContractMismatch,
        ));
    }
    let summary = inspect_onnx_model(bytes)?;
    if contract.ir_version != Some(summary.ir_version)
        || contract.opsets != summary.opsets
        || contract.inputs != summary.inputs
        || contract.outputs != summary.outputs
        || contract.operators != summary.operators
    {
        return Err(ModelContractError::new(
            ModelContractErrorCode::ContractMismatch,
        ));
    }
    Ok(summary)
}

/// Returns the checked-in qualified detector contract.
///
/// # Errors
///
/// Returns an invalid-contract error if the embedded review artifact is corrupted.
pub fn detector_contract() -> Result<ModelContract, ModelContractError> {
    parse_model_contract(include_bytes!("../contracts/pp-ocrv5-mobile-detector.json"))
}

/// Returns the checked-in qualified English recognizer contract.
///
/// # Errors
///
/// Returns an invalid-contract error if the embedded review artifact is corrupted.
pub fn english_recognizer_contract() -> Result<ModelContract, ModelContractError> {
    parse_model_contract(include_bytes!(
        "../contracts/pp-ocrv5-english-mobile-recognizer.json"
    ))
}

fn validate_contract_document(contract: &ModelContract) -> Result<(), ModelContractError> {
    let invalid = || ModelContractError::new(ModelContractErrorCode::InvalidContract);
    if contract.schema_version != 1
        || contract.contract_id.is_empty()
        || contract.contract_id.len() > MAX_TEXT_BYTES
        || !is_lower_hex_sha256(&contract.artifact_sha256)
        || contract.artifact_byte_length == 0
        || contract.artifact_byte_length > MAX_ONNX_MODEL_BYTES as u64
        || contract.opsets.len() > MAX_OPSETS
        || contract.inputs.len() > MAX_GRAPH_IO
        || contract.outputs.len() > MAX_GRAPH_IO
        || contract.operators.len() > MAX_GRAPH_NODES
        || has_duplicate_opsets(&contract.opsets)
        || has_duplicate_operators(&contract.operators)
        || has_duplicate_tensor_names(&contract.inputs)
        || has_duplicate_tensor_names(&contract.outputs)
        || !contract.opsets.windows(2).all(|pair| pair[0] < pair[1])
        || !contract.operators.windows(2).all(|pair| pair[0] < pair[1])
    {
        return Err(invalid());
    }
    for tensor in contract.inputs.iter().chain(&contract.outputs) {
        if tensor.name.is_empty()
            || tensor.name.len() > MAX_TEXT_BYTES
            || tensor.element_type.is_some_and(|value| value <= 0)
            || tensor.dimensions.as_ref().is_some_and(|dimensions| {
                dimensions.len() > MAX_TENSOR_RANK
                    || dimensions.iter().any(|dimension| {
                        matches!(dimension, DimensionContract::Fixed(value) if *value > i64::MAX as u64)
                            || matches!(dimension, DimensionContract::Symbol(value) if value.is_empty() || value.len() > MAX_TEXT_BYTES)
                    })
            })
        {
            return Err(invalid());
        }
    }
    if contract
        .opsets
        .iter()
        .any(|opset| !is_standard_domain(&opset.domain) || opset.version <= 0)
        || contract.operators.iter().any(|operator| {
            !is_standard_domain(&operator.domain)
                || operator.op_type.is_empty()
                || operator.op_type.len() > MAX_TEXT_BYTES
                || operator.count == 0
                || operator.count > MAX_GRAPH_NODES as u64
        })
    {
        return Err(invalid());
    }
    if contract.qualification == QualificationStatus::Qualified
        && (contract.ir_version.is_none_or(|version| version <= 0)
            || contract.opsets.is_empty()
            || contract.inputs.is_empty()
            || contract.outputs.is_empty()
            || contract.operators.is_empty()
            || contract
                .inputs
                .iter()
                .chain(&contract.outputs)
                .any(|tensor| tensor.element_type.is_none() || tensor.dimensions.is_none()))
    {
        return Err(invalid());
    }
    Ok(())
}

fn has_duplicate_opsets(values: &[OpsetContract]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(&value.domain))
}

fn has_duplicate_operators(values: &[OperatorContract]) -> bool {
    let mut seen = BTreeSet::new();
    values
        .iter()
        .any(|value| !seen.insert((&value.domain, &value.op_type)))
}

fn has_duplicate_tensor_names(values: &[TensorContract]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(&value.name))
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

fn summarize_model(model: &ModelProto) -> Result<OnnxGraphSummary, ModelContractError> {
    if model.ir_version <= 0 || model.opset_import.is_empty() || model.graph.is_none() {
        return malformed();
    }
    if !model.training_info.is_empty() || !model.functions.is_empty() {
        return unsupported();
    }
    let mut opsets = Vec::with_capacity(model.opset_import.len());
    let mut domains = BTreeSet::new();
    for opset in &model.opset_import {
        if !is_standard_domain(&opset.domain)
            || opset.version <= 0
            || !domains.insert(opset.domain.clone())
        {
            return unsupported();
        }
        opsets.push(OpsetContract {
            domain: opset.domain.clone(),
            version: opset.version,
        });
    }
    opsets.sort();
    let mut budget = GraphBudget::default();
    let mut operators = BTreeMap::<(String, String), u64>::new();
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
    validate_graph(graph, 0, &mut budget, &mut operators)?;
    let inputs = tensor_contracts(&graph.input)?;
    let outputs = tensor_contracts(&graph.output)?;
    Ok(OnnxGraphSummary {
        ir_version: model.ir_version,
        opsets,
        inputs,
        outputs,
        operators: operators
            .into_iter()
            .map(|((domain, op_type), count)| OperatorContract {
                domain,
                op_type,
                count,
            })
            .collect(),
    })
}

#[derive(Default)]
struct GraphBudget {
    nodes: usize,
    attributes: usize,
    tensors: usize,
    values: usize,
}

fn validate_graph(
    graph: &GraphProto,
    depth: usize,
    budget: &mut GraphBudget,
    operators: &mut BTreeMap<(String, String), u64>,
) -> Result<(), ModelContractError> {
    if depth > MAX_PROTO_DEPTH || graph.input.is_empty() || graph.output.is_empty() {
        return malformed();
    }
    budget.nodes = budget
        .nodes
        .checked_add(graph.node.len())
        .ok_or_else(limit)?;
    budget.tensors = budget
        .tensors
        .checked_add(graph.initializer.len() + graph.sparse_initializer.len())
        .ok_or_else(limit)?;
    budget.values = budget
        .values
        .checked_add(graph.input.len() + graph.output.len() + graph.value_info.len())
        .ok_or_else(limit)?;
    if budget.nodes > MAX_GRAPH_NODES
        || budget.tensors > MAX_GRAPH_TENSORS
        || budget.values > MAX_GRAPH_VALUES
    {
        return Err(limit());
    }
    validate_value_names(&graph.input)?;
    validate_value_names(&graph.output)?;
    validate_value_names(&graph.value_info)?;

    let mut initializer_names = BTreeSet::new();
    for tensor in &graph.initializer {
        validate_tensor(tensor)?;
        if tensor.name.is_empty() || !initializer_names.insert(tensor.name.clone()) {
            return malformed();
        }
    }
    for sparse in &graph.sparse_initializer {
        validate_sparse_tensor(sparse)?;
        let name = sparse
            .values
            .as_ref()
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();
        if name.is_empty() || !initializer_names.insert(name) {
            return malformed();
        }
    }

    let mut produced = initializer_names;
    for node in &graph.node {
        if node.op_type.is_empty() || !is_standard_domain(&node.domain) || !node.overload.is_empty()
        {
            return unsupported();
        }
        let count = operators
            .entry((node.domain.clone(), node.op_type.clone()))
            .or_default();
        *count = count.checked_add(1).ok_or_else(limit)?;
        budget.attributes = budget
            .attributes
            .checked_add(node.attribute.len())
            .ok_or_else(limit)?;
        if budget.attributes > MAX_GRAPH_ATTRIBUTES {
            return Err(limit());
        }
        let mut attribute_names = BTreeSet::new();
        for attribute in &node.attribute {
            if attribute.name.is_empty()
                || !attribute_names.insert(&attribute.name)
                || !attribute.ref_attr_name.is_empty()
            {
                return malformed();
            }
            validate_attribute(attribute, depth + 1, budget, operators)?;
        }
        for output in &node.output {
            if !output.is_empty() && !produced.insert(output.clone()) {
                return malformed();
            }
        }
    }
    Ok(())
}

fn validate_attribute(
    attribute: &AttributeProto,
    depth: usize,
    budget: &mut GraphBudget,
    operators: &mut BTreeMap<(String, String), u64>,
) -> Result<(), ModelContractError> {
    let tensor_count = usize::from(attribute.tensor.is_some())
        .checked_add(attribute.tensors.len())
        .and_then(|count| count.checked_add(usize::from(attribute.sparse_tensor.is_some())))
        .and_then(|count| count.checked_add(attribute.sparse_tensors.len()))
        .ok_or_else(limit)?;
    budget.tensors = budget.tensors.checked_add(tensor_count).ok_or_else(limit)?;
    if budget.tensors > MAX_GRAPH_TENSORS {
        return Err(limit());
    }
    if let Some(tensor) = &attribute.tensor {
        validate_tensor(tensor)?;
    }
    for tensor in &attribute.tensors {
        validate_tensor(tensor)?;
    }
    if let Some(sparse) = &attribute.sparse_tensor {
        validate_sparse_tensor(sparse)?;
    }
    for sparse in &attribute.sparse_tensors {
        validate_sparse_tensor(sparse)?;
    }
    if let Some(graph) = &attribute.graph {
        validate_graph(graph, depth, budget, operators)?;
    }
    for graph in &attribute.graphs {
        validate_graph(graph, depth, budget, operators)?;
    }
    Ok(())
}

fn validate_tensor(tensor: &TensorProto) -> Result<(), ModelContractError> {
    if tensor.data_location != 0 || !tensor.external_data.is_empty() {
        return unsupported();
    }
    Ok(())
}

fn validate_sparse_tensor(tensor: &SparseTensorProto) -> Result<(), ModelContractError> {
    let values = tensor
        .values
        .as_ref()
        .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
    let indices = tensor
        .indices
        .as_ref()
        .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
    validate_tensor(values)?;
    validate_tensor(indices)
}

fn validate_value_names(values: &[ValueInfoProto]) -> Result<(), ModelContractError> {
    let mut names = BTreeSet::new();
    for value in values {
        if value.name.is_empty() || !names.insert(&value.name) {
            return malformed();
        }
    }
    Ok(())
}

fn tensor_contracts(values: &[ValueInfoProto]) -> Result<Vec<TensorContract>, ModelContractError> {
    values
        .iter()
        .map(|value| {
            let tensor = value
                .r#type
                .as_ref()
                .and_then(|value_type| value_type.value.as_ref())
                .and_then(|value| match value {
                    type_proto::Value::Tensor(tensor) => Some(tensor),
                    type_proto::Value::Sequence(_)
                    | type_proto::Value::Map(_)
                    | type_proto::Value::SparseTensor(_)
                    | type_proto::Value::Optional(_) => None,
                })
                .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::UnsupportedModel))?;
            if tensor.element_type <= 0 {
                return malformed();
            }
            let shape = tensor
                .shape
                .as_ref()
                .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
            if shape.dim.len() > MAX_TENSOR_RANK {
                return Err(limit());
            }
            let dimensions = shape
                .dim
                .iter()
                .map(|dimension| match dimension.value.as_ref() {
                    Some(tensor_shape_dimension::Value::Fixed(value)) if *value >= 0 => {
                        u64::try_from(*value)
                            .map(DimensionContract::Fixed)
                            .map_err(|_| limit())
                    }
                    Some(tensor_shape_dimension::Value::Symbol(value)) if !value.is_empty() => {
                        Ok(DimensionContract::Symbol(value.clone()))
                    }
                    _ => malformed(),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorContract {
                name: value.name.clone(),
                element_type: Some(tensor.element_type),
                dimensions: Some(dimensions),
            })
        })
        .collect()
}

fn malformed<T>() -> Result<T, ModelContractError> {
    Err(ModelContractError::new(
        ModelContractErrorCode::MalformedModel,
    ))
}

fn unsupported<T>() -> Result<T, ModelContractError> {
    Err(ModelContractError::new(
        ModelContractErrorCode::UnsupportedModel,
    ))
}

const fn limit() -> ModelContractError {
    ModelContractError::new(ModelContractErrorCode::LimitExceeded)
}

#[derive(Default)]
struct PreflightBudget {
    fields: usize,
}

#[derive(Clone, Copy)]
enum MessageSchema {
    Model,
    Opset,
    Graph,
    Node,
    Attribute,
    Tensor,
    TensorSegment,
    SparseTensor,
    ValueInfo,
    Type,
    TensorType,
    SequenceType,
    MapType,
    OptionalType,
    Shape,
    Dimension,
    KeyValue,
    TensorAnnotation,
}

#[derive(Clone, Copy)]
enum FieldPayload {
    Scalar,
    Bytes(usize),
    Message(MessageSchema),
    Forbidden,
}

fn scan_message(
    bytes: &[u8],
    schema: MessageSchema,
    depth: usize,
    budget: &mut PreflightBudget,
) -> Result<(), ModelContractError> {
    if depth > MAX_PROTO_DEPTH {
        return Err(limit());
    }
    let mut position = 0_usize;
    while position < bytes.len() {
        budget.fields = budget.fields.checked_add(1).ok_or_else(limit)?;
        if budget.fields > MAX_PROTO_FIELDS {
            return Err(limit());
        }
        let key = read_varint(bytes, &mut position)?;
        let tag = u32::try_from(key >> 3)
            .map_err(|_| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
        let wire = u8::try_from(key & 0x07)
            .map_err(|_| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
        if tag == 0 {
            return malformed();
        }
        let payload = field_payload(schema, tag, wire)
            .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::UnsupportedModel))?;
        match payload {
            FieldPayload::Scalar => skip_scalar(bytes, &mut position, wire)?,
            FieldPayload::Bytes(maximum) => {
                let field = read_length_delimited(bytes, &mut position)?;
                if field.len() > maximum {
                    return Err(limit());
                }
            }
            FieldPayload::Message(child) => {
                let field = read_length_delimited(bytes, &mut position)?;
                scan_message(field, child, depth + 1, budget)?;
            }
            FieldPayload::Forbidden => return unsupported(),
        }
    }
    Ok(())
}

#[allow(
    clippy::match_same_arms,
    clippy::too_many_lines,
    clippy::unnested_or_patterns
)]
fn field_payload(schema: MessageSchema, tag: u32, wire: u8) -> Option<FieldPayload> {
    use FieldPayload::{Bytes, Forbidden, Message, Scalar};
    use MessageSchema as S;
    let text = || Bytes(MAX_TEXT_BYTES);
    let packed = || Bytes(MAX_ONNX_MODEL_BYTES);
    match (schema, tag, wire) {
        (S::Model, 1 | 5, 0) => Some(Scalar),
        (S::Model, 2 | 3 | 4 | 6, 2) => Some(text()),
        (S::Model, 7, 2) => Some(Message(S::Graph)),
        (S::Model, 8, 2) => Some(Message(S::Opset)),
        (S::Model, 14, 2) => Some(Message(S::KeyValue)),
        (S::Model, 20 | 25, 2) => Some(Forbidden),

        (S::Opset, 1, 2) => Some(text()),
        (S::Opset, 2, 0) => Some(Scalar),

        (S::Graph, 1, 2) => Some(Message(S::Node)),
        (S::Graph, 2 | 10, 2) => Some(text()),
        (S::Graph, 5, 2) => Some(Message(S::Tensor)),
        (S::Graph, 11..=13, 2) => Some(Message(S::ValueInfo)),
        (S::Graph, 14, 2) => Some(Message(S::TensorAnnotation)),
        (S::Graph, 15, 2) => Some(Message(S::SparseTensor)),
        (S::Graph, 16, 2) => Some(Message(S::KeyValue)),

        (S::Node, 1..=4 | 6..=8, 2) => Some(text()),
        (S::Node, 5, 2) => Some(Message(S::Attribute)),
        (S::Node, 9, 2) => Some(Message(S::KeyValue)),

        (S::Attribute, 1 | 4 | 9 | 13 | 21, 2) => Some(text()),
        (S::Attribute, 2, 5) | (S::Attribute, 3 | 20, 0) => Some(Scalar),
        (S::Attribute, 5 | 10, 2) => Some(Message(S::Tensor)),
        (S::Attribute, 6 | 11, 2) => Some(Message(S::Graph)),
        (S::Attribute, 7, 2) | (S::Attribute, 8, 2) => Some(packed()),
        (S::Attribute, 7, 5) | (S::Attribute, 8, 0) => Some(Scalar),
        (S::Attribute, 14 | 15, 2) => Some(Message(S::Type)),
        (S::Attribute, 22 | 23, 2) => Some(Message(S::SparseTensor)),

        (S::Tensor, 1, 0) | (S::Tensor, 2 | 14, 0) => Some(Scalar),
        (S::Tensor, 1 | 4 | 5 | 7 | 10 | 11, 2) => Some(packed()),
        (S::Tensor, 3, 2) => Some(Message(S::TensorSegment)),
        (S::Tensor, 4, 5) | (S::Tensor, 5 | 7 | 11, 0) | (S::Tensor, 10, 1) => Some(Scalar),
        (S::Tensor, 6 | 9, 2) => Some(packed()),
        (S::Tensor, 8 | 12, 2) => Some(text()),
        (S::Tensor, 13 | 16, 2) => Some(Message(S::KeyValue)),

        (S::TensorSegment, 1 | 2, 0) => Some(Scalar),
        (S::SparseTensor, 1 | 2, 2) => Some(Message(S::Tensor)),
        (S::SparseTensor, 3, 0) => Some(Scalar),
        (S::SparseTensor, 3, 2) => Some(packed()),

        (S::ValueInfo, 1 | 3, 2) => Some(text()),
        (S::ValueInfo, 2, 2) => Some(Message(S::Type)),
        (S::ValueInfo, 4, 2) => Some(Message(S::KeyValue)),

        (S::Type, 1, 2) => Some(Message(S::TensorType)),
        (S::Type, 4, 2) => Some(Message(S::SequenceType)),
        (S::Type, 5, 2) => Some(Message(S::MapType)),
        (S::Type, 6, 2) => Some(text()),
        (S::Type, 8, 2) => Some(Message(S::TensorType)),
        (S::Type, 9, 2) => Some(Message(S::OptionalType)),
        (S::TensorType, 1, 0) => Some(Scalar),
        (S::TensorType, 2, 2) => Some(Message(S::Shape)),
        (S::SequenceType | S::OptionalType, 1, 2) => Some(Message(S::Type)),
        (S::MapType, 1, 0) => Some(Scalar),
        (S::MapType, 2, 2) => Some(Message(S::Type)),
        (S::Shape, 1, 2) => Some(Message(S::Dimension)),
        (S::Dimension, 1, 0) => Some(Scalar),
        (S::Dimension, 2 | 3, 2) => Some(text()),
        (S::KeyValue, 1 | 2, 2) => Some(text()),
        (S::TensorAnnotation, 1, 2) => Some(text()),
        (S::TensorAnnotation, 2, 2) => Some(Message(S::KeyValue)),
        _ => None,
    }
}

fn read_varint(bytes: &[u8], position: &mut usize) -> Result<u64, ModelContractError> {
    let mut value = 0_u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*position)
            .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
        *position = position.checked_add(1).ok_or_else(limit)?;
        if shift == 63 && byte > 1 {
            return malformed();
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    malformed()
}

fn read_length_delimited<'a>(
    bytes: &'a [u8],
    position: &mut usize,
) -> Result<&'a [u8], ModelContractError> {
    let length = usize::try_from(read_varint(bytes, position)?).map_err(|_| limit())?;
    let end = position.checked_add(length).ok_or_else(limit)?;
    let field = bytes
        .get(*position..end)
        .ok_or_else(|| ModelContractError::new(ModelContractErrorCode::MalformedModel))?;
    *position = end;
    Ok(field)
}

fn skip_scalar(bytes: &[u8], position: &mut usize, wire: u8) -> Result<(), ModelContractError> {
    match wire {
        0 => {
            let _ = read_varint(bytes, position)?;
        }
        1 => {
            *position = position.checked_add(8).ok_or_else(limit)?;
        }
        5 => {
            *position = position.checked_add(4).ok_or_else(limit)?;
        }
        _ => return malformed(),
    }
    if *position > bytes.len() {
        return malformed();
    }
    Ok(())
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(bytes = "vec", repeated, tag = "20")]
    training_info: Vec<Vec<u8>>,
    #[prost(bytes = "vec", repeated, tag = "25")]
    functions: Vec<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "15")]
    sparse_initializer: Vec<SparseTensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    value_info: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    domain: String,
    #[prost(string, tag = "8")]
    overload: String,
}

#[derive(Clone, PartialEq, Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "5")]
    tensor: Option<TensorProto>,
    #[prost(message, optional, boxed, tag = "6")]
    graph: Option<Box<GraphProto>>,
    #[prost(message, repeated, tag = "10")]
    tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    graphs: Vec<GraphProto>,
    #[prost(string, tag = "21")]
    ref_attr_name: String,
    #[prost(message, optional, tag = "22")]
    sparse_tensor: Option<SparseTensorProto>,
    #[prost(message, repeated, tag = "23")]
    sparse_tensors: Vec<SparseTensorProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(string, tag = "8")]
    name: String,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    data_location: i32,
}

#[derive(Clone, PartialEq, Message)]
struct SparseTensorProto {
    #[prost(message, optional, tag = "1")]
    values: Option<TensorProto>,
    #[prost(message, optional, tag = "2")]
    indices: Option<TensorProto>,
}

#[derive(Clone, PartialEq, Message)]
struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(oneof = "type_proto::Value", tags = "1, 4, 5, 8, 9")]
    value: Option<type_proto::Value>,
}

mod type_proto {
    use prost::Oneof;

    use super::TensorTypeProto;

    #[derive(Clone, PartialEq, Oneof)]
    pub(super) enum Value {
        #[prost(message, tag = "1")]
        Tensor(TensorTypeProto),
        #[prost(bytes, tag = "4")]
        Sequence(Vec<u8>),
        #[prost(bytes, tag = "5")]
        Map(Vec<u8>),
        #[prost(bytes, tag = "8")]
        SparseTensor(Vec<u8>),
        #[prost(bytes, tag = "9")]
        Optional(Vec<u8>),
    }
}

#[derive(Clone, PartialEq, Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    element_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<TensorShapeDimension>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeDimension {
    #[prost(oneof = "tensor_shape_dimension::Value", tags = "1, 2")]
    value: Option<tensor_shape_dimension::Value>,
}

mod tensor_shape_dimension {
    use prost::Oneof;

    #[derive(Clone, PartialEq, Oneof)]
    pub(super) enum Value {
        #[prost(int64, tag = "1")]
        Fixed(i64),
        #[prost(string, tag = "2")]
        Symbol(String),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_value(name: &str, dimensions: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::Tensor(TensorTypeProto {
                    element_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dimensions
                            .iter()
                            .copied()
                            .map(|value| TensorShapeDimension {
                                value: Some(tensor_shape_dimension::Value::Fixed(value)),
                            })
                            .collect(),
                    }),
                })),
            }),
        }
    }

    fn model() -> ModelProto {
        ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    input: vec!["x".to_owned()],
                    output: vec!["y".to_owned()],
                    op_type: "Identity".to_owned(),
                    attribute: Vec::new(),
                    domain: String::new(),
                    overload: String::new(),
                }],
                initializer: Vec::new(),
                sparse_initializer: Vec::new(),
                input: vec![tensor_value("x", &[1, 3, 48, 320])],
                output: vec![tensor_value("y", &[1, 10, 438])],
                value_info: Vec::new(),
            }),
            training_info: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn qualified_contract(bytes: &[u8]) -> ModelContract {
        ModelContract {
            schema_version: 1,
            contract_id: "synthetic-qualified-fixture".to_owned(),
            role: ContractRole::EnglishRecognizer,
            qualification: QualificationStatus::Qualified,
            artifact_sha256: format!("{:x}", Sha256::digest(bytes)),
            artifact_byte_length: bytes.len() as u64,
            ir_version: Some(9),
            opsets: vec![OpsetContract {
                domain: String::new(),
                version: 17,
            }],
            inputs: vec![TensorContract {
                name: "x".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Fixed(1),
                    DimensionContract::Fixed(3),
                    DimensionContract::Fixed(48),
                    DimensionContract::Fixed(320),
                ]),
            }],
            outputs: vec![TensorContract {
                name: "y".to_owned(),
                element_type: Some(1),
                dimensions: Some(vec![
                    DimensionContract::Fixed(1),
                    DimensionContract::Fixed(10),
                    DimensionContract::Fixed(438),
                ]),
            }],
            operators: vec![OperatorContract {
                domain: String::new(),
                op_type: "Identity".to_owned(),
                count: 1,
            }],
        }
    }

    #[test]
    fn synthetic_graph_is_admitted_only_by_an_exact_qualified_contract()
    -> Result<(), Box<dyn Error>> {
        let bytes = model().encode_to_vec();
        let contract = qualified_contract(&bytes);
        let summary = admit_onnx_model(&bytes, &contract)?;
        assert_eq!(summary.ir_version, 9);

        let mut wrong = contract.clone();
        wrong.operators[0].count = 2;
        assert_eq!(
            admit_onnx_model(&bytes, &wrong).map_err(|error| error.code()),
            Err(ModelContractErrorCode::ContractMismatch)
        );
        let mut wrong = contract.clone();
        wrong.artifact_byte_length += 1;
        assert_eq!(
            admit_onnx_model(&bytes, &wrong).map_err(|error| error.code()),
            Err(ModelContractErrorCode::ContractMismatch)
        );
        let mut wrong = contract;
        wrong.artifact_sha256 = "0".repeat(64);
        assert_eq!(
            admit_onnx_model(&bytes, &wrong).map_err(|error| error.code()),
            Err(ModelContractErrorCode::ContractMismatch)
        );
        Ok(())
    }

    #[test]
    fn checked_in_contracts_are_complete_and_qualified() -> Result<(), Box<dyn Error>> {
        for contract in [detector_contract()?, english_recognizer_contract()?] {
            assert_eq!(contract.qualification, QualificationStatus::Qualified);
            assert!(contract.ir_version.is_some());
            assert!(!contract.opsets.is_empty());
            assert!(!contract.inputs.is_empty());
            assert!(!contract.outputs.is_empty());
            assert!(!contract.operators.is_empty());
            assert_eq!(
                admit_onnx_model(&model().encode_to_vec(), &contract).map_err(|error| error.code()),
                Err(ModelContractErrorCode::ContractMismatch)
            );
        }
        Ok(())
    }

    #[test]
    fn external_data_custom_domains_functions_and_training_are_rejected()
    -> Result<(), Box<dyn Error>> {
        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.initializer = vec![TensorProto {
            name: "weights".to_owned(),
            external_data: vec![StringStringEntryProto {
                key: "location".to_owned(),
                value: "outside".to_owned(),
            }],
            data_location: 1,
        }];
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.node[0].domain = "vendor.custom".to_owned();
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        let mut mutant = model();
        mutant.functions.push(vec![0]);
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        let mut mutant = model();
        mutant.training_info.push(vec![0]);
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);
        Ok(())
    }

    #[test]
    fn nested_external_data_and_custom_domain_are_rejected_recursively()
    -> Result<(), Box<dyn Error>> {
        let mut nested = model().graph.ok_or("missing graph")?;
        nested.initializer.push(TensorProto {
            name: "nested-weights".to_owned(),
            external_data: Vec::new(),
            data_location: 1,
        });
        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.node[0].attribute = vec![AttributeProto {
            name: "body".to_owned(),
            tensor: None,
            graph: Some(Box::new(nested)),
            tensors: Vec::new(),
            graphs: Vec::new(),
            ref_attr_name: String::new(),
            sparse_tensor: None,
            sparse_tensors: Vec::new(),
        }];
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        let mut nested = model().graph.ok_or("missing graph")?;
        nested.node[0].domain = "vendor.custom".to_owned();
        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.node[0].attribute = vec![AttributeProto {
            name: "body".to_owned(),
            tensor: None,
            graph: Some(Box::new(nested)),
            tensors: Vec::new(),
            graphs: Vec::new(),
            ref_attr_name: String::new(),
            sparse_tensor: None,
            sparse_tensors: Vec::new(),
        }];
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);
        Ok(())
    }

    #[test]
    fn malformed_duplicate_io_and_wire_mutants_fail_closed() -> Result<(), Box<dyn Error>> {
        let mut mutant = model();
        mutant
            .graph
            .as_mut()
            .ok_or("missing graph")?
            .input
            .push(tensor_value("x", &[1]));
        assert_rejected(&mutant, ModelContractErrorCode::MalformedModel);

        let mut mutant = model();
        mutant
            .graph
            .as_mut()
            .ok_or("missing graph")?
            .output
            .push(tensor_value("y", &[1]));
        assert_rejected(&mutant, ModelContractErrorCode::MalformedModel);

        let mut mutant = model();
        mutant
            .graph
            .as_mut()
            .ok_or("missing graph")?
            .node
            .push(NodeProto {
                input: vec!["x".to_owned()],
                output: vec!["y".to_owned()],
                op_type: "Identity".to_owned(),
                attribute: Vec::new(),
                domain: String::new(),
                overload: String::new(),
            });
        assert_rejected(&mutant, ModelContractErrorCode::MalformedModel);

        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.output[0].r#type = None;
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        assert_eq!(
            inspect_onnx_model(&[0x3a, 0x02, 0x08]).map_err(|error| error.code()),
            Err(ModelContractErrorCode::MalformedModel)
        );
        assert_eq!(
            inspect_onnx_model(&[0xd8, 0x07, 0x00]).map_err(|error| error.code()),
            Err(ModelContractErrorCode::UnsupportedModel)
        );
        assert_eq!(
            inspect_onnx_model(&vec![0; MAX_ONNX_MODEL_BYTES + 1]).map_err(|error| error.code()),
            Err(ModelContractErrorCode::ModelTooLarge)
        );
        Ok(())
    }

    #[test]
    fn protobuf_preflight_enforces_field_string_and_recursion_budgets() -> Result<(), Box<dyn Error>>
    {
        let repeated_scalar = [0x28_u8, 0x00].repeat(MAX_PROTO_FIELDS + 1);
        assert_eq!(
            inspect_onnx_model(&repeated_scalar).map_err(|error| error.code()),
            Err(ModelContractErrorCode::LimitExceeded)
        );

        let long_text = vec![b'x'; MAX_TEXT_BYTES + 1];
        let mut oversized_string = vec![0x12];
        prost::encoding::encode_varint(
            u64::try_from(long_text.len()).map_err(|_| "length conversion")?,
            &mut oversized_string,
        );
        oversized_string.extend_from_slice(&long_text);
        assert_eq!(
            inspect_onnx_model(&oversized_string).map_err(|error| error.code()),
            Err(ModelContractErrorCode::LimitExceeded)
        );

        let mut nested = model().graph.ok_or("missing graph")?;
        for _ in 0..=MAX_PROTO_DEPTH {
            let parent = model().graph.ok_or("missing graph")?;
            let mut parent = parent;
            parent.node[0].attribute = vec![AttributeProto {
                name: "body".to_owned(),
                tensor: None,
                graph: Some(Box::new(nested)),
                tensors: Vec::new(),
                graphs: Vec::new(),
                ref_attr_name: String::new(),
                sparse_tensor: None,
                sparse_tensors: Vec::new(),
            }];
            nested = parent;
        }
        let mut deeply_nested = model();
        deeply_nested.graph = Some(nested);
        assert_eq!(
            inspect_onnx_model(&deeply_nested.encode_to_vec()).map_err(|error| error.code()),
            Err(ModelContractErrorCode::LimitExceeded)
        );
        Ok(())
    }

    #[test]
    fn custom_opset_and_unknown_proto_fields_fail_closed() -> Result<(), Box<dyn Error>> {
        let mut mutant = model();
        mutant.opset_import[0].domain = "vendor.custom".to_owned();
        assert_rejected(&mutant, ModelContractErrorCode::UnsupportedModel);

        assert_eq!(
            inspect_onnx_model(&[0xd8, 0x07, 0x00]).map_err(|error| error.code()),
            Err(ModelContractErrorCode::UnsupportedModel)
        );

        let mut mutant = model();
        mutant.graph.as_mut().ok_or("missing graph")?.node[0].domain =
            "private-model-name".to_owned();
        let error = inspect_onnx_model(&mutant.encode_to_vec())
            .err()
            .ok_or("private-domain mutant was accepted")?;
        let rendered = format!("{error:?} {error}");
        assert!(!rendered.contains("private-model-name"));
        Ok(())
    }

    #[test]
    fn contract_parser_rejects_unknown_fields_duplicates_and_fake_qualification() {
        let unknown = br#"{"schema_version":1,"unknown":true}"#;
        assert_eq!(
            parse_model_contract(unknown).map_err(|error| error.code()),
            Err(ModelContractErrorCode::InvalidContract)
        );

        let mut contract = qualified_contract(&model().encode_to_vec());
        contract.inputs.push(contract.inputs[0].clone());
        assert_eq!(
            validate_contract_document(&contract).map_err(|error| error.code()),
            Err(ModelContractErrorCode::InvalidContract)
        );

        let mut contract = qualified_contract(&model().encode_to_vec());
        contract.outputs[0].dimensions = None;
        assert_eq!(
            validate_contract_document(&contract).map_err(|error| error.code()),
            Err(ModelContractErrorCode::InvalidContract)
        );

        assert_eq!(
            parse_model_contract(&vec![b' '; MAX_CONTRACT_BYTES + 1]).map_err(|error| error.code()),
            Err(ModelContractErrorCode::InvalidContract)
        );
    }

    fn assert_rejected(model: &ModelProto, expected: ModelContractErrorCode) {
        assert_eq!(
            inspect_onnx_model(&model.encode_to_vec()).map_err(|error| error.code()),
            Err(expected)
        );
    }
}
