//! Validated, transport-neutral contracts for raster OCR.

use std::{cmp::Ordering, error::Error, fmt, sync::Arc};

use serde::{Deserialize, Deserializer, Serialize};

/// Maximum encoded image size accepted by the product contract: 32 MiB.
pub const MAX_ENCODED_BYTES: u64 = 32 * 1024 * 1024;
/// Maximum width or height accepted by the product contract.
pub const MAX_DIMENSION: u32 = 16_384;
/// Maximum decoded pixel count accepted by the product contract.
pub const MAX_PIXELS: u64 = 64 * 1024 * 1024;
/// Maximum decoded RGB byte length accepted by the product contract: 192 MiB.
pub const MAX_DECODED_BYTES: u64 = MAX_PIXELS * 3;
/// Vertical top-left tolerance used by the canonical Paddle-compatible reading order.
pub const READING_ORDER_ROW_TOLERANCE: f32 = 10.0;

/// Validated per-decoder resource limits bounded by the public product maxima.
#[allow(clippy::struct_field_names)] // The shared prefix makes every private bound unambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterLimits {
    max_encoded_bytes: u64,
    max_width: u32,
    max_height: u32,
    max_pixels: u64,
    max_decoded_bytes: u64,
}

impl RasterLimits {
    /// Constructs a limit set that cannot exceed the public product maxima.
    ///
    /// # Errors
    /// Returns `invalid_request` if any bound is zero or exceeds its product maximum.
    pub fn new(
        max_encoded_bytes: u64,
        max_width: u32,
        max_height: u32,
        max_pixels: u64,
        max_decoded_bytes: u64,
    ) -> Result<Self, OcrError> {
        if max_encoded_bytes == 0
            || max_encoded_bytes > MAX_ENCODED_BYTES
            || max_width == 0
            || max_width > MAX_DIMENSION
            || max_height == 0
            || max_height > MAX_DIMENSION
            || max_pixels == 0
            || max_pixels > MAX_PIXELS
            || max_decoded_bytes == 0
            || max_decoded_bytes > MAX_DECODED_BYTES
        {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
        Ok(Self {
            max_encoded_bytes,
            max_width,
            max_height,
            max_pixels,
            max_decoded_bytes,
        })
    }

    /// Maximum encoded bytes read or retained.
    #[must_use]
    pub const fn max_encoded_bytes(self) -> u64 {
        self.max_encoded_bytes
    }

    /// Maximum decoded width.
    #[must_use]
    pub const fn max_width(self) -> u32 {
        self.max_width
    }

    /// Maximum decoded height.
    #[must_use]
    pub const fn max_height(self) -> u32 {
        self.max_height
    }

    /// Maximum decoded pixel count.
    #[must_use]
    pub const fn max_pixels(self) -> u64 {
        self.max_pixels
    }

    /// Maximum RGB output byte length.
    #[must_use]
    pub const fn max_decoded_bytes(self) -> u64 {
        self.max_decoded_bytes
    }
}

impl Default for RasterLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: MAX_ENCODED_BYTES,
            max_width: MAX_DIMENSION,
            max_height: MAX_DIMENSION,
            max_pixels: MAX_PIXELS,
            max_decoded_bytes: MAX_DECODED_BYTES,
        }
    }
}

/// Stable OCR-specific failure categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OcrErrorCode {
    /// Request shape, image type, or option is invalid.
    InvalidRequest,
    /// Raster bytes cannot be decoded safely.
    InvalidImage,
    /// No verified and warm backend is available.
    ModelUnavailable,
    /// A bounded resource limit is exhausted.
    Overloaded,
    /// Work was cancelled.
    Cancelled,
    /// Work exceeded its deadline.
    DeadlineExceeded,
    /// An unexpected internal failure occurred.
    Internal,
}

impl OcrErrorCode {
    /// Returns the stable public wire representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::InvalidImage => "invalid_image",
            Self::ModelUnavailable => "model_unavailable",
            Self::Overloaded => "overloaded",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Internal => "internal",
        }
    }
}

/// Privacy-safe OCR error. It never stores input bytes, recognized text, or paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OcrError {
    code: OcrErrorCode,
    message: &'static str,
}

impl OcrError {
    /// Builds the canonical public error for `code`.
    #[must_use]
    pub const fn for_code(code: OcrErrorCode) -> Self {
        let message = match code {
            OcrErrorCode::InvalidRequest => "the OCR request is invalid",
            OcrErrorCode::InvalidImage => "the raster image is invalid",
            OcrErrorCode::ModelUnavailable => "the OCR model bundle is unavailable",
            OcrErrorCode::Overloaded => "the OCR service is temporarily overloaded",
            OcrErrorCode::Cancelled => "the OCR request was cancelled",
            OcrErrorCode::DeadlineExceeded => "the OCR request deadline was exceeded",
            OcrErrorCode::Internal => "an internal OCR error occurred",
        };
        Self { code, message }
    }

    /// Returns the stable error category.
    #[must_use]
    pub const fn code(self) -> OcrErrorCode {
        self.code
    }

    /// Returns the fixed privacy-safe message.
    #[must_use]
    pub const fn message(self) -> &'static str {
        self.message
    }
}

impl fmt::Display for OcrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for OcrError {}

/// MIME type admitted by the raster-only v0.1 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RasterFormat {
    /// Portable Network Graphics.
    Png,
    /// Baseline or progressive JPEG, subject to decoder validation.
    Jpeg,
}

/// Validated metadata supplied alongside encoded raster bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputMetadata {
    /// Declared raster format. A decoder must still verify magic bytes.
    pub format: RasterFormat,
    /// Encoded byte length.
    pub encoded_bytes: u64,
    /// Declared pixel width.
    pub width: u32,
    /// Declared pixel height.
    pub height: u32,
}

impl InputMetadata {
    /// Validates size and dimension bounds without decoding the image.
    ///
    /// # Errors
    /// Returns `invalid_request` when any value is zero or exceeds a product bound.
    pub fn validate(self) -> Result<(), OcrError> {
        self.validate_with(RasterLimits::default())
    }

    /// Validates size and dimension bounds against a configured limit set.
    ///
    /// # Errors
    /// Returns `invalid_request` when any value is zero or exceeds a configured bound.
    pub fn validate_with(self, limits: RasterLimits) -> Result<(), OcrError> {
        let dimensions_valid = self.width > 0
            && self.height > 0
            && self.width <= limits.max_width()
            && self.height <= limits.max_height();
        let pixels = u64::from(self.width).checked_mul(u64::from(self.height));
        if self.encoded_bytes == 0
            || self.encoded_bytes > limits.max_encoded_bytes()
            || !dimensions_valid
            || pixels.is_none_or(|value| value > limits.max_pixels())
            || pixels
                .and_then(|value| value.checked_mul(3))
                .is_none_or(|value| value > limits.max_decoded_bytes())
        {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
        Ok(())
    }
}

/// Encoded raster input whose debug representation omits bytes.
#[derive(Clone)]
pub struct RasterInput {
    bytes: Arc<[u8]>,
    metadata: InputMetadata,
}

impl RasterInput {
    /// Constructs input after checking metadata and encoded length consistency.
    ///
    /// # Errors
    /// Returns `invalid_request` when metadata is invalid or the lengths differ.
    pub fn new(bytes: impl Into<Arc<[u8]>>, metadata: InputMetadata) -> Result<Self, OcrError> {
        metadata.validate()?;
        let bytes = bytes.into();
        if u64::try_from(bytes.len()).ok() != Some(metadata.encoded_bytes) {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
        Ok(Self { bytes, metadata })
    }

    /// Returns encoded bytes for the trusted decoder boundary.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns validated metadata.
    #[must_use]
    pub const fn metadata(&self) -> InputMetadata {
        self.metadata
    }
}

impl fmt::Debug for RasterInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RasterInput")
            .field("bytes", &"[REDACTED]")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Language contract supported by the initial service.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrLanguage {
    /// English text with ASCII digits, punctuation, and spaces.
    #[default]
    English,
}

/// Caller-controlled OCR behavior within the v0.1 contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OcrOptions {
    /// Requested language contract.
    pub language: OcrLanguage,
    /// Whether word-level geometry should be returned.
    pub include_words: bool,
    /// Minimum confidence retained in canonical output.
    pub minimum_confidence: Confidence,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            language: OcrLanguage::English,
            include_words: true,
            minimum_confidence: Confidence::ZERO,
        }
    }
}

/// Finite confidence in the inclusive range 0 through 1.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Confidence(f32);

impl Confidence {
    /// Zero confidence.
    pub const ZERO: Self = Self(0.0);

    /// Constructs a bounded finite confidence.
    ///
    /// # Errors
    /// Returns `invalid_request` when the value is non-finite or outside 0 through 1.
    pub fn new(value: f32) -> Result<Self, OcrError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(OcrError::for_code(OcrErrorCode::InvalidRequest))
        }
    }

    /// Returns the numeric value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Confidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Self::new(value).map_err(|_| serde::de::Error::custom("confidence must be between 0 and 1"))
    }
}

/// One finite image-space coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Horizontal coordinate in pixels.
    pub x: f32,
    /// Vertical coordinate in pixels.
    pub y: f32,
}

/// Four-corner polygon in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    /// Corners in clockwise order beginning at the logical top-left corner.
    pub points: [Point; 4],
}

impl Polygon {
    #[allow(clippy::cast_precision_loss)] // The product cap is 16,384, exactly representable in f32.
    fn validate(self, width: u32, height: u32) -> Result<(), OcrError> {
        let max_x = width as f32;
        let max_y = height as f32;
        let points_valid = self.points.iter().all(|point| {
            point.x.is_finite()
                && point.y.is_finite()
                && (0.0..=max_x).contains(&point.x)
                && (0.0..=max_y).contains(&point.y)
        });
        let signed_double_area = self
            .points
            .iter()
            .zip(self.points.iter().cycle().skip(1))
            .take(self.points.len())
            .map(|(current, next)| current.x * next.y - next.x * current.y)
            .sum::<f32>();
        if points_valid && signed_double_area.is_finite() && signed_double_area > 0.0 {
            Ok(())
        } else {
            Err(OcrError::for_code(OcrErrorCode::Internal))
        }
    }
}

/// Sorts items into the canonical Paddle-compatible top-to-bottom, then left-to-right order.
///
/// Items are first stably sorted by the top-left corner's vertical and horizontal coordinates.
/// Paddle's reference OCR system then performs backward adjacent swaps when neighboring top-left
/// corners differ vertically by less than [`READING_ORDER_ROW_TOLERANCE`] and are horizontally
/// reversed. Keeping the whole operation here ensures assembly and validation use one definition,
/// including its strict tolerance boundary and stable tie behavior.
pub fn sort_reading_order<T>(items: &mut [T], polygon: impl Fn(&T) -> Polygon) {
    items.sort_by(|left, right| {
        let left = polygon(left).points[0];
        let right = polygon(right).points[0];
        compare_key((left.y, left.x), (right.y, right.x))
    });
    for index in 0..items.len().saturating_sub(1) {
        for previous in (0..=index).rev() {
            let left = polygon(&items[previous]).points[0];
            let right = polygon(&items[previous + 1]).points[0];
            if (right.y - left.y).abs() < READING_ORDER_ROW_TOLERANCE && right.x < left.x {
                items.swap(previous, previous + 1);
            } else {
                break;
            }
        }
    }
}

/// Recognized word and its geometry.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrWord {
    /// Recognized UTF-8 text.
    pub text: String,
    /// Word confidence.
    pub confidence: Confidence,
    /// Word polygon.
    pub polygon: Polygon,
}

impl fmt::Debug for OcrWord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcrWord")
            .field("text", &"[REDACTED]")
            .field("confidence", &self.confidence)
            .field("polygon", &self.polygon)
            .finish()
    }
}

/// Recognized line and optional words.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrLine {
    /// Recognized UTF-8 text.
    pub text: String,
    /// Line confidence.
    pub confidence: Confidence,
    /// Line polygon.
    pub polygon: Polygon,
    /// Word results in deterministic reading order.
    pub words: Vec<OcrWord>,
}

impl fmt::Debug for OcrLine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcrLine")
            .field("text", &"[REDACTED]")
            .field("confidence", &self.confidence)
            .field("polygon", &self.polygon)
            .field("words", &self.words)
            .finish()
    }
}

/// Related group of text lines.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrBlock {
    /// Block polygon.
    pub polygon: Polygon,
    /// Lines in deterministic reading order.
    pub lines: Vec<OcrLine>,
}

impl fmt::Debug for OcrBlock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcrBlock")
            .field("polygon", &self.polygon)
            .field("lines", &self.lines)
            .finish()
    }
}

/// One raster page. Version 0.1 always returns page index zero.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    /// Zero-based page index.
    pub index: u32,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Blocks in deterministic reading order.
    pub blocks: Vec<OcrBlock>,
}

impl fmt::Debug for OcrPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcrPage")
            .field("index", &self.index)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("blocks", &self.blocks)
            .finish()
    }
}

/// Canonical OCR result.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrResult {
    /// Exactly one page for raster v0.1.
    pub pages: Vec<OcrPage>,
}

impl fmt::Debug for OcrResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OcrResult")
            .field("pages", &self.pages)
            .finish()
    }
}

impl OcrResult {
    /// Validates raster cardinality, geometry, finite values, and deterministic ordering.
    ///
    /// # Errors
    /// Returns a sanitized internal error for invalid backend output.
    pub fn validate(&self) -> Result<(), OcrError> {
        if self.pages.len() != 1 || self.pages[0].index != 0 {
            return Err(OcrError::for_code(OcrErrorCode::Internal));
        }
        let page = &self.pages[0];
        if page.width == 0 || page.height == 0 {
            return Err(OcrError::for_code(OcrErrorCode::Internal));
        }
        validate_reading_order(&page.blocks, |block| block.polygon)?;
        for block in &page.blocks {
            block.polygon.validate(page.width, page.height)?;
            validate_reading_order(&block.lines, |line| line.polygon)?;
            for line in &block.lines {
                line.polygon.validate(page.width, page.height)?;
                validate_text(&line.text)?;
                validate_reading_order(&line.words, |word| word.polygon)?;
                for word in &line.words {
                    word.polygon.validate(page.width, page.height)?;
                    validate_text(&word.text)?;
                }
            }
        }
        Ok(())
    }
}

fn validate_text(text: &str) -> Result<(), OcrError> {
    if !text.is_empty() && text.is_ascii() && !text.chars().any(char::is_control) {
        Ok(())
    } else {
        Err(OcrError::for_code(OcrErrorCode::Internal))
    }
}

fn validate_reading_order<T>(items: &[T], polygon: impl Fn(&T) -> Polygon) -> Result<(), OcrError> {
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(items.len())
        .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
    indices.extend(0..items.len());
    sort_reading_order(&mut indices, |index| polygon(&items[*index]));
    if indices.iter().copied().eq(0..items.len()) {
        Ok(())
    } else {
        Err(OcrError::for_code(OcrErrorCode::Internal))
    }
}

fn compare_key(first: (f32, f32), second: (f32, f32)) -> Ordering {
    first
        .0
        .total_cmp(&second.0)
        .then_with(|| first.1.total_cmp(&second.1))
}

#[cfg(test)]
mod tests {
    use super::{
        Confidence, InputMetadata, MAX_DECODED_BYTES, MAX_DIMENSION, OcrBlock, OcrErrorCode,
        OcrLine, OcrPage, OcrResult, OcrWord, Point, Polygon, READING_ORDER_ROW_TOLERANCE,
        RasterFormat, RasterInput, RasterLimits, sort_reading_order,
    };

    fn polygon(x: f32, y: f32) -> Polygon {
        Polygon {
            points: [
                Point { x, y },
                Point { x: x + 1.0, y },
                Point {
                    x: x + 1.0,
                    y: y + 1.0,
                },
                Point { x, y: y + 1.0 },
            ],
        }
    }

    fn block(x: f32, y: f32, text: &str) -> OcrBlock {
        let polygon = polygon(x, y);
        OcrBlock {
            polygon,
            lines: vec![OcrLine {
                text: text.to_owned(),
                confidence: Confidence::ZERO,
                polygon,
                words: Vec::new(),
            }],
        }
    }

    #[test]
    fn confidence_rejects_non_finite_and_out_of_range_values() {
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert_eq!(
                Confidence::new(value).map_err(super::OcrError::code),
                Err(OcrErrorCode::InvalidRequest)
            );
        }
        assert_eq!(Confidence::new(1.0).map(Confidence::get), Ok(1.0));
    }

    #[test]
    fn input_bytes_are_redacted_and_bounds_are_checked() -> Result<(), Box<dyn std::error::Error>> {
        let bytes = b"private-raster-sentinel".to_vec();
        let metadata = InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes: u64::try_from(bytes.len())?,
            width: 1,
            height: 1,
        };
        let input = RasterInput::new(bytes, metadata)?;
        assert!(!format!("{input:?}").contains("private-raster-sentinel"));
        let invalid = InputMetadata {
            width: MAX_DIMENSION + 1,
            ..metadata
        };
        assert!(invalid.validate().is_err());
        Ok(())
    }

    #[test]
    fn result_rejects_multiple_pages_and_out_of_bounds_geometry() {
        let empty = OcrPage {
            index: 0,
            width: 10,
            height: 10,
            blocks: Vec::new(),
        };
        assert!(
            OcrResult {
                pages: vec![empty.clone()]
            }
            .validate()
            .is_ok()
        );
        assert!(
            OcrResult {
                pages: vec![empty.clone(), empty]
            }
            .validate()
            .is_err()
        );
        assert!(polygon(11.0, 0.0).validate(10, 10).is_err());
    }

    #[test]
    fn reading_order_matches_paddle_adjacent_row_swaps() {
        let mut items = vec![
            ("right-high", polygon(30.0, 0.0)),
            ("left-middle", polygon(10.0, 5.0)),
            ("middle-low", polygon(20.0, 9.0)),
        ];
        sort_reading_order(&mut items, |item| item.1);
        assert_eq!(
            items.iter().map(|item| item.0).collect::<Vec<_>>(),
            vec!["left-middle", "middle-low", "right-high"]
        );
    }

    #[test]
    fn reading_order_tolerance_is_strict_and_equal_keys_are_stable() {
        let mut rows = vec![
            ("upper-right", polygon(30.0, 0.0)),
            ("lower-left", polygon(10.0, READING_ORDER_ROW_TOLERANCE)),
        ];
        sort_reading_order(&mut rows, |item| item.1);
        assert_eq!(
            rows.iter().map(|item| item.0).collect::<Vec<_>>(),
            vec!["upper-right", "lower-left"]
        );

        let shared = polygon(2.0, 3.0);
        let mut ties = vec![(0_u8, shared), (1, shared), (2, shared)];
        sort_reading_order(&mut ties, |item| item.1);
        assert_eq!(
            ties.iter().map(|item| item.0).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn result_validation_uses_the_same_canonical_reading_order() {
        let canonical = vec![
            block(10.0, 5.0, "left"),
            block(20.0, 9.0, "middle"),
            block(30.0, 0.0, "right"),
        ];
        let result = OcrResult {
            pages: vec![OcrPage {
                index: 0,
                width: 64,
                height: 32,
                blocks: canonical.clone(),
            }],
        };
        assert!(result.validate().is_ok());

        let mut noncanonical = canonical;
        noncanonical.rotate_right(1);
        assert!(
            OcrResult {
                pages: vec![OcrPage {
                    index: 0,
                    width: 64,
                    height: 32,
                    blocks: noncanonical,
                }],
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn configured_raster_limits_validate_every_bound() -> Result<(), Box<dyn std::error::Error>> {
        let limits = RasterLimits::new(8, 4, 5, 20, 60)?;
        assert_eq!(limits.max_encoded_bytes(), 8);
        assert_eq!(limits.max_width(), 4);
        assert_eq!(limits.max_height(), 5);
        assert_eq!(limits.max_pixels(), 20);
        assert_eq!(limits.max_decoded_bytes(), 60);
        for invalid in [
            RasterLimits::new(0, 1, 1, 1, 3),
            RasterLimits::new(1, 0, 1, 1, 3),
            RasterLimits::new(1, 1, 0, 1, 3),
            RasterLimits::new(1, 1, 1, 0, 3),
            RasterLimits::new(1, 1, 1, 1, 0),
            RasterLimits::new(1, 1, 1, 1, MAX_DECODED_BYTES + 1),
        ] {
            assert!(invalid.is_err());
        }
        let metadata = InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes: 8,
            width: 4,
            height: 5,
        };
        assert!(metadata.validate_with(limits).is_ok());
        assert!(
            InputMetadata {
                width: 5,
                ..metadata
            }
            .validate_with(limits)
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn confidence_deserialization_cannot_bypass_validation() {
        let invalid = serde::de::value::F32Deserializer::<serde::de::value::Error>::new(1.1);
        assert!(
            serde::Deserialize::deserialize(invalid)
                .map(|_: Confidence| ())
                .is_err()
        );
        let valid = serde::de::value::F32Deserializer::<serde::de::value::Error>::new(0.25);
        assert_eq!(
            serde::Deserialize::deserialize(valid).map(Confidence::get),
            Ok(0.25)
        );
    }

    #[test]
    fn nested_result_debug_redacts_all_recognized_text() -> Result<(), Box<dyn std::error::Error>> {
        let sentinel = "private-recognized-sentinel";
        let confidence = Confidence::new(0.75)?;
        let word = OcrWord {
            text: sentinel.to_owned(),
            confidence,
            polygon: polygon(1.0, 1.0),
        };
        assert!(!format!("{word:?}").contains(sentinel));
        let line = OcrLine {
            text: sentinel.to_owned(),
            confidence,
            polygon: polygon(1.0, 1.0),
            words: vec![word],
        };
        assert!(!format!("{line:?}").contains(sentinel));
        let result = OcrResult {
            pages: vec![OcrPage {
                index: 0,
                width: 10,
                height: 10,
                blocks: vec![OcrBlock {
                    polygon: polygon(0.0, 0.0),
                    lines: vec![line],
                }],
            }],
        };
        assert!(!format!("{result:?}").contains(sentinel));
        Ok(())
    }
}
