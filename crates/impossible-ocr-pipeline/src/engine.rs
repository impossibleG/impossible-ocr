use std::{fmt, sync::Arc};

use impossible_ocr_domain::{
    MAX_PIXELS, OcrBlock, OcrError, OcrErrorCode, OcrLine, OcrOptions, OcrPage, OcrResult, OcrWord,
    Polygon, RasterInput, sort_reading_order,
};
use impossible_server_core::RequestContext;

use crate::{
    BackendFuture, BackendState, CtcDecoder, DbPostProcessor, DecodedSequence, DetectorInput,
    DetectorMap, OcrBackend, RasterDecoder, RecognizerBatch, perspective_crop, preprocess_detector,
    preprocess_recognizer_batch,
};

/// Maximum number of text crops submitted to one recognizer invocation.
pub const RECOGNIZER_BATCH_LIMIT: usize = 8;

/// Validated detector tensor output. Probability data is redacted from `Debug`.
#[derive(Clone, PartialEq)]
pub struct DetectorTensorOutput {
    map: DetectorMap,
}

impl DetectorTensorOutput {
    /// Constructs a finite singleton `[1, 1, height, width]` probability tensor.
    ///
    /// # Errors
    /// Returns a sanitized internal error when dimensions, length, or values violate the admitted
    /// detector output contract.
    pub fn new(probabilities: Vec<f32>, width: u32, height: u32) -> Result<Self, OcrError> {
        Ok(Self {
            map: DetectorMap::new(probabilities, width, height)?,
        })
    }

    /// Probability-map width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.map.width()
    }

    /// Probability-map height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.map.height()
    }

    fn into_map(self, input: &DetectorInput) -> Result<DetectorMap, OcrError> {
        if self.map.width() != input.tensor_width() || self.map.height() != input.tensor_height() {
            return Err(internal());
        }
        Ok(self.map)
    }
}

impl fmt::Debug for DetectorTensorOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DetectorTensorOutput")
            .field("probabilities", &"[REDACTED]")
            .field("width", &self.map.width())
            .field("height", &self.map.height())
            .finish()
    }
}

/// Validated row-major recognizer tensor with shape `[batch, time, classes]`.
#[derive(Clone, PartialEq)]
pub struct RecognizerTensorOutput {
    probabilities: Vec<f32>,
    batch_size: usize,
    time_steps: usize,
    class_count: usize,
}

impl RecognizerTensorOutput {
    /// Constructs a finite recognizer probability tensor.
    ///
    /// # Errors
    /// Returns a sanitized internal error for zero dimensions, checked shape overflow, a length
    /// mismatch, or a probability outside the inclusive range zero through one.
    pub fn new(
        probabilities: Vec<f32>,
        batch_size: usize,
        time_steps: usize,
        class_count: usize,
    ) -> Result<Self, OcrError> {
        let expected = batch_size
            .checked_mul(time_steps)
            .and_then(|length| length.checked_mul(class_count))
            .ok_or_else(internal)?;
        if batch_size == 0
            || time_steps == 0
            || class_count == 0
            || probabilities.len() != expected
            || probabilities
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(internal());
        }
        Ok(Self {
            probabilities,
            batch_size,
            time_steps,
            class_count,
        })
    }

    /// Number of batch items.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Number of timesteps per batch item.
    #[must_use]
    pub const fn time_steps(&self) -> usize {
        self.time_steps
    }

    /// Number of model classes per timestep, including blank class zero.
    #[must_use]
    pub const fn class_count(&self) -> usize {
        self.class_count
    }

    /// Reports whether every timestep is a normalized categorical probability distribution.
    ///
    /// This exposes only a boolean invariant for qualification; tensor values remain private and
    /// redacted. A non-finite, negative, or overly large tolerance is rejected.
    #[must_use]
    pub fn probability_rows_are_normalized(&self, tolerance: f32) -> bool {
        if !tolerance.is_finite() || !(0.0..=0.01).contains(&tolerance) {
            return false;
        }
        self.probabilities
            .chunks_exact(self.class_count)
            .all(|row| (row.iter().copied().sum::<f32>() - 1.0).abs() <= tolerance)
    }

    fn decode(
        self,
        expected_batch: usize,
        decoder: &CtcDecoder,
    ) -> Result<Vec<DecodedSequence>, OcrError> {
        if self.batch_size != expected_batch || self.class_count != decoder.class_count() {
            return Err(internal());
        }
        let item_length = self
            .time_steps
            .checked_mul(self.class_count)
            .ok_or_else(internal)?;
        let mut sequences = Vec::new();
        sequences
            .try_reserve_exact(self.batch_size)
            .map_err(|_| internal())?;
        for probabilities in self.probabilities.chunks_exact(item_length) {
            sequences.push(decoder.decode_probabilities(
                probabilities,
                self.time_steps,
                self.class_count,
            )?);
        }
        if sequences.len() != self.batch_size {
            return Err(internal());
        }
        Ok(sequences)
    }
}

impl fmt::Debug for RecognizerTensorOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecognizerTensorOutput")
            .field("probabilities", &"[REDACTED]")
            .field("batch_size", &self.batch_size)
            .field("time_steps", &self.time_steps)
            .field("class_count", &self.class_count)
            .finish()
    }
}

/// Raw synchronous tensor-inference boundary implemented by a qualified model runtime.
///
/// Implementations receive only validated, preprocessed tensors and return validated output tensor
/// wrappers. They must not perform image decoding, OCR post-processing, or transport work.
pub trait TensorOcrRuntime: Send + Sync + 'static {
    /// Current lifecycle state of the complete detector/recognizer model bundle.
    fn state(&self) -> BackendState;

    /// Verifies and warms the complete model bundle.
    ///
    /// # Errors
    /// Returns a sanitized model or internal error when warm-up fails.
    fn warm_up(&self) -> Result<(), OcrError>;

    /// Runs the detector tensor.
    ///
    /// # Errors
    /// Returns a sanitized runtime or model-output error.
    fn run_detector(
        &self,
        input: &DetectorInput,
        context: &RequestContext,
    ) -> Result<DetectorTensorOutput, OcrError>;

    /// Runs one recognizer tensor batch.
    ///
    /// # Errors
    /// Returns a sanitized runtime or model-output error.
    fn run_recognizer(
        &self,
        input: &RecognizerBatch,
        context: &RequestContext,
    ) -> Result<RecognizerTensorOutput, OcrError>;
}

/// Production PP-OCR orchestration over an injected raw tensor runtime.
pub struct PpOcrBackend<R> {
    runtime: Arc<R>,
    raster_decoder: RasterDecoder,
    ctc_decoder: CtcDecoder,
}

impl<R> PpOcrBackend<R> {
    /// Creates a backend from a shared runtime, bounded raster decoder, and exact model dictionary.
    #[must_use]
    pub const fn new(
        runtime: Arc<R>,
        raster_decoder: RasterDecoder,
        ctc_decoder: CtcDecoder,
    ) -> Self {
        Self {
            runtime,
            raster_decoder,
            ctc_decoder,
        }
    }
}

impl<R> fmt::Debug for PpOcrBackend<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PpOcrBackend")
            .field("runtime", &"[REDACTED]")
            .field("raster_decoder", &self.raster_decoder)
            .field("ctc_decoder", &self.ctc_decoder)
            .finish()
    }
}

impl<R: TensorOcrRuntime> OcrBackend for PpOcrBackend<R> {
    fn state(&self) -> BackendState {
        self.runtime.state()
    }

    fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
        let runtime = Arc::clone(&self.runtime);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || runtime.warm_up())
                .await
                .map_err(|_| internal())?
        })
    }

    fn recognize<'a>(
        &'a self,
        input: &'a RasterInput,
        options: OcrOptions,
        context: &'a RequestContext,
    ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
        let runtime = Arc::clone(&self.runtime);
        let raster_decoder = self.raster_decoder;
        let ctc_decoder = self.ctc_decoder.clone();
        let input = input.clone();
        let context = context.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                recognize_blocking(
                    runtime.as_ref(),
                    raster_decoder,
                    &ctc_decoder,
                    &input,
                    options,
                    &context,
                )
            })
            .await
            .map_err(|_| internal())?
        })
    }
}

fn recognize_blocking<R: TensorOcrRuntime>(
    runtime: &R,
    raster_decoder: RasterDecoder,
    ctc_decoder: &CtcDecoder,
    input: &RasterInput,
    options: OcrOptions,
    context: &RequestContext,
) -> Result<OcrResult, OcrError> {
    check_context(context)?;
    let raster = raster_decoder.decode(input)?;
    check_context(context)?;

    let detector_input = preprocess_detector(&raster)?;
    check_context(context)?;
    let detector_output = runtime.run_detector(&detector_input, context)?;
    check_context(context)?;
    let detector_map = detector_output.into_map(&detector_input)?;
    let mut regions = DbPostProcessor.process(&detector_map, &detector_input, context)?;
    sort_reading_order(&mut regions, |region| region.polygon());
    check_context(context)?;

    if regions.is_empty() {
        return finish_result(raster.width(), raster.height(), Vec::new());
    }

    let mut crops = Vec::new();
    crops
        .try_reserve_exact(regions.len())
        .map_err(|_| internal())?;
    let mut temporary_pixels = 0_u64;
    for region in &regions {
        check_context(context)?;
        temporary_pixels = temporary_pixels
            .checked_add(crop_pixel_count(region.polygon())?)
            .filter(|pixels| *pixels <= MAX_PIXELS)
            .ok_or_else(overloaded)?;
        crops.push(perspective_crop(&raster, region.polygon())?);
    }
    check_context(context)?;

    let mut sequences = Vec::new();
    sequences
        .try_reserve_exact(crops.len())
        .map_err(|_| internal())?;
    for chunk in crops.chunks(RECOGNIZER_BATCH_LIMIT) {
        check_context(context)?;
        let batch = preprocess_recognizer_batch(chunk)?;
        let output = runtime.run_recognizer(&batch, context)?;
        check_context(context)?;
        let sorted = output.decode(batch.batch_size(), ctc_decoder)?;
        sequences.extend(batch.restore_order(sorted)?);
    }
    if sequences.len() != regions.len() {
        return Err(internal());
    }
    check_context(context)?;

    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(regions.len())
        .map_err(|_| internal())?;
    for (region, sequence) in regions.into_iter().zip(sequences) {
        check_context(context)?;
        if sequence.text.is_empty() || sequence.confidence < options.minimum_confidence {
            continue;
        }
        let polygon = region.polygon();
        let words = if options.include_words {
            vec![OcrWord {
                text: sequence.text.clone(),
                confidence: sequence.confidence,
                polygon,
            }]
        } else {
            Vec::new()
        };
        blocks.push(OcrBlock {
            polygon,
            lines: vec![OcrLine {
                text: sequence.text,
                confidence: sequence.confidence,
                polygon,
                words,
            }],
        });
    }
    finish_result(raster.width(), raster.height(), blocks)
}

fn finish_result(width: u32, height: u32, blocks: Vec<OcrBlock>) -> Result<OcrResult, OcrError> {
    let result = OcrResult {
        pages: vec![OcrPage {
            index: 0,
            width,
            height,
            blocks,
        }],
    };
    result.validate()?;
    Ok(result)
}

fn crop_pixel_count(polygon: Polygon) -> Result<u64, OcrError> {
    let [top_left, top_right, bottom_right, bottom_left] = polygon.points;
    let width = edge_length(top_left, top_right)
        .max(edge_length(bottom_left, bottom_right))
        .floor();
    let height = edge_length(top_left, bottom_left)
        .max(edge_length(top_right, bottom_right))
        .floor();
    if !width.is_finite()
        || !height.is_finite()
        || width < 1.0
        || height < 1.0
        || width > f64::from(u32::MAX)
        || height > f64::from(u32::MAX)
    {
        return Err(internal());
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let width = width as u64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let height = height as u64;
    width.checked_mul(height).ok_or_else(internal)
}

fn edge_length(first: impossible_ocr_domain::Point, second: impossible_ocr_domain::Point) -> f64 {
    f64::from(second.x - first.x).hypot(f64::from(second.y - first.y))
}

fn check_context(context: &RequestContext) -> Result<(), OcrError> {
    if context.cancellation().is_cancelled() {
        return Err(OcrError::for_code(OcrErrorCode::Cancelled));
    }
    if context
        .remaining()
        .is_some_and(|remaining| remaining.is_zero())
    {
        return Err(OcrError::for_code(OcrErrorCode::DeadlineExceeded));
    }
    Ok(())
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

const fn overloaded() -> OcrError {
    OcrError::for_code(OcrErrorCode::Overloaded)
}
