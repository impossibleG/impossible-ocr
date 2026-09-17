use std::{
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};
use impossible_ocr_domain::{
    Confidence, InputMetadata, OcrError, OcrErrorCode, OcrOptions, RasterFormat, RasterInput,
    RasterLimits,
};
use impossible_server_core::{CancellationToken, RequestContext, RequestIdSource};

use crate::{
    BackendState, CtcDecoder, DetectorInput, DetectorTensorOutput, OcrBackend, PpOcrBackend,
    RasterDecoder, RecognizerBatch, RecognizerTensorOutput, TensorOcrRuntime,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FakeMode {
    Normal,
    WrongDetectorShape,
    WrongRecognizerBatch,
    WrongRecognizerClasses,
    CancelAfterDetector,
    CancelAfterRecognizer,
}

#[derive(Debug)]
struct FakeTensorRuntime {
    regions: usize,
    mode: FakeMode,
    recognizer_batches: Mutex<Vec<usize>>,
    warmups: AtomicUsize,
}

impl FakeTensorRuntime {
    fn new(regions: usize, mode: FakeMode) -> Self {
        Self {
            regions,
            mode,
            recognizer_batches: Mutex::new(Vec::new()),
            warmups: AtomicUsize::new(0),
        }
    }

    fn recognizer_batches(&self) -> Result<Vec<usize>, OcrError> {
        self.recognizer_batches
            .lock()
            .map(|batches| batches.clone())
            .map_err(|_| internal())
    }
}

impl TensorOcrRuntime for FakeTensorRuntime {
    fn state(&self) -> BackendState {
        BackendState::Ready
    }

    fn warm_up(&self) -> Result<(), OcrError> {
        self.warmups.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn run_detector(
        &self,
        input: &DetectorInput,
        context: &RequestContext,
    ) -> Result<DetectorTensorOutput, OcrError> {
        if self.mode == FakeMode::WrongDetectorShape {
            return DetectorTensorOutput::new(vec![0.0], 1, 1);
        }
        let width = usize::try_from(input.tensor_width()).map_err(|_| internal())?;
        let height = usize::try_from(input.tensor_height()).map_err(|_| internal())?;
        let length = width.checked_mul(height).ok_or_else(internal)?;
        let mut probabilities = Vec::new();
        probabilities
            .try_reserve_exact(length)
            .map_err(|_| internal())?;
        probabilities.resize(length, 0.0);
        for index in 0..self.regions {
            let x_start = 4_usize
                .checked_add((index % 17).checked_mul(14).ok_or_else(internal)?)
                .ok_or_else(internal)?;
            let y_start = 20_usize
                .checked_add((index / 17).checked_mul(20).ok_or_else(internal)?)
                .ok_or_else(internal)?;
            let x_end = x_start.checked_add(7).ok_or_else(internal)?;
            let y_end = y_start.checked_add(7).ok_or_else(internal)?;
            if x_end >= width || y_end >= height {
                return Err(internal());
            }
            for y in y_start..=y_end {
                for x in x_start..=x_end {
                    probabilities[y * width + x] = 0.9;
                }
            }
        }
        let output =
            DetectorTensorOutput::new(probabilities, input.tensor_width(), input.tensor_height())?;
        if self.mode == FakeMode::CancelAfterDetector {
            let _was_first_cancellation = context.cancellation().cancel();
        }
        Ok(output)
    }

    fn run_recognizer(
        &self,
        input: &RecognizerBatch,
        context: &RequestContext,
    ) -> Result<RecognizerTensorOutput, OcrError> {
        self.recognizer_batches
            .lock()
            .map_err(|_| internal())?
            .push(input.batch_size());
        let output_batch = if self.mode == FakeMode::WrongRecognizerBatch {
            input.batch_size().checked_sub(1).ok_or_else(internal)?
        } else {
            input.batch_size()
        };
        let class_count = if self.mode == FakeMode::WrongRecognizerClasses {
            3
        } else {
            2
        };
        let mut probabilities = Vec::new();
        probabilities
            .try_reserve_exact(output_batch.checked_mul(class_count).ok_or_else(internal)?)
            .map_err(|_| internal())?;
        for _ in 0..output_batch {
            if class_count == 2 {
                probabilities.extend_from_slice(&[0.1, 0.9]);
            } else {
                probabilities.extend_from_slice(&[0.1, 0.8, 0.1]);
            }
        }
        let output = RecognizerTensorOutput::new(probabilities, output_batch, 1, class_count)?;
        if self.mode == FakeMode::CancelAfterRecognizer {
            let _was_first_cancellation = context.cancellation().cancel();
        }
        Ok(output)
    }
}

fn backend(runtime: Arc<FakeTensorRuntime>) -> Result<PpOcrBackend<FakeTensorRuntime>, OcrError> {
    Ok(PpOcrBackend::new(
        runtime,
        RasterDecoder::new(RasterLimits::default()),
        CtcDecoder::new(vec!["z".to_owned()], false)?,
    ))
}

fn raster_input(width: u32, height: u32) -> Result<RasterInput, Box<dyn Error>> {
    let length = usize::try_from(u64::from(width) * u64::from(height) * 3)?;
    let pixels = vec![255_u8; length];
    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded).write_image(&pixels, width, height, ColorType::Rgb8.into())?;
    let encoded_bytes = u64::try_from(encoded.len())?;
    Ok(RasterInput::new(
        encoded,
        InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes,
            width,
            height,
        },
    )?)
}

fn context(
    cancellation: CancellationToken,
    timeout: Option<Duration>,
) -> Result<RequestContext, Box<dyn Error>> {
    Ok(RequestContext::new(
        RequestIdSource::default().next()?,
        cancellation,
        timeout,
    )?)
}

fn runtime() -> Result<tokio::runtime::Runtime, std::io::Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
}

#[test]
fn tensor_outputs_validate_shapes_and_redact_values() -> Result<(), OcrError> {
    let detector = DetectorTensorOutput::new(vec![0.123_456], 1, 1)?;
    let detector_debug = format!("{detector:?}");
    assert!(detector_debug.contains("[REDACTED]"));
    assert!(!detector_debug.contains("0.123456"));
    assert!(DetectorTensorOutput::new(vec![0.0], 0, 1).is_err());
    assert!(DetectorTensorOutput::new(vec![f32::NAN], 1, 1).is_err());
    assert!(DetectorTensorOutput::new(vec![0.0], 2, 1).is_err());

    let recognizer = RecognizerTensorOutput::new(vec![0.123_456, 0.876_544], 1, 1, 2)?;
    let recognizer_debug = format!("{recognizer:?}");
    assert!(recognizer_debug.contains("[REDACTED]"));
    assert!(!recognizer_debug.contains("0.123456"));
    assert!(RecognizerTensorOutput::new(vec![0.0], 0, 1, 1).is_err());
    assert!(RecognizerTensorOutput::new(vec![0.0], 1, 0, 1).is_err());
    assert!(RecognizerTensorOutput::new(vec![0.0], 1, 1, 0).is_err());
    assert!(RecognizerTensorOutput::new(vec![f32::INFINITY], 1, 1, 1).is_err());
    assert!(RecognizerTensorOutput::new(vec![1.1], 1, 1, 1).is_err());
    assert!(RecognizerTensorOutput::new(vec![0.0], 1, 1, 2).is_err());
    Ok(())
}

#[test]
fn lifecycle_delegates_and_empty_detection_skips_recognizer() -> Result<(), Box<dyn Error>> {
    let fake = Arc::new(FakeTensorRuntime::new(0, FakeMode::Normal));
    let backend = backend(Arc::clone(&fake))?;
    assert_eq!(backend.state(), BackendState::Ready);
    runtime()?.block_on(backend.warm_up())?;
    assert_eq!(fake.warmups.load(Ordering::Relaxed), 1);

    let input = raster_input(256, 96)?;
    let result = runtime()?.block_on(backend.recognize(
        &input,
        OcrOptions::default(),
        &context(CancellationToken::new(), None)?,
    ))?;
    assert_eq!((result.pages[0].width, result.pages[0].height), (256, 96));
    assert!(result.pages[0].blocks.is_empty());
    assert!(fake.recognizer_batches()?.is_empty());
    Ok(())
}

#[test]
fn seventeen_regions_use_eight_eight_one_and_keep_reading_order() -> Result<(), Box<dyn Error>> {
    let fake = Arc::new(FakeTensorRuntime::new(17, FakeMode::Normal));
    let backend = backend(Arc::clone(&fake))?;
    let input = raster_input(256, 96)?;
    let result = runtime()?.block_on(backend.recognize(
        &input,
        OcrOptions::default(),
        &context(CancellationToken::new(), None)?,
    ))?;
    assert_eq!(fake.recognizer_batches()?, vec![8, 8, 1]);
    let blocks = &result.pages[0].blocks;
    assert_eq!(blocks.len(), 17);
    for pair in blocks.windows(2) {
        assert!(pair[0].polygon.points[0].x < pair[1].polygon.points[0].x);
    }
    for block in blocks {
        assert_eq!(block.lines.len(), 1);
        let line = &block.lines[0];
        assert_eq!(line.text, "z");
        assert_eq!(line.words.len(), 1);
        assert_eq!(line.words[0].text, line.text);
        assert_eq!(line.words[0].polygon, line.polygon);
        assert_eq!(line.polygon, block.polygon);
    }
    result.validate()?;
    Ok(())
}

#[test]
fn include_words_and_minimum_confidence_are_exact() -> Result<(), Box<dyn Error>> {
    let fake = Arc::new(FakeTensorRuntime::new(1, FakeMode::Normal));
    let backend = backend(Arc::clone(&fake))?;
    let input = raster_input(256, 96)?;
    let without_words = runtime()?.block_on(backend.recognize(
        &input,
        OcrOptions {
            include_words: false,
            ..OcrOptions::default()
        },
        &context(CancellationToken::new(), None)?,
    ))?;
    assert_eq!(without_words.pages[0].blocks.len(), 1);
    assert!(without_words.pages[0].blocks[0].lines[0].words.is_empty());

    let filtered = runtime()?.block_on(backend.recognize(
        &input,
        OcrOptions {
            minimum_confidence: Confidence::new(0.91)?,
            ..OcrOptions::default()
        },
        &context(CancellationToken::new(), None)?,
    ))?;
    assert!(filtered.pages[0].blocks.is_empty());
    assert_eq!(fake.recognizer_batches()?, vec![1, 1]);
    Ok(())
}

#[test]
fn malformed_runtime_shapes_fail_closed() -> Result<(), Box<dyn Error>> {
    let input = raster_input(256, 96)?;
    for (mode, regions) in [
        (FakeMode::WrongDetectorShape, 0),
        (FakeMode::WrongRecognizerBatch, 2),
        (FakeMode::WrongRecognizerClasses, 2),
    ] {
        let backend = backend(Arc::new(FakeTensorRuntime::new(regions, mode)))?;
        assert_eq!(
            runtime()?
                .block_on(backend.recognize(
                    &input,
                    OcrOptions::default(),
                    &context(CancellationToken::new(), None)?,
                ))
                .map_err(OcrError::code),
            Err(OcrErrorCode::Internal)
        );
    }
    Ok(())
}

#[test]
fn cancellation_precedes_deadline_and_is_checked_between_stages() -> Result<(), Box<dyn Error>> {
    let input = raster_input(256, 96)?;
    let cancellation = CancellationToken::new();
    assert!(cancellation.cancel());
    let never_called = Arc::new(FakeTensorRuntime::new(1, FakeMode::Normal));
    let initially_cancelled_backend = backend(Arc::clone(&never_called))?;
    assert_eq!(
        runtime()?
            .block_on(initially_cancelled_backend.recognize(
                &input,
                OcrOptions::default(),
                &context(cancellation, Some(Duration::ZERO))?,
            ))
            .map_err(OcrError::code),
        Err(OcrErrorCode::Cancelled)
    );
    assert!(never_called.recognizer_batches()?.is_empty());

    let cancel_after_detector = Arc::new(FakeTensorRuntime::new(1, FakeMode::CancelAfterDetector));
    let cancels_between_stages_backend = backend(Arc::clone(&cancel_after_detector))?;
    assert_eq!(
        runtime()?
            .block_on(cancels_between_stages_backend.recognize(
                &input,
                OcrOptions::default(),
                &context(CancellationToken::new(), None)?,
            ))
            .map_err(OcrError::code),
        Err(OcrErrorCode::Cancelled)
    );
    assert!(cancel_after_detector.recognizer_batches()?.is_empty());

    let cancel_after_recognizer =
        Arc::new(FakeTensorRuntime::new(1, FakeMode::CancelAfterRecognizer));
    let cancels_after_inference_backend = backend(Arc::clone(&cancel_after_recognizer))?;
    assert_eq!(
        runtime()?
            .block_on(cancels_after_inference_backend.recognize(
                &input,
                OcrOptions::default(),
                &context(CancellationToken::new(), None)?,
            ))
            .map_err(OcrError::code),
        Err(OcrErrorCode::Cancelled)
    );
    assert_eq!(cancel_after_recognizer.recognizer_batches()?, vec![1]);

    let deadline = backend(Arc::new(FakeTensorRuntime::new(1, FakeMode::Normal)))?;
    assert_eq!(
        runtime()?
            .block_on(deadline.recognize(
                &input,
                OcrOptions::default(),
                &context(CancellationToken::new(), Some(Duration::ZERO))?,
            ))
            .map_err(OcrError::code),
        Err(OcrErrorCode::DeadlineExceeded)
    );
    Ok(())
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}
