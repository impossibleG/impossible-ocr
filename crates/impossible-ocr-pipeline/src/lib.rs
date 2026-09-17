//! Backend-independent synchronous OCR pipeline boundaries.

use std::{future::Future, pin::Pin};

mod crop;
mod ctc;
mod db;
mod detector;
mod engine;
mod geometry;
mod raster;
mod recognizer;
mod resize;

pub use crop::perspective_crop;
pub use ctc::{CtcDecoder, DecodedSequence};
pub use db::{
    DB_BOX_THRESHOLD, DB_MAX_CANDIDATES, DB_THRESHOLD, DB_UNCLIP_RATIO, DbPostProcessor,
    DetectedTextRegion, DetectorMap,
};
pub use detector::{DETECTOR_MAX_SIDE, DetectorInput, preprocess_detector};
pub use engine::{
    DetectorTensorOutput, PpOcrBackend, RECOGNIZER_BATCH_LIMIT, RecognizerTensorOutput,
    TensorOcrRuntime,
};
pub use raster::{
    ColorProfilePolicy, DecodedRaster, MetadataPolicy, OrientationPolicy, RasterDecoder,
};
pub use recognizer::{
    RECOGNIZER_HEIGHT, RECOGNIZER_MAX_WIDTH, RecognizerBatch, preprocess_recognizer_batch,
};

use impossible_ocr_domain::{OcrError, OcrErrorCode, OcrOptions, OcrResult, RasterInput};
use impossible_server_core::RequestContext;

#[cfg(test)]
mod engine_tests;

/// Boxed backend future tied to the backend borrow.
pub type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Observable backend lifecycle without exposing paths or model identifiers publicly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendState {
    /// No complete verified bundle is installed.
    Missing,
    /// The bundle is being validated or warmed.
    Warming,
    /// The complete bundle is verified and warm.
    Ready,
    /// Initialization failed and must be retried explicitly.
    Failed,
}

/// Extension interface implemented by a qualified OCR backend.
pub trait OcrBackend: Send + Sync + 'static {
    /// Returns current lifecycle state.
    fn state(&self) -> BackendState;

    /// Verifies and warms the complete backend bundle.
    fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>>;

    /// Performs one synchronous raster OCR operation.
    fn recognize<'a>(
        &'a self,
        input: &'a RasterInput,
        options: OcrOptions,
        context: &'a RequestContext,
    ) -> BackendFuture<'a, Result<OcrResult, OcrError>>;
}

/// Validates inputs and outputs around a backend implementation.
#[derive(Debug)]
pub struct OcrPipeline<B> {
    backend: B,
}

impl<B: OcrBackend> OcrPipeline<B> {
    /// Creates a pipeline around one backend.
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Returns whether the backend can currently accept work.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.backend.state() == BackendState::Ready
    }

    /// Verifies and warms the backend.
    ///
    /// # Errors
    /// Returns a stable backend error when integrity checks or warm-up fail.
    pub async fn warm_up(&self) -> Result<(), OcrError> {
        self.backend.warm_up().await
    }

    /// Runs one bounded synchronous request.
    ///
    /// # Errors
    /// Returns a stable public error when the backend is unavailable or either contract fails.
    pub async fn recognize(
        &self,
        input: &RasterInput,
        options: OcrOptions,
        context: &RequestContext,
    ) -> Result<OcrResult, OcrError> {
        input.metadata().validate()?;
        if !self.is_ready() {
            return Err(OcrError::for_code(OcrErrorCode::ModelUnavailable));
        }
        let result = self.backend.recognize(input, options, context).await?;
        result.validate()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use impossible_ocr_domain::{
        InputMetadata, OcrError, OcrErrorCode, OcrOptions, OcrPage, OcrResult, RasterFormat,
        RasterInput,
    };
    use impossible_server_core::{CancellationToken, RequestContext, RequestIdSource};

    use super::{BackendFuture, BackendState, OcrBackend, OcrPipeline};

    #[derive(Debug)]
    struct StubBackend(BackendState);

    impl OcrBackend for StubBackend {
        fn state(&self) -> BackendState {
            self.0
        }

        fn warm_up(&self) -> BackendFuture<'_, Result<(), OcrError>> {
            Box::pin(async { Ok(()) })
        }

        fn recognize<'a>(
            &'a self,
            input: &'a RasterInput,
            _options: OcrOptions,
            _context: &'a RequestContext,
        ) -> BackendFuture<'a, Result<OcrResult, OcrError>> {
            Box::pin(async move {
                let metadata = input.metadata();
                Ok(OcrResult {
                    pages: vec![OcrPage {
                        index: 0,
                        width: metadata.width,
                        height: metadata.height,
                        blocks: Vec::new(),
                    }],
                })
            })
        }
    }

    fn input() -> Result<RasterInput, OcrError> {
        RasterInput::new(
            Vec::from(*b"test"),
            InputMetadata {
                format: RasterFormat::Png,
                encoded_bytes: 4,
                width: 1,
                height: 1,
            },
        )
    }

    fn context() -> Result<RequestContext, Box<dyn std::error::Error>> {
        Ok(RequestContext::new(
            RequestIdSource::default().next()?,
            CancellationToken::new(),
            Some(Duration::from_secs(1)),
        )?)
    }

    #[test]
    fn missing_backend_fails_closed() -> Result<(), Box<dyn std::error::Error>> {
        let pipeline = OcrPipeline::new(StubBackend(BackendState::Missing));
        let runtime = tokio_test_runtime()?;
        let error =
            runtime.block_on(pipeline.recognize(&input()?, OcrOptions::default(), &context()?));
        assert_eq!(
            error.map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::ModelUnavailable)
        );
        Ok(())
    }

    #[test]
    fn ready_backend_result_is_validated() -> Result<(), Box<dyn std::error::Error>> {
        let pipeline = OcrPipeline::new(StubBackend(BackendState::Ready));
        let runtime = tokio_test_runtime()?;
        let result =
            runtime.block_on(pipeline.recognize(&input()?, OcrOptions::default(), &context()?))?;
        assert_eq!(result.pages.len(), 1);
        Ok(())
    }

    fn tokio_test_runtime() -> Result<tokio::runtime::Runtime, std::io::Error> {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
    }
}
