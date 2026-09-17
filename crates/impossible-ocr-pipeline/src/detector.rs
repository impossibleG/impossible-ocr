use std::fmt;

use impossible_ocr_domain::{OcrError, OcrErrorCode};

use crate::{DecodedRaster, resize::resize_inter_linear_rgb8};

/// Maximum detector content side for the pinned `PaddleX` transform.
pub const DETECTOR_MAX_SIDE: u32 = 960;
const STRIDE: u32 = 32;
const MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];

/// Detector input tensor and the exact geometry required to map predictions back.
#[derive(Clone, PartialEq)]
pub struct DetectorInput {
    data: Vec<f32>,
    tensor_width: u32,
    tensor_height: u32,
    content_width: u32,
    content_height: u32,
    pre_resize_width: u32,
    pre_resize_height: u32,
    original_width: u32,
    original_height: u32,
    scale_x: f32,
    scale_y: f32,
}

impl DetectorInput {
    /// NCHW tensor data with shape `[1, 3, tensor_height, tensor_width]` in BGR order.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Tensor width after the exact multiple-of-32 resize.
    #[must_use]
    pub const fn tensor_width(&self) -> u32 {
        self.tensor_width
    }

    /// Tensor height after the exact multiple-of-32 resize.
    #[must_use]
    pub const fn tensor_height(&self) -> u32 {
        self.tensor_height
    }

    /// Width occupied by resized image content (equal to the tensor width).
    #[must_use]
    pub const fn content_width(&self) -> u32 {
        self.content_width
    }

    /// Height occupied by resized image content (equal to the tensor height).
    #[must_use]
    pub const fn content_height(&self) -> u32 {
        self.content_height
    }

    /// Width used as the scale-factor denominator after the tiny-image padding rule.
    #[must_use]
    pub const fn pre_resize_width(&self) -> u32 {
        self.pre_resize_width
    }

    /// Height used as the scale-factor denominator after the tiny-image padding rule.
    #[must_use]
    pub const fn pre_resize_height(&self) -> u32 {
        self.pre_resize_height
    }

    /// Original decoded width.
    #[must_use]
    pub const fn original_width(&self) -> u32 {
        self.original_width
    }

    /// Original decoded height.
    #[must_use]
    pub const fn original_height(&self) -> u32 {
        self.original_height
    }

    /// Horizontal content scale relative to the decoded image.
    #[must_use]
    pub const fn scale_x(&self) -> f32 {
        self.scale_x
    }

    /// Vertical content scale relative to the decoded image.
    #[must_use]
    pub const fn scale_y(&self) -> f32 {
        self.scale_y
    }
}

impl fmt::Debug for DetectorInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DetectorInput")
            .field("data", &"[REDACTED]")
            .field("tensor_width", &self.tensor_width)
            .field("tensor_height", &self.tensor_height)
            .field("content_width", &self.content_width)
            .field("content_height", &self.content_height)
            .field("pre_resize_width", &self.pre_resize_width)
            .field("pre_resize_height", &self.pre_resize_height)
            .field("original_width", &self.original_width)
            .field("original_height", &self.original_height)
            .field("scale_x", &self.scale_x)
            .field("scale_y", &self.scale_y)
            .finish()
    }
}

/// Applies the pinned `paddlex-ocr-3.7-max960` detector transform.
///
/// Images whose height plus width is below 64 are first black-padded at bottom/right to at least
/// 32 by 32. A side above 960 is reduced proportionally; the preliminary dimensions are truncated,
/// then rounded to the nearest multiple of 32 (Python ties-to-even semantics, minimum 32). The
/// complete image is bilinearly resized to that exact tensor shape. Pixels are converted from RGB8
/// to BGR, scaled by `1/255`, normalized using mean `[0.485, 0.456, 0.406]` and standard deviation
/// `[0.229, 0.224, 0.225]`, and emitted NCHW.
///
/// # Errors
/// Returns a sanitized internal error if checked shape arithmetic or allocation fails.
#[allow(clippy::cast_precision_loss)] // All admitted dimensions are at most 16,384.
pub fn preprocess_detector(image: &DecodedRaster) -> Result<DetectorInput, OcrError> {
    let original_width = image.width();
    let original_height = image.height();
    let is_tiny = original_width
        .checked_add(original_height)
        .ok_or_else(internal)?
        < 64;
    let pre_resize_width = if is_tiny {
        original_width.max(STRIDE)
    } else {
        original_width
    };
    let pre_resize_height = if is_tiny {
        original_height.max(STRIDE)
    } else {
        original_height
    };
    let (tensor_width, tensor_height) = detector_resize_size(pre_resize_width, pre_resize_height)?;

    let content = if is_tiny {
        let padded_len =
            usize::try_from(u64::from(pre_resize_width) * u64::from(pre_resize_height) * 3)
                .map_err(|_| internal())?;
        let mut padded = Vec::new();
        padded
            .try_reserve_exact(padded_len)
            .map_err(|_| internal())?;
        padded.resize(padded_len, 0);
        for y in 0..original_height {
            let source_start = usize::try_from(u64::from(y) * u64::from(original_width) * 3)
                .map_err(|_| internal())?;
            let source_end = source_start
                .checked_add(
                    usize::try_from(u64::from(original_width) * 3).map_err(|_| internal())?,
                )
                .ok_or_else(internal)?;
            let target_start = usize::try_from(u64::from(y) * u64::from(pre_resize_width) * 3)
                .map_err(|_| internal())?;
            let target_end = target_start
                .checked_add(source_end - source_start)
                .ok_or_else(internal)?;
            padded[target_start..target_end]
                .copy_from_slice(&image.rgb()[source_start..source_end]);
        }
        resize_inter_linear_rgb8(
            &padded,
            pre_resize_width,
            pre_resize_height,
            tensor_width,
            tensor_height,
        )?
    } else {
        resize_inter_linear_rgb8(
            image.rgb(),
            original_width,
            original_height,
            tensor_width,
            tensor_height,
        )?
    };

    let plane = usize::try_from(u64::from(tensor_width) * u64::from(tensor_height))
        .map_err(|_| internal())?;
    let tensor_len = plane.checked_mul(3).ok_or_else(internal)?;
    let mut data = Vec::new();
    data.try_reserve_exact(tensor_len).map_err(|_| internal())?;
    data.resize(tensor_len, 0.0);

    for (pixel_index, pixel) in content.chunks_exact(3).enumerate() {
        let [red, green, blue] = [pixel[0], pixel[1], pixel[2]];
        for (channel, value) in [blue, green, red].into_iter().enumerate() {
            data[channel * plane + pixel_index] = normalize(value, channel);
        }
    }

    Ok(DetectorInput {
        data,
        tensor_width,
        tensor_height,
        content_width: tensor_width,
        content_height: tensor_height,
        pre_resize_width,
        pre_resize_height,
        original_width,
        original_height,
        scale_x: tensor_width as f32 / pre_resize_width as f32,
        scale_y: tensor_height as f32 / pre_resize_height as f32,
    })
}

fn detector_resize_size(width: u32, height: u32) -> Result<(u32, u32), OcrError> {
    if width == 0 || height == 0 {
        return Err(internal());
    }
    let maximum = width.max(height);
    let (scaled_width, scaled_height) = if maximum > DETECTOR_MAX_SIDE {
        (
            floor_ratio(width, DETECTOR_MAX_SIDE, maximum)?,
            floor_ratio(height, DETECTOR_MAX_SIDE, maximum)?,
        )
    } else {
        (width, height)
    };
    Ok((
        nearest_stride_ties_even(scaled_width)?,
        nearest_stride_ties_even(scaled_height)?,
    ))
}

fn floor_ratio(value: u32, numerator: u32, denominator: u32) -> Result<u32, OcrError> {
    let scaled = u64::from(value)
        .checked_mul(u64::from(numerator))
        .ok_or_else(internal)?
        / u64::from(denominator);
    u32::try_from(scaled).map_err(|_| internal())
}

fn nearest_stride_ties_even(value: u32) -> Result<u32, OcrError> {
    let quotient = value / STRIDE;
    let remainder = value % STRIDE;
    let rounded_quotient = match remainder.cmp(&(STRIDE / 2)) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient.checked_add(1).ok_or_else(internal)?,
        std::cmp::Ordering::Equal if quotient % 2 == 0 => quotient,
        std::cmp::Ordering::Equal => quotient.checked_add(1).ok_or_else(internal)?,
    };
    rounded_quotient
        .checked_mul(STRIDE)
        .map(|rounded| rounded.max(STRIDE))
        .ok_or_else(internal)
}

fn normalize(value: u8, channel: usize) -> f32 {
    (f32::from(value) / 255.0 - MEAN_BGR[channel]) / STD_BGR[channel]
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use impossible_ocr_domain::OcrError;

    use crate::{DecodedRaster, preprocess_detector};

    fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Result<DecodedRaster, OcrError> {
        let pixels = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| OcrError::for_code(impossible_ocr_domain::OcrErrorCode::InvalidImage))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(pixels.saturating_mul(3))
            .map_err(|_| OcrError::for_code(impossible_ocr_domain::OcrErrorCode::InvalidImage))?;
        for _ in 0..pixels {
            bytes.extend_from_slice(&rgb);
        }
        DecodedRaster::from_rgb8(width, height, bytes)
    }

    #[test]
    fn stride_boundaries_preserve_content_below_the_max() -> Result<(), OcrError> {
        for (side, tensor) in [
            (31, 32),
            (32, 32),
            (63, 64),
            (64, 64),
            (959, 960),
            (960, 960),
            (961, 960),
        ] {
            let output = preprocess_detector(&solid(side, 1, [0, 0, 0])?)?;
            assert_eq!(output.tensor_width(), tensor, "side {side}");
            assert_eq!(output.content_width(), tensor, "content side {side}");
            assert_eq!(output.tensor_height(), 32);
        }
        Ok(())
    }

    #[test]
    fn bgr_planes_and_normalization_are_exact() -> Result<(), OcrError> {
        let output = preprocess_detector(&solid(1, 1, [255, 128, 0])?)?;
        let plane =
            usize::try_from(u64::from(output.tensor_width()) * u64::from(output.tensor_height()))
                .map_err(|_| OcrError::for_code(impossible_ocr_domain::OcrErrorCode::Internal))?;
        let expected = [
            (0.0 - 0.485) / 0.229,
            (128.0 / 255.0 - 0.456) / 0.224,
            (1.0 - 0.406) / 0.225,
        ];
        assert!((output.data()[0] - expected[0]).abs() < 1e-6);
        assert!((output.data()[plane] - expected[1]).abs() < 1e-6);
        assert!((output.data()[2 * plane] - expected[2]).abs() < 1e-6);
        assert!(!format!("{output:?}").contains("-2.117"));
        Ok(())
    }

    #[test]
    fn extreme_aspect_ratio_is_bounded_and_scale_is_reported() -> Result<(), OcrError> {
        let output = preprocess_detector(&solid(16_384, 1, [1, 2, 3])?)?;
        assert_eq!(output.content_width(), 960);
        assert_eq!(output.content_height(), 32);
        assert_eq!(output.tensor_width(), 960);
        assert_eq!(output.tensor_height(), 32);
        assert!(output.scale_x() < 1.0);
        assert_eq!(output.scale_y().to_bits(), 32.0_f32.to_bits());
        assert_eq!(output.pre_resize_height(), 1);
        Ok(())
    }

    #[test]
    fn nearest_stride_uses_python_ties_to_even() -> Result<(), OcrError> {
        let lower_half = preprocess_detector(&solid(48, 32, [0, 0, 0])?)?;
        let upper_half = preprocess_detector(&solid(80, 32, [0, 0, 0])?)?;
        assert_eq!(lower_half.tensor_width(), 64);
        assert_eq!(upper_half.tensor_width(), 64);
        Ok(())
    }

    #[test]
    fn tiny_images_are_black_padded_before_resize() -> Result<(), OcrError> {
        let output = preprocess_detector(&solid(1, 1, [255, 0, 0])?)?;
        assert_eq!((output.original_width(), output.original_height()), (1, 1));
        assert_eq!(
            (output.pre_resize_width(), output.pre_resize_height()),
            (32, 32)
        );
        assert_eq!((output.tensor_width(), output.tensor_height()), (32, 32));
        assert_eq!(output.scale_x().to_bits(), 1.0_f32.to_bits());
        assert_eq!(output.scale_y().to_bits(), 1.0_f32.to_bits());
        Ok(())
    }
}
