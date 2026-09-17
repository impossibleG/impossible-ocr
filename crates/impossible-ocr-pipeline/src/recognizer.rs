use std::{cmp::Ordering, fmt};

use impossible_ocr_domain::{OcrError, OcrErrorCode};

use crate::{DecodedRaster, resize::resize_inter_linear_rgb8};

/// Fixed recognizer input height.
pub const RECOGNIZER_HEIGHT: u32 = 48;
/// Maximum dynamic recognizer batch width.
pub const RECOGNIZER_MAX_WIDTH: u32 = 3200;
const RECOGNIZER_BASE_WIDTH: u32 = 320;

/// One aspect-sorted, dynamically padded recognizer batch.
#[derive(Clone, PartialEq)]
pub struct RecognizerBatch {
    data: Vec<f32>,
    batch_width: u32,
    resized_widths: Vec<u32>,
    sorted_to_original: Vec<usize>,
}

impl RecognizerBatch {
    /// NCHW tensor data with shape `[batch_size, 3, 48, batch_width]`, BGR channels, and
    /// normalized content in `[-1, 1]`. Right padding is exactly zero.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Number of crops in the batch.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.sorted_to_original.len()
    }

    /// Shared dynamic tensor width.
    #[must_use]
    pub const fn batch_width(&self) -> u32 {
        self.batch_width
    }

    /// Content width for each item in sorted tensor order.
    #[must_use]
    pub fn resized_widths(&self) -> &[u32] {
        &self.resized_widths
    }

    /// Original crop index for each item in sorted tensor order.
    #[must_use]
    pub fn sorted_to_original(&self) -> &[usize] {
        &self.sorted_to_original
    }

    /// Restores per-item outputs from tensor order to caller order.
    ///
    /// # Errors
    /// Returns a sanitized internal error when the output cardinality differs from the batch.
    pub fn restore_order<T>(&self, sorted: Vec<T>) -> Result<Vec<T>, OcrError> {
        if sorted.len() != self.batch_size() {
            return Err(internal());
        }
        let mut indexed: Vec<_> = self
            .sorted_to_original
            .iter()
            .copied()
            .zip(sorted)
            .collect();
        indexed.sort_unstable_by_key(|(original, _)| *original);
        Ok(indexed.into_iter().map(|(_, value)| value).collect())
    }
}

impl fmt::Debug for RecognizerBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecognizerBatch")
            .field("data", &"[REDACTED]")
            .field("batch_width", &self.batch_width)
            .field("resized_widths", &self.resized_widths)
            .field("sorted_to_original", &self.sorted_to_original)
            .finish()
    }
}

/// Sorts crops stably by aspect ratio and constructs a dynamic recognizer batch.
///
/// Each crop preserves aspect ratio at height 48, is capped at width 3200, converted from RGB8 to
/// BGR, normalized as `value / 127.5 - 1`, and right-padded with exact zero. The shared batch width
/// is at least 320 and no larger than 3200.
///
/// # Errors
/// Returns a sanitized error for an empty batch, invalid shape, checked arithmetic failure, or
/// allocation failure.
pub fn preprocess_recognizer_batch(crops: &[DecodedRaster]) -> Result<RecognizerBatch, OcrError> {
    if crops.is_empty() {
        return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
    }
    let mut sorted_to_original: Vec<usize> = (0..crops.len()).collect();
    sorted_to_original
        .sort_by(|left, right| compare_aspect(&crops[*left], &crops[*right], *left, *right));

    let mut resized_widths = Vec::new();
    resized_widths
        .try_reserve_exact(crops.len())
        .map_err(|_| internal())?;
    for original in &sorted_to_original {
        resized_widths.push(recognizer_width(&crops[*original])?);
    }
    let batch_width = resized_widths
        .iter()
        .copied()
        .max()
        .ok_or_else(internal)?
        .clamp(RECOGNIZER_BASE_WIDTH, RECOGNIZER_MAX_WIDTH);
    let plane = usize::try_from(u64::from(RECOGNIZER_HEIGHT) * u64::from(batch_width))
        .map_err(|_| internal())?;
    let item_len = plane.checked_mul(3).ok_or_else(internal)?;
    let tensor_len = item_len.checked_mul(crops.len()).ok_or_else(internal)?;
    let mut data = Vec::new();
    data.try_reserve_exact(tensor_len).map_err(|_| internal())?;
    data.resize(tensor_len, 0.0);

    for (sorted_index, original) in sorted_to_original.iter().copied().enumerate() {
        let crop = &crops[original];
        let resized_width = resized_widths[sorted_index];
        let resized = resize_inter_linear_rgb8(
            crop.rgb(),
            crop.width(),
            crop.height(),
            resized_width,
            RECOGNIZER_HEIGHT,
        )?;
        let item_offset = sorted_index.checked_mul(item_len).ok_or_else(internal)?;
        let resized_width_usize = usize::try_from(resized_width).map_err(|_| internal())?;
        for (source_pixel_index, pixel) in resized.chunks_exact(3).enumerate() {
            let y = source_pixel_index / resized_width_usize;
            let x = source_pixel_index % resized_width_usize;
            let pixel_index = y
                .checked_mul(usize::try_from(batch_width).map_err(|_| internal())?)
                .and_then(|row| row.checked_add(x))
                .ok_or_else(internal)?;
            let [red, green, blue] = [pixel[0], pixel[1], pixel[2]];
            for (channel, value) in [blue, green, red].into_iter().enumerate() {
                data[item_offset + channel * plane + pixel_index] = normalize(value);
            }
        }
    }

    Ok(RecognizerBatch {
        data,
        batch_width,
        resized_widths,
        sorted_to_original,
    })
}

fn compare_aspect(
    left: &DecodedRaster,
    right: &DecodedRaster,
    left_index: usize,
    right_index: usize,
) -> Ordering {
    let left_cross = u64::from(left.width()) * u64::from(right.height());
    let right_cross = u64::from(right.width()) * u64::from(left.height());
    left_cross
        .cmp(&right_cross)
        .then_with(|| left_index.cmp(&right_index))
}

fn recognizer_width(crop: &DecodedRaster) -> Result<u32, OcrError> {
    let numerator = u64::from(crop.width())
        .checked_mul(u64::from(RECOGNIZER_HEIGHT))
        .ok_or_else(internal)?;
    let height = u64::from(crop.height());
    let rounded_up = numerator.checked_add(height - 1).ok_or_else(internal)? / height;
    u32::try_from(rounded_up)
        .map(|width| width.clamp(1, RECOGNIZER_MAX_WIDTH))
        .map_err(|_| internal())
}

fn normalize(value: u8) -> f32 {
    f32::from(value) / 127.5 - 1.0
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use impossible_ocr_domain::{OcrError, OcrErrorCode};

    use crate::{DecodedRaster, preprocess_recognizer_batch};

    fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Result<DecodedRaster, OcrError> {
        let pixels = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| OcrError::for_code(OcrErrorCode::InvalidImage))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(pixels.saturating_mul(3))
            .map_err(|_| OcrError::for_code(OcrErrorCode::InvalidImage))?;
        for _ in 0..pixels {
            bytes.extend_from_slice(&rgb);
        }
        DecodedRaster::from_rgb8(width, height, bytes)
    }

    #[test]
    fn batch_is_aspect_sorted_and_restores_original_order() -> Result<(), OcrError> {
        let crops = [
            solid(200, 50, [1, 2, 3])?,
            solid(10, 50, [4, 5, 6])?,
            solid(100, 50, [7, 8, 9])?,
        ];
        let batch = preprocess_recognizer_batch(&crops)?;
        assert_eq!(batch.sorted_to_original(), &[1, 2, 0]);
        assert_eq!(
            batch.restore_order(vec!["narrow", "middle", "wide"])?,
            vec!["wide", "narrow", "middle"]
        );
        assert_eq!(batch.batch_width(), 320);
        Ok(())
    }

    #[test]
    fn very_wide_crop_is_capped_at_3200() -> Result<(), OcrError> {
        let batch = preprocess_recognizer_batch(&[solid(16_384, 1, [0, 0, 0])?])?;
        assert_eq!(batch.resized_widths(), &[3200]);
        assert_eq!(batch.batch_width(), 3200);
        Ok(())
    }

    #[test]
    fn bgr_normalization_and_zero_padding_are_exact() -> Result<(), OcrError> {
        let batch = preprocess_recognizer_batch(&[solid(1, 48, [255, 128, 0])?])?;
        let plane = usize::try_from(u64::from(48_u32) * u64::from(batch.batch_width()))
            .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
        assert_eq!(batch.resized_widths(), &[1]);
        assert_eq!(batch.data()[0].to_bits(), (-1.0_f32).to_bits());
        assert!((batch.data()[plane] - (128.0 / 127.5 - 1.0)).abs() < 1e-6);
        assert_eq!(batch.data()[2 * plane].to_bits(), 1.0_f32.to_bits());
        assert_eq!(batch.data()[1].to_bits(), 0.0_f32.to_bits());
        assert!(!format!("{batch:?}").contains("-1.0"));
        Ok(())
    }

    #[test]
    fn recognizer_path_uses_opencv_pixel_centers_for_80_to_120_upscale() -> Result<(), OcrError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(80 * 32 * 3)
            .map_err(|_| OcrError::for_code(OcrErrorCode::Internal))?;
        for _ in 0..32 {
            for x in 0..80_u8 {
                let value = x.wrapping_mul(17);
                bytes.extend_from_slice(&[value; 3]);
            }
        }
        let crop = DecodedRaster::from_rgb8(80, 32, bytes)?;
        let batch = preprocess_recognizer_batch(&[crop])?;
        assert_eq!(batch.resized_widths(), &[120]);
        let expected_raw = [0_u8, 9, 20, 31, 43];
        for (actual, expected) in batch.data()[..expected_raw.len()].iter().zip(expected_raw) {
            assert_eq!(actual.to_bits(), super::normalize(expected).to_bits());
        }
        Ok(())
    }

    #[test]
    fn empty_batch_and_bad_restore_cardinality_fail() -> Result<(), OcrError> {
        assert_eq!(
            preprocess_recognizer_batch(&[]).map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::InvalidRequest)
        );
        let batch = preprocess_recognizer_batch(&[solid(1, 1, [0, 0, 0])?])?;
        assert!(batch.restore_order::<u8>(Vec::new()).is_err());
        Ok(())
    }
}
