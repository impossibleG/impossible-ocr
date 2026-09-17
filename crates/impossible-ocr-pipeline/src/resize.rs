use impossible_ocr_domain::{OcrError, OcrErrorCode};

const CHANNELS: usize = 3;
const COEFFICIENT_BITS: u32 = 11;
const COEFFICIENT_SCALE: u32 = 1 << COEFFICIENT_BITS;
const OUTPUT_ROUNDING: u64 = 1 << (COEFFICIENT_BITS * 2 - 1);

#[derive(Clone, Copy)]
struct Coefficients {
    first: u32,
    second: u32,
    first_weight: u32,
    second_weight: u32,
}

/// Resizes interleaved RGB8 with `OpenCV INTER_LINEAR` pixel-center and fixed-point semantics.
pub(crate) fn resize_inter_linear_rgb8(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, OcrError> {
    if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
        return Err(internal());
    }
    let source_len = byte_len(source_width, source_height)?;
    let target_len = byte_len(target_width, target_height)?;
    if source.len() != source_len {
        return Err(internal());
    }
    if source_width == target_width && source_height == target_height {
        return Ok(source.to_vec());
    }

    let x_coefficients = coefficients(source_width, target_width)?;
    let y_coefficients = coefficients(source_height, target_height)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(target_len)
        .map_err(|_| internal())?;

    for y in &y_coefficients {
        for x in &x_coefficients {
            for channel in 0..CHANNELS {
                let top_left = sample(source, source_width, x.first, y.first, channel)?;
                let top_right = sample(source, source_width, x.second, y.first, channel)?;
                let bottom_left = sample(source, source_width, x.first, y.second, channel)?;
                let bottom_right = sample(source, source_width, x.second, y.second, channel)?;

                let top = u64::from(top_left) * u64::from(x.first_weight)
                    + u64::from(top_right) * u64::from(x.second_weight);
                let bottom = u64::from(bottom_left) * u64::from(x.first_weight)
                    + u64::from(bottom_right) * u64::from(x.second_weight);
                let accumulated =
                    top * u64::from(y.first_weight) + bottom * u64::from(y.second_weight);
                let value = (accumulated + OUTPUT_ROUNDING) >> (COEFFICIENT_BITS * 2);
                output.push(u8::try_from(value.min(u64::from(u8::MAX))).map_err(|_| internal())?);
            }
        }
    }
    Ok(output)
}

fn coefficients(source_size: u32, target_size: u32) -> Result<Vec<Coefficients>, OcrError> {
    let capacity = usize::try_from(target_size).map_err(|_| internal())?;
    let mut result = Vec::new();
    result.try_reserve_exact(capacity).map_err(|_| internal())?;
    let scale = f64::from(source_size) / f64::from(target_size);
    for target in 0..target_size {
        let position = (f64::from(target) + 0.5) * scale - 0.5;
        let first = position.floor();
        if first < 0.0 || source_size == 1 {
            result.push(edge_coefficients(0));
            continue;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = first as u32;
        if first >= source_size - 1 {
            result.push(edge_coefficients(source_size - 1));
            continue;
        }
        #[allow(clippy::cast_possible_truncation)] // OpenCV quantizes the fraction to f32.
        let fraction = (position - f64::from(first)) as f32;
        #[allow(clippy::cast_precision_loss)] // 2,048 is exactly representable as f32.
        let second_weight = cv_round_nonnegative(fraction * COEFFICIENT_SCALE as f32)?;
        #[allow(clippy::cast_precision_loss)] // 2,048 is exactly representable as f32.
        let first_weight = cv_round_nonnegative((1.0_f32 - fraction) * COEFFICIENT_SCALE as f32)?;
        result.push(Coefficients {
            first,
            second: first + 1,
            first_weight,
            second_weight,
        });
    }
    Ok(result)
}

fn edge_coefficients(index: u32) -> Coefficients {
    Coefficients {
        first: index,
        second: index,
        first_weight: COEFFICIENT_SCALE,
        second_weight: 0,
    }
}

fn cv_round_nonnegative(value: f32) -> Result<u32, OcrError> {
    if !value.is_finite() || value < 0.0 {
        return Err(internal());
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rounded = value.round_ties_even() as u32;
    Ok(rounded)
}

fn sample(
    source: &[u8],
    source_width: u32,
    x: u32,
    y: u32,
    channel: usize,
) -> Result<u8, OcrError> {
    let pixel = u64::from(y)
        .checked_mul(u64::from(source_width))
        .and_then(|row| row.checked_add(u64::from(x)))
        .ok_or_else(internal)?;
    let offset = usize::try_from(pixel)
        .map_err(|_| internal())?
        .checked_mul(CHANNELS)
        .and_then(|base| base.checked_add(channel))
        .ok_or_else(internal)?;
    source.get(offset).copied().ok_or_else(internal)
}

fn byte_len(width: u32, height: u32) -> Result<usize, OcrError> {
    let length = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(CHANNELS as u64))
        .ok_or_else(internal)?;
    usize::try_from(length).map_err(|_| internal())
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use super::resize_inter_linear_rgb8;

    fn gray_row(width: u32, multiplier: u8) -> Vec<u8> {
        (0..width)
            .flat_map(|x| {
                let value = u8::try_from(x).unwrap_or_default().wrapping_mul(multiplier);
                [value; 3]
            })
            .collect()
    }

    fn red(output: &[u8]) -> Vec<u8> {
        output.chunks_exact(3).map(|pixel| pixel[0]).collect()
    }

    #[test]
    fn downsample_matches_opencv_adversarial_wraparound_golden()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let output = resize_inter_linear_rgb8(&gray_row(80, 17), 80, 1, 64, 1)?;
        assert_eq!(
            red(&output),
            vec![
                2, 23, 45, 66, 87, 108, 130, 151, 172, 193, 215, 236, 225, 22, 44, 65, 86, 107,
                129, 150, 171, 192, 214, 235, 224, 21, 43, 64, 85, 106, 128, 149, 170, 191, 213,
                234, 223, 20, 42, 63, 84, 105, 127, 148, 169, 190, 212, 233, 222, 19, 41, 62, 83,
                104, 126, 147, 168, 189, 211, 232, 221, 18, 40, 61,
            ]
        );
        Ok(())
    }

    #[test]
    fn upsample_matches_opencv_adversarial_wraparound_golden()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let output = resize_inter_linear_rgb8(&gray_row(80, 17), 80, 1, 120, 1)?;
        assert_eq!(
            red(&output),
            vec![
                0, 9, 20, 31, 43, 54, 65, 77, 88, 99, 111, 122, 133, 145, 156, 167, 179, 190, 201,
                213, 224, 235, 247, 215, 56, 25, 36, 47, 59, 70, 81, 93, 104, 115, 127, 138, 149,
                161, 172, 183, 195, 206, 217, 229, 240, 251, 135, 18, 29, 41, 52, 63, 75, 86, 97,
                109, 120, 131, 143, 154, 165, 177, 188, 199, 211, 222, 233, 245, 213, 54, 23, 34,
                45, 57, 68, 79, 91, 102, 113, 125, 136, 147, 159, 170, 181, 193, 204, 215, 227,
                238, 249, 133, 16, 27, 39, 50, 61, 73, 84, 95, 107, 118, 129, 141, 152, 163, 175,
                186, 197, 209, 220, 231, 243, 211, 52, 21, 32, 43, 55, 63,
            ]
        );
        Ok(())
    }

    #[test]
    fn identity_upscale_and_two_axis_edges_are_stable()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let source = vec![0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 245];
        assert_eq!(resize_inter_linear_rgb8(&source, 2, 2, 2, 2)?, source);
        let up = resize_inter_linear_rgb8(&source, 2, 2, 3, 3)?;
        assert_eq!(&up[..3], &[0, 10, 20]);
        assert_eq!(&up[up.len() - 3..], &[255, 250, 245]);
        assert_eq!(&up[12..15], &[139, 145, 151]);
        Ok(())
    }
}
