use impossible_ocr_domain::{
    MAX_DECODED_BYTES, MAX_DIMENSION, MAX_PIXELS, OcrError, OcrErrorCode, Point, Polygon,
};

use crate::DecodedRaster;

const CHANNELS: usize = 3;
const INTER_BITS: i64 = 5;
const INTER_TAB_SIZE: i64 = 1 << INTER_BITS;
const INTER_TAB_SIZE_F64: f64 = 32.0;
const COEFFICIENT_BITS: u32 = 15;
const COEFFICIENT_SCALE: i32 = 1 << COEFFICIENT_BITS;
const COEFFICIENT_SCALE_F32: f32 = 32_768.0;
const TALL_CROP_RATIO: f64 = 1.5;
const CUBIC_A: f32 = -0.75;

/// Extracts a quadrilateral text crop with PP-OCR/OpenCV-compatible sizing and sampling.
///
/// Points must be ordered top-left, top-right, bottom-right, bottom-left. Output width is the
/// truncated maximum of the two horizontal edge lengths and output height is the truncated maximum
/// of the two vertical edge lengths, matching PP-OCR's `get_rotate_crop_image`. Perspective
/// sampling uses inverse mapping, cubic interpolation, and replicated borders. Crops whose height
/// is at least 1.5 times their width are rotated exactly 90 degrees counter-clockwise.
///
/// # Errors
/// Returns `invalid_image` for non-finite, out-of-bounds, degenerate, self-intersecting, or
/// excessively large geometry.
pub fn perspective_crop(
    source: &DecodedRaster,
    polygon: Polygon,
) -> Result<DecodedRaster, OcrError> {
    validate_polygon(source, polygon)?;
    let [top_left, top_right, bottom_right, bottom_left] = polygon.points;
    let width = edge_length(top_left, top_right)
        .max(edge_length(bottom_left, bottom_right))
        .floor();
    let height = edge_length(top_left, bottom_left)
        .max(edge_length(top_right, bottom_right))
        .floor();
    let width = checked_dimension(width)?;
    let height = checked_dimension(height)?;
    checked_output_len(width, height)?;

    let transform = ProjectiveMap::from_rectangle(
        f64::from(width),
        f64::from(height),
        [top_left, top_right, bottom_right, bottom_left],
    )?;
    let rgb = warp_cubic_replicate(source, width, height, transform)?;
    let crop = DecodedRaster::from_rgb8(width, height, rgb)?;
    if f64::from(height) / f64::from(width) >= TALL_CROP_RATIO {
        rotate_90_counter_clockwise(&crop)
    } else {
        Ok(crop)
    }
}

#[derive(Clone, Copy)]
struct ProjectiveMap {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    width: f64,
    height: f64,
}

impl ProjectiveMap {
    #[allow(clippy::similar_names)] // Conventional projective-transform delta names.
    fn from_rectangle(width: f64, height: f64, points: [Point; 4]) -> Result<Self, OcrError> {
        let [p0, p1, p2, p3] = points.map(|point| (f64::from(point.x), f64::from(point.y)));
        let dx1 = p1.0 - p2.0;
        let dx2 = p3.0 - p2.0;
        let dx3 = p0.0 - p1.0 + p2.0 - p3.0;
        let dy1 = p1.1 - p2.1;
        let dy2 = p3.1 - p2.1;
        let dy3 = p0.1 - p1.1 + p2.1 - p3.1;
        let (g, h) = if dx3.abs() <= f64::EPSILON && dy3.abs() <= f64::EPSILON {
            (0.0, 0.0)
        } else {
            let determinant = dx1.mul_add(dy2, -(dx2 * dy1));
            if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
                return Err(invalid_image());
            }
            (
                dx3.mul_add(dy2, -(dx2 * dy3)) / determinant,
                dx1.mul_add(dy3, -(dx3 * dy1)) / determinant,
            )
        };
        let map = Self {
            a: p1.0 - p0.0 + g * p1.0,
            b: p3.0 - p0.0 + h * p3.0,
            c: p0.0,
            d: p1.1 - p0.1 + g * p1.1,
            e: p3.1 - p0.1 + h * p3.1,
            f: p0.1,
            g,
            h,
            width,
            height,
        };
        if [map.a, map.b, map.c, map.d, map.e, map.f, map.g, map.h]
            .into_iter()
            .all(f64::is_finite)
        {
            Ok(map)
        } else {
            Err(invalid_image())
        }
    }

    fn map(self, x: u32, y: u32) -> Result<(f64, f64), OcrError> {
        let horizontal = f64::from(x) / self.width;
        let vertical = f64::from(y) / self.height;
        let denominator = self.g.mul_add(horizontal, self.h * vertical) + 1.0;
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return Err(invalid_image());
        }
        let source_x = (self.a.mul_add(horizontal, self.b * vertical) + self.c) / denominator;
        let source_y = (self.d.mul_add(horizontal, self.e * vertical) + self.f) / denominator;
        if source_x.is_finite() && source_y.is_finite() {
            Ok((source_x, source_y))
        } else {
            Err(invalid_image())
        }
    }
}

fn validate_polygon(source: &DecodedRaster, polygon: Polygon) -> Result<(), OcrError> {
    let max_x = f64::from(source.width().saturating_sub(1));
    let max_y = f64::from(source.height().saturating_sub(1));
    let points = polygon.points;
    if !points.iter().all(|point| {
        let x = f64::from(point.x);
        let y = f64::from(point.y);
        x.is_finite() && y.is_finite() && (0.0..=max_x).contains(&x) && (0.0..=max_y).contains(&y)
    }) {
        return Err(invalid_image());
    }
    let mut direction = 0_i8;
    for index in 0..4 {
        let first = points[index];
        let second = points[(index + 1) % 4];
        let third = points[(index + 2) % 4];
        let cross = f64::from(second.x - first.x).mul_add(
            f64::from(third.y - second.y),
            -f64::from(second.y - first.y) * f64::from(third.x - second.x),
        );
        if !cross.is_finite() || cross.abs() <= f64::EPSILON {
            return Err(invalid_image());
        }
        let current = if cross.is_sign_positive() { 1 } else { -1 };
        if direction != 0 && current != direction {
            return Err(invalid_image());
        }
        direction = current;
    }
    if direction != 1 {
        return Err(invalid_image());
    }
    Ok(())
}

fn edge_length(first: Point, second: Point) -> f64 {
    f64::from(second.x - first.x).hypot(f64::from(second.y - first.y))
}

fn checked_dimension(value: f64) -> Result<u32, OcrError> {
    if !value.is_finite() || value < 1.0 || value > f64::from(MAX_DIMENSION) {
        return Err(invalid_image());
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(value as u32)
}

fn checked_output_len(width: u32, height: u32) -> Result<usize, OcrError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .filter(|pixels| *pixels <= MAX_PIXELS)
        .ok_or_else(invalid_image)?;
    let bytes = pixels
        .checked_mul(CHANNELS as u64)
        .filter(|bytes| *bytes <= MAX_DECODED_BYTES)
        .ok_or_else(invalid_image)?;
    usize::try_from(bytes).map_err(|_| invalid_image())
}

fn warp_cubic_replicate(
    source: &DecodedRaster,
    width: u32,
    height: u32,
    transform: ProjectiveMap,
) -> Result<Vec<u8>, OcrError> {
    let capacity = checked_output_len(width, height)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_image())?;
    let interpolation_tables = cubic_tables()?;
    for y in 0..height {
        for x in 0..width {
            let (source_x, source_y) = transform.map(x, y)?;
            let (base_x, phase_x) = quantize_coordinate(source_x)?;
            let (base_y, phase_y) = quantize_coordinate(source_y)?;
            let table_index = phase_y
                .checked_mul(usize::try_from(INTER_TAB_SIZE).map_err(|_| invalid_image())?)
                .and_then(|row| row.checked_add(phase_x))
                .ok_or_else(invalid_image)?;
            let weights = interpolation_tables
                .get(table_index)
                .ok_or_else(invalid_image)?;
            for channel in 0..CHANNELS {
                let mut sum = 0_i64;
                for row in 0..4_i64 {
                    for column in 0..4_i64 {
                        let sample = sample_replicated(
                            source,
                            base_x + column - 1,
                            base_y + row - 1,
                            channel,
                        )?;
                        let index =
                            usize::try_from(row * 4 + column).map_err(|_| invalid_image())?;
                        sum = sum
                            .checked_add(i64::from(sample) * i64::from(weights[index]))
                            .ok_or_else(invalid_image)?;
                    }
                }
                let rounded = (sum + (1_i64 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS;
                output.push(u8::try_from(rounded.clamp(0, 255)).map_err(|_| invalid_image())?);
            }
        }
    }
    Ok(output)
}

fn cubic_tables() -> Result<Vec<[i32; 16]>, OcrError> {
    let side = usize::try_from(INTER_TAB_SIZE).map_err(|_| invalid_image())?;
    let capacity = side.checked_mul(side).ok_or_else(invalid_image)?;
    let mut tables = Vec::new();
    tables
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_image())?;
    for phase_y in 0..side {
        for phase_x in 0..side {
            tables.push(cubic_table(phase_x, phase_y));
        }
    }
    Ok(tables)
}

fn quantize_coordinate(value: f64) -> Result<(i64, usize), OcrError> {
    let scaled = value * INTER_TAB_SIZE_F64;
    if !scaled.is_finite() || scaled.abs() > f64::from(u32::MAX) {
        return Err(invalid_image());
    }
    #[allow(clippy::cast_possible_truncation)]
    let fixed = scaled.round_ties_even() as i64;
    let base = fixed.div_euclid(INTER_TAB_SIZE);
    let phase = usize::try_from(fixed.rem_euclid(INTER_TAB_SIZE)).map_err(|_| invalid_image())?;
    Ok((base, phase))
}

fn cubic_table(phase_x: usize, phase_y: usize) -> [i32; 16] {
    let x = cubic_coefficients(phase_x);
    let y = cubic_coefficients(phase_y);
    let mut table = [0_i32; 16];
    let mut sum = 0_i32;
    for row in 0..4 {
        for column in 0..4 {
            let value = (x[column] * y[row] * COEFFICIENT_SCALE_F32).round_ties_even();
            #[allow(clippy::cast_possible_truncation)]
            let value = value as i32;
            table[row * 4 + column] = value;
            sum += value;
        }
    }
    let difference = COEFFICIENT_SCALE - sum;
    if difference != 0 {
        let central = [5_usize, 6, 9, 10];
        let index = if difference > 0 {
            central
                .into_iter()
                .max_by_key(|index| table[*index])
                .unwrap_or(5)
        } else {
            central
                .into_iter()
                .min_by_key(|index| table[*index])
                .unwrap_or(5)
        };
        table[index] += difference;
    }
    table
}

fn cubic_coefficients(phase: usize) -> [f32; 4] {
    #[allow(clippy::cast_precision_loss)]
    let x = phase as f32 / INTER_TAB_SIZE as f32;
    let one_minus_x = 1.0 - x;
    let first = ((CUBIC_A * (x + 1.0) - 5.0 * CUBIC_A) * (x + 1.0) + 8.0 * CUBIC_A) * (x + 1.0)
        - 4.0 * CUBIC_A;
    let second = ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0;
    let third = ((CUBIC_A + 2.0) * one_minus_x - (CUBIC_A + 3.0)) * one_minus_x * one_minus_x + 1.0;
    [first, second, third, 1.0 - first - second - third]
}

fn sample_replicated(
    source: &DecodedRaster,
    x: i64,
    y: i64,
    channel: usize,
) -> Result<u8, OcrError> {
    let max_x = i64::from(source.width()) - 1;
    let max_y = i64::from(source.height()) - 1;
    let x = u64::try_from(x.clamp(0, max_x)).map_err(|_| invalid_image())?;
    let y = u64::try_from(y.clamp(0, max_y)).map_err(|_| invalid_image())?;
    let offset = y
        .checked_mul(u64::from(source.width()))
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(CHANNELS as u64))
        .and_then(|base| base.checked_add(channel as u64))
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or_else(invalid_image)?;
    source.rgb().get(offset).copied().ok_or_else(invalid_image)
}

fn rotate_90_counter_clockwise(source: &DecodedRaster) -> Result<DecodedRaster, OcrError> {
    let width = source.height();
    let height = source.width();
    let capacity = checked_output_len(width, height)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| invalid_image())?;
    for y in 0..height {
        for x in 0..width {
            let source_x = source.width() - 1 - y;
            let source_y = x;
            let pixel = u64::from(source_y)
                .checked_mul(u64::from(source.width()))
                .and_then(|row| row.checked_add(u64::from(source_x)))
                .and_then(|pixel| pixel.checked_mul(CHANNELS as u64))
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(invalid_image)?;
            let end = pixel.checked_add(CHANNELS).ok_or_else(invalid_image)?;
            output.extend_from_slice(source.rgb().get(pixel..end).ok_or_else(invalid_image)?);
        }
    }
    DecodedRaster::from_rgb8(width, height, output)
}

const fn invalid_image() -> OcrError {
    OcrError::for_code(OcrErrorCode::InvalidImage)
}

#[cfg(test)]
mod tests {
    use impossible_ocr_domain::{OcrErrorCode, Point, Polygon};

    use super::perspective_crop;
    use crate::DecodedRaster;

    fn point(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    fn gradient(width: u32, height: u32) -> Result<DecodedRaster, impossible_ocr_domain::OcrError> {
        let rgb = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let red = u8::try_from(x * 17 + y * 3).unwrap_or_default();
                    let green = u8::try_from(x * 5 + y * 19).unwrap_or_default();
                    let blue = u8::try_from(x * 11 + y * 7).unwrap_or_default();
                    [red, green, blue]
                })
            })
            .collect();
        DecodedRaster::from_rgb8(width, height, rgb)
    }

    #[test]
    fn axis_aligned_identity_uses_pp_exclusive_corner_sizing()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let source = gradient(5, 4)?;
        let crop = perspective_crop(
            &source,
            Polygon {
                points: [
                    point(0.0, 0.0),
                    point(4.0, 0.0),
                    point(4.0, 3.0),
                    point(0.0, 3.0),
                ],
            },
        )?;
        assert_eq!((crop.width(), crop.height()), (4, 3));
        assert_eq!(&crop.rgb()[..3], &[0, 0, 0]);
        assert_eq!(&crop.rgb()[crop.rgb().len() - 3..], &[57, 53, 47]);
        assert!(!format!("{crop:?}").contains("57, 53, 47"));
        Ok(())
    }

    #[test]
    fn tall_crop_rotates_exactly_counter_clockwise() -> Result<(), impossible_ocr_domain::OcrError>
    {
        let source = gradient(3, 6)?;
        let crop = perspective_crop(
            &source,
            Polygon {
                points: [
                    point(0.0, 0.0),
                    point(2.0, 0.0),
                    point(2.0, 5.0),
                    point(0.0, 5.0),
                ],
            },
        )?;
        assert_eq!((crop.width(), crop.height()), (5, 2));
        assert_eq!(&crop.rgb()[..3], &[17, 5, 11]);
        assert_eq!(&crop.rgb()[crop.rgb().len() - 3..], &[12, 76, 28]);
        Ok(())
    }

    #[test]
    fn perspective_cubic_replicate_matches_opencv_golden()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let source = gradient(8, 7)?;
        let crop = perspective_crop(
            &source,
            Polygon {
                points: [
                    point(0.2, 0.4),
                    point(6.8, 1.1),
                    point(6.1, 5.9),
                    point(0.7, 5.2),
                ],
            },
        )?;
        assert_eq!((crop.width(), crop.height()), (6, 4));
        assert_eq!(
            crop.rgb(),
            &[
                3, 7, 4, 24, 14, 17, 42, 22, 30, 61, 30, 43, 80, 38, 56, 98, 46, 69, 10, 34, 15,
                29, 43, 29, 47, 51, 41, 65, 58, 53, 83, 66, 65, 101, 73, 78, 16, 60, 25, 34, 68,
                38, 51, 75, 50, 69, 82, 62, 86, 89, 73, 103, 95, 85, 21, 82, 35, 38, 90, 47, 55,
                96, 58, 71, 103, 69, 88, 109, 80, 105, 116, 92,
            ]
        );
        Ok(())
    }

    #[test]
    fn invalid_geometry_fails_closed() -> Result<(), impossible_ocr_domain::OcrError> {
        let source = gradient(8, 7)?;
        for points in [
            [
                point(0.0, 0.0),
                point(7.0, 6.0),
                point(7.0, 0.0),
                point(0.0, 6.0),
            ],
            [
                point(0.0, 0.0),
                point(9.0, 0.0),
                point(7.0, 6.0),
                point(0.0, 6.0),
            ],
            [
                point(f32::NAN, 0.0),
                point(7.0, 0.0),
                point(7.0, 6.0),
                point(0.0, 6.0),
            ],
            [
                point(1.0, 1.0),
                point(1.0, 1.0),
                point(1.0, 1.0),
                point(1.0, 1.0),
            ],
        ] {
            assert_eq!(
                perspective_crop(&source, Polygon { points })
                    .map_err(impossible_ocr_domain::OcrError::code),
                Err(OcrErrorCode::InvalidImage)
            );
        }
        Ok(())
    }
}
