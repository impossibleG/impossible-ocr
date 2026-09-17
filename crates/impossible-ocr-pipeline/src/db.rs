//! Pure-Rust post-processing for the pinned PP-OCRv5 DB detector profile.

use std::fmt;

use impossible_ocr_domain::{Confidence, OcrError, OcrErrorCode, Point, Polygon};
use impossible_server_core::RequestContext;

use crate::{
    DetectorInput,
    geometry::{GeometryPoint, OrientedRect, canonical_quad, minimum_area_rect, unclip_rect},
};

/// Strict foreground threshold from the admitted detector configuration.
pub const DB_THRESHOLD: f32 = 0.3;
/// Inclusive fast-score threshold from the admitted detector configuration.
pub const DB_BOX_THRESHOLD: f32 = 0.6;
/// Maximum number of contours considered before geometric filtering.
pub const DB_MAX_CANDIDATES: usize = 1_000;
/// Pinned area/perimeter expansion multiplier.
pub const DB_UNCLIP_RATIO: f64 = 1.5;
const DB_MIN_SIDE: f64 = 3.0;
const DB_MIN_UNCLIPPED_SIDE: f64 = 5.0;

/// Validated singleton detector probability map. Probability data is redacted from `Debug`.
#[derive(Clone, PartialEq)]
pub struct DetectorMap {
    probabilities: Vec<f32>,
    width: u32,
    height: u32,
}

impl DetectorMap {
    /// Constructs a finite `[1, 1, height, width]` probability map.
    ///
    /// # Errors
    /// Returns a sanitized internal error when dimensions, length, or probabilities violate the
    /// admitted detector output contract.
    pub fn new(probabilities: Vec<f32>, width: u32, height: u32) -> Result<Self, OcrError> {
        let expected = map_len(width, height)?;
        if probabilities.len() != expected
            || probabilities
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(internal());
        }
        Ok(Self {
            probabilities,
            width,
            height,
        })
    }

    /// Probability-map width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Probability-map height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Finite row-major probabilities.
    #[must_use]
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
}

impl fmt::Debug for DetectorMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DetectorMap")
            .field("probabilities", &"[REDACTED]")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

/// One detector region projected into the decoded image coordinate space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedTextRegion {
    polygon: Polygon,
    detector_score: Confidence,
}

impl DetectedTextRegion {
    /// Canonical top-left, top-right, bottom-right, bottom-left image-space polygon.
    #[must_use]
    pub const fn polygon(self) -> Polygon {
        self.polygon
    }

    /// Mean detector probability inside the pre-unclip minimum-area rectangle.
    #[must_use]
    pub const fn detector_score(self) -> Confidence {
        self.detector_score
    }
}

/// Pinned PP-OCRv5 DB `quad`/`fast` postprocessor.
#[derive(Debug, Clone, Copy, Default)]
pub struct DbPostProcessor;

impl DbPostProcessor {
    /// Converts one validated detector map into bounded image-space text regions.
    ///
    /// Foreground uses the strict `probability > 0.3` rule. Components are discovered with
    /// deterministic raster-order, eight-neighbor connectivity and capped at 1,000 candidates
    /// before filtering. Fast scores are means over an integer-filled minimum-area rectangle.
    ///
    /// # Errors
    /// Returns a sanitized internal error for an invalid model-output shape or checked allocation
    /// failure. Cancellation and deadline errors retain their stable public categories.
    pub fn process(
        self,
        map: &DetectorMap,
        input: &DetectorInput,
        context: &RequestContext,
    ) -> Result<Vec<DetectedTextRegion>, OcrError> {
        check_context(context)?;
        if map.width != input.tensor_width() || map.height != input.tensor_height() {
            return Err(internal());
        }

        let map_length = map_len(map.width, map.height)?;
        let mut foreground = Vec::new();
        foreground
            .try_reserve_exact(map_length)
            .map_err(|_| internal())?;
        foreground.extend(
            map.probabilities
                .iter()
                .map(|probability| u8::from(*probability > DB_THRESHOLD)),
        );

        let contours = extract_contours(
            &foreground,
            usize::try_from(map.width).map_err(|_| internal())?,
            usize::try_from(map.height).map_err(|_| internal())?,
            context,
        )?;
        let mut regions = Vec::new();
        regions
            .try_reserve_exact(contours.len())
            .map_err(|_| internal())?;

        for (index, contour) in contours.iter().enumerate() {
            if index % 32 == 0 {
                check_context(context)?;
            }
            let Ok(rectangle) = minimum_area_rect(contour) else {
                continue;
            };
            if rectangle.short_side() < DB_MIN_SIDE {
                continue;
            }
            let Some(score) = fast_score(map, rectangle)? else {
                continue;
            };
            if score < DB_BOX_THRESHOLD {
                continue;
            }
            let Ok(expanded) = unclip_rect(rectangle, DB_UNCLIP_RATIO) else {
                continue;
            };
            if expanded.short_side() < DB_MIN_UNCLIPPED_SIDE {
                continue;
            }
            let Some(polygon) = project_to_image(expanded, input)? else {
                continue;
            };
            regions.push(DetectedTextRegion {
                polygon,
                detector_score: Confidence::new(score).map_err(|_| internal())?,
            });
        }
        check_context(context)?;
        Ok(regions)
    }
}

fn extract_contours(
    foreground: &[u8],
    width: usize,
    height: usize,
    context: &RequestContext,
) -> Result<Vec<Vec<GeometryPoint>>, OcrError> {
    let length = width.checked_mul(height).ok_or_else(internal)?;
    if width == 0 || height == 0 || foreground.len() != length {
        return Err(internal());
    }
    let mut visited = Vec::new();
    visited.try_reserve_exact(length).map_err(|_| internal())?;
    visited.resize(length, 0_u8);
    let mut stack = Vec::new();
    stack.try_reserve_exact(length).map_err(|_| internal())?;
    let mut boundary = Vec::new();
    boundary.try_reserve_exact(length).map_err(|_| internal())?;
    let mut contours = Vec::new();
    contours
        .try_reserve_exact(DB_MAX_CANDIDATES.min(length))
        .map_err(|_| internal())?;

    'scan: for y in 0..height {
        check_context(context)?;
        for x in 0..width {
            let start = y
                .checked_mul(width)
                .and_then(|row| row.checked_add(x))
                .ok_or_else(internal)?;
            if foreground[start] == 0 || visited[start] != 0 {
                continue;
            }
            if contours.len() == DB_MAX_CANDIDATES {
                break 'scan;
            }
            stack.clear();
            boundary.clear();
            stack.push(start);
            visited[start] = 1;
            let mut examined = 0_usize;

            while let Some(pixel) = stack.pop() {
                examined = examined.checked_add(1).ok_or_else(internal)?;
                if examined % 1_024 == 0 {
                    check_context(context)?;
                }
                let pixel_y = pixel / width;
                let pixel_x = pixel % width;
                let mut is_boundary = false;
                for delta_y in -1_isize..=1 {
                    for delta_x in -1_isize..=1 {
                        if delta_x == 0 && delta_y == 0 {
                            continue;
                        }
                        let Some(neighbor_x) = pixel_x.checked_add_signed(delta_x) else {
                            is_boundary = true;
                            continue;
                        };
                        let Some(neighbor_y) = pixel_y.checked_add_signed(delta_y) else {
                            is_boundary = true;
                            continue;
                        };
                        if neighbor_x >= width || neighbor_y >= height {
                            is_boundary = true;
                            continue;
                        }
                        let neighbor = neighbor_y
                            .checked_mul(width)
                            .and_then(|row| row.checked_add(neighbor_x))
                            .ok_or_else(internal)?;
                        if foreground[neighbor] == 0 {
                            is_boundary = true;
                        } else if visited[neighbor] == 0 {
                            visited[neighbor] = 1;
                            stack.push(neighbor);
                        }
                    }
                }
                if is_boundary {
                    boundary.push(GeometryPoint::new(
                        usize_to_f64(pixel_x)?,
                        usize_to_f64(pixel_y)?,
                    ));
                }
            }
            let mut contour = Vec::new();
            contour
                .try_reserve_exact(boundary.len())
                .map_err(|_| internal())?;
            contour.extend_from_slice(&boundary);
            contours.push(contour);
        }
    }
    Ok(contours)
}

fn fast_score(map: &DetectorMap, rectangle: OrientedRect) -> Result<Option<f32>, OcrError> {
    let corners = rectangle.corners();
    let min_x = corners
        .iter()
        .map(|point| point.x())
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0);
    let max_x = corners
        .iter()
        .map(|point| point.x())
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .min(f64::from(map.width - 1));
    let min_y = corners
        .iter()
        .map(|point| point.y())
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0);
    let max_y = corners
        .iter()
        .map(|point| point.y())
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .min(f64::from(map.height - 1));
    let Some(x_start) = nonnegative_f64_to_usize(min_x) else {
        return Ok(None);
    };
    let Some(x_end) = nonnegative_f64_to_usize(max_x) else {
        return Ok(None);
    };
    let Some(y_start) = nonnegative_f64_to_usize(min_y) else {
        return Ok(None);
    };
    let Some(y_end) = nonnegative_f64_to_usize(max_y) else {
        return Ok(None);
    };
    if x_start > x_end || y_start > y_end {
        return Ok(None);
    }
    let mut integer_corners = [(0_i32, 0_i32); 4];
    for (target, source) in integer_corners.iter_mut().zip(corners) {
        let Some(x) = finite_f64_to_i32(source.x().trunc()) else {
            return Ok(None);
        };
        let Some(y) = finite_f64_to_i32(source.y().trunc()) else {
            return Ok(None);
        };
        *target = (x, y);
    }
    let width = usize::try_from(map.width).map_err(|_| internal())?;
    let spans = filled_quad_spans(&integer_corners, x_start, x_end, y_start, y_end)?;
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for (row, span) in spans.into_iter().enumerate() {
        let Some((left, right)) = span else {
            continue;
        };
        let y = y_start.checked_add(row).ok_or_else(internal)?;
        for x in left..=right {
            let offset = y
                .checked_mul(width)
                .and_then(|row| row.checked_add(x))
                .ok_or_else(internal)?;
            sum += f64::from(*map.probabilities.get(offset).ok_or_else(internal)?);
            count = count.checked_add(1).ok_or_else(internal)?;
        }
    }
    if count == 0 {
        return Ok(None);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let score = (sum / count as f64) as f32;
    if score.is_finite() && (0.0..=1.0).contains(&score) {
        Ok(Some(score))
    } else {
        Err(internal())
    }
}

fn filled_quad_spans(
    corners: &[(i32, i32); 4],
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
) -> Result<Vec<Option<(usize, usize)>>, OcrError> {
    let rows = y_end
        .checked_sub(y_start)
        .and_then(|difference| difference.checked_add(1))
        .ok_or_else(internal)?;
    let mut spans: Vec<Option<(usize, usize)>> = Vec::new();
    spans.try_reserve_exact(rows).map_err(|_| internal())?;
    spans.resize(rows, None);
    for (start, end) in corners
        .iter()
        .copied()
        .zip(corners.iter().copied().cycle().skip(1))
        .take(corners.len())
    {
        rasterize_line(start, end, |x, y| {
            let Ok(x) = usize::try_from(x) else {
                return;
            };
            let Ok(y) = usize::try_from(y) else {
                return;
            };
            if !(x_start..=x_end).contains(&x) || !(y_start..=y_end).contains(&y) {
                return;
            }
            let row = y - y_start;
            spans[row] =
                Some(spans[row].map_or((x, x), |(left, right)| (left.min(x), right.max(x))));
        })?;
    }
    Ok(spans)
}

fn rasterize_line(
    start: (i32, i32),
    end: (i32, i32),
    mut visit: impl FnMut(i64, i64),
) -> Result<(), OcrError> {
    let mut x = i64::from(start.0);
    let mut y = i64::from(start.1);
    let target_x = i64::from(end.0);
    let target_y = i64::from(end.1);
    let delta_x = (target_x - x).abs();
    let step_x = if x < target_x { 1 } else { -1 };
    let delta_y = -(target_y - y).abs();
    let step_y = if y < target_y { 1 } else { -1 };
    let mut error = delta_x.checked_add(delta_y).ok_or_else(internal)?;
    let maximum_steps = delta_x.max(-delta_y).checked_add(1).ok_or_else(internal)?;
    for _ in 0..maximum_steps {
        visit(x, y);
        if x == target_x && y == target_y {
            return Ok(());
        }
        let twice_error = error.checked_mul(2).ok_or_else(internal)?;
        if twice_error >= delta_y {
            error = error.checked_add(delta_y).ok_or_else(internal)?;
            x = x.checked_add(step_x).ok_or_else(internal)?;
        }
        if twice_error <= delta_x {
            error = error.checked_add(delta_x).ok_or_else(internal)?;
            y = y.checked_add(step_y).ok_or_else(internal)?;
        }
    }
    Err(internal())
}

fn project_to_image(
    rectangle: OrientedRect,
    input: &DetectorInput,
) -> Result<Option<Polygon>, OcrError> {
    let mut projected = [GeometryPoint::new(0.0, 0.0); 4];
    for (target, source) in projected.iter_mut().zip(rectangle.corners()) {
        let max_x = f64::from(input.original_width().checked_sub(1).ok_or_else(internal)?);
        let max_y = f64::from(
            input
                .original_height()
                .checked_sub(1)
                .ok_or_else(internal)?,
        );
        let x = round_ties_even(source.x() / f64::from(input.scale_x()))?.clamp(0.0, max_x);
        let y = round_ties_even(source.y() / f64::from(input.scale_y()))?.clamp(0.0, max_y);
        *target = GeometryPoint::new(x, y);
    }
    let Ok(projected) = canonical_quad(projected) else {
        return Ok(None);
    };
    let mut points = [Point { x: 0.0, y: 0.0 }; 4];
    for (target, source) in points.iter_mut().zip(projected) {
        #[allow(clippy::cast_possible_truncation)]
        let x = source.x() as f32;
        #[allow(clippy::cast_possible_truncation)]
        let y = source.y() as f32;
        *target = Point { x, y };
    }
    Ok(Some(Polygon { points }))
}

fn round_ties_even(value: f64) -> Result<f64, OcrError> {
    if !value.is_finite() {
        return Err(internal());
    }
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        return Ok(floor);
    }
    if fraction > 0.5 {
        return Ok(floor + 1.0);
    }
    let Some(integer) = finite_f64_to_i32(floor) else {
        return Err(internal());
    };
    if integer % 2 == 0 {
        Ok(floor)
    } else {
        Ok(floor + 1.0)
    }
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

fn map_len(width: u32, height: u32) -> Result<usize, OcrError> {
    if width == 0 || height == 0 {
        return Err(internal());
    }
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(internal)
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> Result<f64, OcrError> {
    let value = u32::try_from(value).map_err(|_| internal())?;
    Ok(f64::from(value))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn nonnegative_f64_to_usize(value: f64) -> Option<usize> {
    (value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX)).then_some(value as usize)
}

#[allow(clippy::cast_possible_truncation)]
fn finite_f64_to_i32(value: f64) -> Option<i32> {
    (value.is_finite() && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX))
        .then_some(value as i32)
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use impossible_ocr_domain::OcrErrorCode;
    use impossible_server_core::{CancellationToken, RequestContext, RequestIdSource};

    use crate::{DecodedRaster, perspective_crop, preprocess_detector};

    use super::{
        DB_BOX_THRESHOLD, DB_MAX_CANDIDATES, DB_THRESHOLD, DbPostProcessor, DetectorMap,
        extract_contours, filled_quad_spans, round_ties_even,
    };

    fn context(timeout: Option<Duration>) -> Result<RequestContext, Box<dyn std::error::Error>> {
        Ok(RequestContext::new(
            RequestIdSource::default().next()?,
            CancellationToken::new(),
            timeout,
        )?)
    }

    fn input(width: u32, height: u32) -> Result<crate::DetectorInput, Box<dyn std::error::Error>> {
        let length = usize::try_from(u64::from(width) * u64::from(height) * 3)?;
        Ok(preprocess_detector(&DecodedRaster::from_rgb8(
            width,
            height,
            vec![0; length],
        )?)?)
    }

    fn rectangle_map(
        width: u32,
        height: u32,
        rectangle: (u32, u32, u32, u32),
        score: f32,
    ) -> Result<DetectorMap, Box<dyn std::error::Error>> {
        let mut probabilities = vec![0.0; usize::try_from(u64::from(width) * u64::from(height))?];
        let width_usize = usize::try_from(width)?;
        for y in rectangle.1..=rectangle.3 {
            for x in rectangle.0..=rectangle.2 {
                probabilities[usize::try_from(y)? * width_usize + usize::try_from(x)?] = score;
            }
        }
        Ok(DetectorMap::new(probabilities, width, height)?)
    }

    #[test]
    fn detector_map_rejects_invalid_outputs_and_redacts_probabilities() {
        for probabilities in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![-0.1],
            vec![1.1],
        ] {
            assert!(DetectorMap::new(probabilities, 1, 1).is_err());
        }
        assert!(DetectorMap::new(vec![0.5], 0, 1).is_err());
        assert!(DetectorMap::new(vec![0.5], 1, 2).is_err());
        let map = DetectorMap::new(vec![0.123_456], 1, 1).unwrap_or_else(|_| unreachable!());
        let debug = format!("{map:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("0.123456"));
    }

    #[test]
    fn rectangle_matches_pinned_projection_golden() -> Result<(), Box<dyn std::error::Error>> {
        let input = input(32, 32)?;
        let map = rectangle_map(32, 32, (8, 10, 23, 17), 0.9)?;
        let regions = DbPostProcessor.process(&map, &input, &context(None)?)?;
        assert_eq!(regions.len(), 1);
        assert!((regions[0].detector_score().get() - 0.9).abs() <= f32::EPSILON);
        let points = regions[0].polygon().points;
        assert_eq!(
            points.map(|point| (point.x, point.y)),
            [(4.0, 6.0), (27.0, 6.0), (27.0, 21.0), (4.0, 21.0)]
        );
        Ok(())
    }

    #[test]
    fn threshold_score_and_short_side_boundaries_are_exact()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = input(32, 32)?;
        let threshold = rectangle_map(32, 32, (8, 8, 15, 15), DB_THRESHOLD)?;
        assert!(
            DbPostProcessor
                .process(&threshold, &input, &context(None)?)?
                .is_empty()
        );
        let accepted = rectangle_map(32, 32, (8, 8, 15, 15), DB_BOX_THRESHOLD)?;
        assert_eq!(
            DbPostProcessor
                .process(&accepted, &input, &context(None)?)?
                .len(),
            1
        );
        let rejected = rectangle_map(32, 32, (8, 8, 15, 15), DB_BOX_THRESHOLD - 0.001)?;
        assert!(
            DbPostProcessor
                .process(&rejected, &input, &context(None)?)?
                .is_empty()
        );
        let too_small = rectangle_map(32, 32, (8, 8, 10, 10), 0.9)?;
        assert!(
            DbPostProcessor
                .process(&too_small, &input, &context(None)?)?
                .is_empty()
        );
        let minimum = rectangle_map(32, 32, (8, 8, 11, 11), 0.9)?;
        assert_eq!(
            DbPostProcessor
                .process(&minimum, &input, &context(None)?)?
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn diagonal_foreground_uses_eight_neighbor_connectivity()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut foreground = vec![0_u8; 32 * 32];
        for y in 4..8 {
            for x in 4..8 {
                foreground[y * 32 + x] = 1;
            }
        }
        for y in 8..12 {
            for x in 8..12 {
                foreground[y * 32 + x] = 1;
            }
        }
        assert_eq!(
            extract_contours(&foreground, 32, 32, &context(None)?)?.len(),
            1
        );
        Ok(())
    }

    #[test]
    fn filled_quad_scanlines_match_opencv_goldens() -> Result<(), impossible_ocr_domain::OcrError> {
        let slanted = filled_quad_spans(&[(1, 1), (6, 2), (5, 5), (0, 4)], 0, 6, 1, 5)?;
        assert_eq!(
            slanted,
            vec![
                Some((1, 3)),
                Some((1, 6)),
                Some((0, 6)),
                Some((0, 5)),
                Some((3, 5)),
            ]
        );
        let trapezoid = filled_quad_spans(&[(1, 1), (5, 1), (4, 4), (0, 4)], 0, 5, 1, 4)?;
        assert_eq!(
            trapezoid,
            vec![Some((1, 5)), Some((1, 5)), Some((0, 4)), Some((0, 4))]
        );
        Ok(())
    }

    #[test]
    fn candidate_cap_is_applied_before_filtering() -> Result<(), Box<dyn std::error::Error>> {
        let input = input(960, 64)?;
        let mut probabilities = vec![0.0; 960 * 64];
        for candidate in 0..=DB_MAX_CANDIDATES {
            let origin_x = (candidate % 150) * 6;
            let origin_y = (candidate / 150) * 6;
            for y in origin_y..origin_y + 4 {
                for x in origin_x..origin_x + 4 {
                    probabilities[y * 960 + x] = 0.9;
                }
            }
        }
        let map = DetectorMap::new(probabilities, 960, 64)?;
        assert_eq!(
            DbPostProcessor
                .process(&map, &input, &context(None)?)?
                .len(),
            DB_MAX_CANDIDATES
        );
        Ok(())
    }

    #[test]
    fn shape_cancellation_and_deadline_fail_with_stable_codes()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = input(32, 32)?;
        let wrong_shape = DetectorMap::new(vec![0.0; 32 * 64], 32, 64)?;
        assert_eq!(
            DbPostProcessor
                .process(&wrong_shape, &input, &context(None)?)
                .map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::Internal)
        );

        let map = rectangle_map(32, 32, (8, 8, 15, 15), 0.9)?;
        let cancellation = CancellationToken::new();
        assert!(cancellation.cancel());
        let cancelled = RequestContext::new(
            RequestIdSource::default().next()?,
            cancellation,
            Some(Duration::ZERO),
        )?;
        assert_eq!(
            DbPostProcessor
                .process(&map, &input, &cancelled)
                .map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::Cancelled)
        );
        assert_eq!(
            DbPostProcessor
                .process(&map, &input, &context(Some(Duration::ZERO))?)
                .map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::DeadlineExceeded)
        );
        Ok(())
    }

    #[test]
    fn ties_to_even_projection_rounding_is_exact() -> Result<(), impossible_ocr_domain::OcrError> {
        for (value, expected) in [
            (-1.5_f64, -2.0_f64),
            (-0.5, 0.0),
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
        ] {
            assert_eq!(round_ties_even(value)?.to_bits(), expected.to_bits());
        }
        Ok(())
    }

    #[test]
    fn right_bottom_projection_stays_crop_safe() -> Result<(), Box<dyn std::error::Error>> {
        let raster = DecodedRaster::from_rgb8(32, 32, vec![0; 32 * 32 * 3])?;
        let input = preprocess_detector(&raster)?;
        let map = rectangle_map(32, 32, (24, 24, 31, 31), 0.9)?;
        let regions = DbPostProcessor.process(&map, &input, &context(None)?)?;
        assert_eq!(regions.len(), 1);
        let polygon = regions[0].polygon();
        assert!(
            polygon
                .points
                .iter()
                .all(|point| point.x <= 31.0 && point.y <= 31.0)
        );
        assert!(
            polygon
                .points
                .iter()
                .any(|point| point.x.to_bits() == 31.0_f32.to_bits())
        );
        assert!(
            polygon
                .points
                .iter()
                .any(|point| point.y.to_bits() == 31.0_f32.to_bits())
        );
        let crop = perspective_crop(&raster, polygon)?;
        assert!(crop.width() > 0 && crop.height() > 0);
        Ok(())
    }

    #[test]
    fn tiny_padding_projection_uses_detector_scale_not_map_ratio()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = input(10, 10)?;
        assert_eq!(input.scale_x().to_bits(), 1.0_f32.to_bits());
        assert_eq!(input.scale_y().to_bits(), 1.0_f32.to_bits());
        let map = rectangle_map(32, 32, (1, 1, 8, 8), 0.9)?;
        let regions = DbPostProcessor.process(&map, &input, &context(None)?)?;
        assert_eq!(regions.len(), 1);
        assert_eq!(
            regions[0].polygon().points.map(|point| (point.x, point.y)),
            [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)]
        );
        Ok(())
    }
}
