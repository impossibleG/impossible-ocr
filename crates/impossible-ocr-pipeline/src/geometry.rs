//! Deterministic, pure-Rust geometry primitives used by OCR post-processing.

// These crate-private primitives are landed ahead of their DB post-processing consumer.
#![allow(dead_code)]

use std::cmp::Ordering;

use impossible_ocr_domain::{OcrError, OcrErrorCode};

const GEOMETRY_EPSILON: f64 = 1.0e-9;

/// Finite two-dimensional point used for geometry calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GeometryPoint {
    x: f64,
    y: f64,
}

impl GeometryPoint {
    pub(crate) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub(crate) const fn x(self) -> f64 {
        self.x
    }

    pub(crate) const fn y(self) -> f64 {
        self.y
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    fn subtract(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn length(self) -> f64 {
        self.x.hypot(self.y)
    }
}

/// Canonically ordered minimum-area rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrientedRect {
    corners: [GeometryPoint; 4],
    side_a: f64,
    side_b: f64,
}

impl OrientedRect {
    /// Corners in top-left, top-right, bottom-right, bottom-left order.
    pub(crate) const fn corners(self) -> [GeometryPoint; 4] {
        self.corners
    }

    pub(crate) fn short_side(self) -> f64 {
        self.side_a.min(self.side_b)
    }

    pub(crate) fn long_side(self) -> f64 {
        self.side_a.max(self.side_b)
    }

    pub(crate) fn area(self) -> f64 {
        self.side_a * self.side_b
    }

    pub(crate) fn perimeter(self) -> f64 {
        2.0 * (self.side_a + self.side_b)
    }
}

/// Returns the deterministic convex hull without repeating the first point.
///
/// Collinear interior points and exact duplicates are removed. The returned hull has positive
/// signed area in image coordinates, which is clockwise when the vertical axis points down.
pub(crate) fn convex_hull(points: &[GeometryPoint]) -> Result<Vec<GeometryPoint>, OcrError> {
    if points.len() < 3 || points.iter().any(|point| !point.is_finite()) {
        return Err(internal());
    }

    let mut sorted = Vec::new();
    sorted
        .try_reserve_exact(points.len())
        .map_err(|_| internal())?;
    sorted.extend_from_slice(points);
    sorted.sort_unstable_by(compare_points);
    sorted.dedup();
    if sorted.len() < 3 {
        return Err(internal());
    }

    let mut lower = Vec::new();
    let mut upper = Vec::new();
    lower
        .try_reserve_exact(sorted.len())
        .map_err(|_| internal())?;
    upper
        .try_reserve_exact(sorted.len())
        .map_err(|_| internal())?;

    for point in sorted.iter().copied() {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    for point in sorted.iter().rev().copied() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() < 3 || signed_double_area(&lower) <= GEOMETRY_EPSILON {
        return Err(internal());
    }
    Ok(lower)
}

/// Computes the deterministic minimum-area rectangle around a point set.
///
/// Rotating support points evaluate every hull-edge orientation in linear time after hull
/// construction. Detector maps are capped at 960 by 960 and contours are capped separately by the
/// DB postprocessor.
pub(crate) fn minimum_area_rect(points: &[GeometryPoint]) -> Result<OrientedRect, OcrError> {
    let hull = convex_hull(points)?;
    let mut best: Option<RectCandidate> = None;

    let first_edge = hull[1].subtract(hull[0]);
    let first_length = first_edge.length();
    if !first_length.is_finite() || first_length <= GEOMETRY_EPSILON {
        return Err(internal());
    }
    let first_u = first_edge.scale(1.0 / first_length);
    let first_v = GeometryPoint::new(-first_u.y, first_u.x);
    let mut supports = [
        support_index(&hull, first_u),
        support_index(&hull, first_u.scale(-1.0)),
        support_index(&hull, first_v),
        support_index(&hull, first_v.scale(-1.0)),
    ];

    for (start, end) in hull
        .iter()
        .copied()
        .zip(hull.iter().copied().cycle().skip(1))
        .take(hull.len())
    {
        let edge = end.subtract(start);
        let edge_length = edge.length();
        if !edge_length.is_finite() || edge_length <= GEOMETRY_EPSILON {
            continue;
        }
        let axis_u = edge.scale(1.0 / edge_length);
        let axis_v = GeometryPoint::new(-axis_u.y, axis_u.x);
        supports[0] = advance_support(&hull, supports[0], axis_u);
        supports[1] = advance_support(&hull, supports[1], axis_u.scale(-1.0));
        supports[2] = advance_support(&hull, supports[2], axis_v);
        supports[3] = advance_support(&hull, supports[3], axis_v.scale(-1.0));
        let max_u = hull[supports[0]].dot(axis_u);
        let min_u = hull[supports[1]].dot(axis_u);
        let max_v = hull[supports[2]].dot(axis_v);
        let min_v = hull[supports[3]].dot(axis_v);
        let side_u = max_u - min_u;
        let side_v = max_v - min_v;
        let area = side_u * side_v;
        if !side_u.is_finite()
            || !side_v.is_finite()
            || !area.is_finite()
            || side_u <= GEOMETRY_EPSILON
            || side_v <= GEOMETRY_EPSILON
        {
            continue;
        }
        let raw = [
            axis_u.scale(min_u).add(axis_v.scale(min_v)),
            axis_u.scale(max_u).add(axis_v.scale(min_v)),
            axis_u.scale(max_u).add(axis_v.scale(max_v)),
            axis_u.scale(min_u).add(axis_v.scale(max_v)),
        ];
        let candidate = RectCandidate {
            corners: canonical_quad(raw)?,
            side_u,
            side_v,
            area,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.precedes(current))
        {
            best = Some(candidate);
        }
    }

    best.map(RectCandidate::into_rect).ok_or_else(internal)
}

/// Expands a minimum-area rectangle using the pinned DB unclip distance.
///
/// DB applies unclip only after reducing a contour to its minimum-area rectangle. Expanding each
/// local half-extent by `area * ratio / perimeter` yields the same enclosing minimum rectangle as
/// the reference round-join polygon offset, while remaining deterministic and dependency-free.
pub(crate) fn unclip_rect(rectangle: OrientedRect, ratio: f64) -> Result<OrientedRect, OcrError> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(internal());
    }
    let area = rectangle.area();
    let perimeter = rectangle.perimeter();
    let distance = area * ratio / perimeter;
    if !distance.is_finite() || distance <= 0.0 {
        return Err(internal());
    }

    let [top_left, top_right, bottom_right, bottom_left] = rectangle.corners;
    let top = top_right.subtract(top_left);
    let left = bottom_left.subtract(top_left);
    let top_length = top.length();
    let left_length = left.length();
    if top_length <= GEOMETRY_EPSILON || left_length <= GEOMETRY_EPSILON {
        return Err(internal());
    }
    let axis_u = top.scale(1.0 / top_length);
    let axis_v = left.scale(1.0 / left_length);
    let center = top_left
        .add(top_right)
        .add(bottom_right)
        .add(bottom_left)
        .scale(0.25);
    let half_u = top_length * 0.5 + distance;
    let half_v = left_length * 0.5 + distance;
    let raw = [
        center
            .subtract(axis_u.scale(half_u))
            .subtract(axis_v.scale(half_v)),
        center
            .add(axis_u.scale(half_u))
            .subtract(axis_v.scale(half_v)),
        center.add(axis_u.scale(half_u)).add(axis_v.scale(half_v)),
        center
            .subtract(axis_u.scale(half_u))
            .add(axis_v.scale(half_v)),
    ];
    Ok(OrientedRect {
        corners: canonical_quad(raw)?,
        side_a: top_length + 2.0 * distance,
        side_b: left_length + 2.0 * distance,
    })
}

/// Orders four finite rectangle corners as top-left, top-right, bottom-right, bottom-left.
pub(crate) fn canonical_quad(
    mut points: [GeometryPoint; 4],
) -> Result<[GeometryPoint; 4], OcrError> {
    if points.iter().any(|point| !point.is_finite()) {
        return Err(internal());
    }
    points.sort_by(compare_points);
    let (left_top, left_bottom) = if points[1].y > points[0].y {
        (points[0], points[1])
    } else {
        (points[1], points[0])
    };
    let (right_top, right_bottom) = if points[3].y > points[2].y {
        (points[2], points[3])
    } else {
        (points[3], points[2])
    };
    let ordered = [left_top, right_top, right_bottom, left_bottom];
    if signed_double_area(&ordered) <= GEOMETRY_EPSILON {
        return Err(internal());
    }
    Ok(ordered)
}

#[derive(Debug, Clone, Copy)]
struct RectCandidate {
    corners: [GeometryPoint; 4],
    side_u: f64,
    side_v: f64,
    area: f64,
}

impl RectCandidate {
    fn precedes(&self, other: &Self) -> bool {
        self.area
            .total_cmp(&other.area)
            .then_with(|| compare_quads(&self.corners, &other.corners))
            == Ordering::Less
    }

    fn into_rect(self) -> OrientedRect {
        OrientedRect {
            corners: self.corners,
            side_a: self.side_u,
            side_b: self.side_v,
        }
    }
}

fn compare_points(left: &GeometryPoint, right: &GeometryPoint) -> Ordering {
    left.x
        .total_cmp(&right.x)
        .then_with(|| left.y.total_cmp(&right.y))
}

fn compare_quads(left: &[GeometryPoint; 4], right: &[GeometryPoint; 4]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| compare_points(left, right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn support_index(hull: &[GeometryPoint], direction: GeometryPoint) -> usize {
    let mut best = 0;
    for candidate in 1..hull.len() {
        if hull[candidate].dot(direction) >= hull[best].dot(direction) {
            best = candidate;
        }
    }
    best
}

fn advance_support(hull: &[GeometryPoint], mut current: usize, direction: GeometryPoint) -> usize {
    for _ in 0..hull.len() {
        let next = (current + 1) % hull.len();
        if hull[next].dot(direction) >= hull[current].dot(direction) {
            current = next;
        } else {
            break;
        }
    }
    current
}

fn cross(origin: GeometryPoint, first: GeometryPoint, second: GeometryPoint) -> f64 {
    let first = first.subtract(origin);
    let second = second.subtract(origin);
    first.x * second.y - first.y * second.x
}

fn signed_double_area(points: &[GeometryPoint]) -> f64 {
    points
        .iter()
        .copied()
        .zip(points.iter().copied().cycle().skip(1))
        .take(points.len())
        .map(|(current, next)| current.x * next.y - next.x * current.y)
        .sum()
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use super::{
        GEOMETRY_EPSILON, GeometryPoint, canonical_quad, convex_hull, minimum_area_rect,
        signed_double_area, unclip_rect,
    };

    fn point(x: f64, y: f64) -> GeometryPoint {
        GeometryPoint::new(x, y)
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() <= 1.0e-8, "{left} != {right}");
    }

    fn brute_force_minimum_area(
        points: &[GeometryPoint],
    ) -> Result<f64, impossible_ocr_domain::OcrError> {
        let hull = convex_hull(points)?;
        let mut best = f64::INFINITY;
        for (start, end) in hull
            .iter()
            .copied()
            .zip(hull.iter().copied().cycle().skip(1))
            .take(hull.len())
        {
            let edge = end.subtract(start);
            let axis_u = edge.scale(1.0 / edge.length());
            let axis_v = GeometryPoint::new(-axis_u.y(), axis_u.x());
            let mut minimum_u = f64::INFINITY;
            let mut maximum_u = f64::NEG_INFINITY;
            let mut minimum_v = f64::INFINITY;
            let mut maximum_v = f64::NEG_INFINITY;
            for point in hull.iter().copied() {
                let u = point.dot(axis_u);
                let v = point.dot(axis_v);
                minimum_u = minimum_u.min(u);
                maximum_u = maximum_u.max(u);
                minimum_v = minimum_v.min(v);
                maximum_v = maximum_v.max(v);
            }
            best = best.min((maximum_u - minimum_u) * (maximum_v - minimum_v));
        }
        Ok(best)
    }

    #[test]
    fn hull_removes_duplicates_collinear_points_and_interior_points()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let hull = convex_hull(&[
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(2.0, 0.0),
            point(2.0, 2.0),
            point(1.0, 1.0),
            point(0.0, 2.0),
            point(0.0, 0.0),
        ])?;
        assert_eq!(
            hull,
            vec![
                point(0.0, 0.0),
                point(2.0, 0.0),
                point(2.0, 2.0),
                point(0.0, 2.0)
            ]
        );
        Ok(())
    }

    #[test]
    fn minimum_rectangle_is_canonical_and_permutation_invariant()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let source = [
            point(1.0, 0.0),
            point(5.0, 2.0),
            point(3.0, 6.0),
            point(-1.0, 4.0),
        ];
        let expected = minimum_area_rect(&source)?;
        assert_close(expected.short_side(), 2.0_f64.hypot(4.0));
        assert_close(expected.long_side(), 2.0_f64.hypot(4.0));
        assert!(signed_double_area(&expected.corners()) > 0.0);

        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let permutation = [a, b, c, d];
                        let mut sorted = permutation;
                        sorted.sort_unstable();
                        if sorted != [0, 1, 2, 3] {
                            continue;
                        }
                        let candidate = minimum_area_rect(&[
                            source[permutation[0]],
                            source[permutation[1]],
                            source[permutation[2]],
                            source[permutation[3]],
                        ])?;
                        for (actual, wanted) in
                            candidate.corners().into_iter().zip(expected.corners())
                        {
                            assert_close(actual.x(), wanted.x());
                            assert_close(actual.y(), wanted.y());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn axis_aligned_unclip_uses_area_ratio_over_perimeter()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let rectangle = minimum_area_rect(&[
            point(0.0, 0.0),
            point(10.0, 0.0),
            point(10.0, 4.0),
            point(0.0, 4.0),
        ])?;
        let expanded = unclip_rect(rectangle, 1.5)?;
        let distance = 40.0 * 1.5 / 28.0;
        assert_close(expanded.short_side(), 4.0 + 2.0 * distance);
        assert_close(expanded.long_side(), 10.0 + 2.0 * distance);
        assert_close(
            expanded.area(),
            (4.0 + 2.0 * distance) * (10.0 + 2.0 * distance),
        );
        assert!(signed_double_area(&expanded.corners()) > 0.0);
        Ok(())
    }

    #[test]
    fn canonical_quad_matches_paddle_corner_convention()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let ordered = canonical_quad([
            point(1.0, 2.0),
            point(2.0, 1.0),
            point(1.0, 0.0),
            point(0.0, 1.0),
        ])?;
        assert_eq!(
            ordered,
            [
                point(1.0, 0.0),
                point(2.0, 1.0),
                point(1.0, 2.0),
                point(0.0, 1.0)
            ]
        );
        assert!(signed_double_area(&ordered) > 0.0);
        Ok(())
    }

    #[test]
    fn invalid_and_degenerate_geometry_fails_closed() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                minimum_area_rect(&[point(0.0, 0.0), point(1.0, 0.0), point(0.0, invalid)])
                    .is_err()
            );
        }
        assert!(minimum_area_rect(&[point(0.0, 0.0), point(1.0, 1.0), point(2.0, 2.0)]).is_err());
        let rectangle = minimum_area_rect(&[
            point(0.0, 0.0),
            point(2.0, 0.0),
            point(2.0, 1.0),
            point(0.0, 1.0),
        ])
        .unwrap_or_else(|_| unreachable!());
        for invalid_ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(unclip_rect(rectangle, invalid_ratio).is_err());
        }
    }

    #[test]
    fn every_small_grid_subset_is_finite_canonical_or_degenerate()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let grid = [
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(2.0, 0.0),
            point(0.0, 1.0),
            point(1.0, 1.0),
            point(2.0, 1.0),
            point(0.0, 2.0),
            point(1.0, 2.0),
            point(2.0, 2.0),
        ];
        for mask in 0_u16..(1_u16 << grid.len()) {
            let points: Vec<_> = grid
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, point)| ((mask & (1 << index)) != 0).then_some(point))
                .collect();
            if points.len() < 3 {
                assert!(minimum_area_rect(&points).is_err());
                continue;
            }
            if let Ok(rectangle) = minimum_area_rect(&points) {
                assert_close(rectangle.area(), brute_force_minimum_area(&points)?);
                assert!(rectangle.short_side().is_finite());
                assert!(rectangle.short_side() > GEOMETRY_EPSILON);
                assert!(rectangle.long_side().is_finite());
                assert!(rectangle.area().is_finite());
                assert!(signed_double_area(&rectangle.corners()) > GEOMETRY_EPSILON);
                for corner in rectangle.corners() {
                    assert!(corner.x().is_finite() && corner.y().is_finite());
                }
                let expanded = unclip_rect(rectangle, 1.5)?;
                assert!(expanded.short_side() > rectangle.short_side());
                assert!(expanded.long_side() > rectangle.long_side());
            }
        }
        Ok(())
    }
}
