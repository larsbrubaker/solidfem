//! Signed-distance functions for the benchmark shapes, negative inside. Distances are exact on the
//! flat faces the supports and loads sit on; combinations by `min` and `max` are exact near each
//! surface, which is what the mesher reads.

/// Signed distance to the axis-aligned box from `low` to `high`.
pub fn box_sdf(p: [f64; 3], low: [f64; 3], high: [f64; 3]) -> f64 {
    let mut outside = 0.0f64;
    let mut largest = f64::NEG_INFINITY;
    for axis in 0..3 {
        let center = 0.5 * (low[axis] + high[axis]);
        let half = 0.5 * (high[axis] - low[axis]);
        let q = (p[axis] - center).abs() - half;
        largest = largest.max(q);
        let positive = q.max(0.0);
        outside += positive * positive;
    }
    outside.sqrt() + largest.min(0.0)
}

/// A hollow closed box: outer box `[0, size]^3` with walls `wall` thick.
pub fn hollow_box(p: [f64; 3], size: f64, wall: f64) -> f64 {
    let outer = box_sdf(p, [0.0; 3], [size; 3]);
    let inner = box_sdf(p, [wall; 3], [size - wall; 3]);
    outer.max(-inner)
}

/// A tube along x from 0 to `length`, outer radius `outer`, wall `wall`, centred on `axis_center`.
pub fn tube(p: [f64; 3], length: f64, outer: f64, wall: f64, axis_center: [f64; 2]) -> f64 {
    let (dy, dz) = (p[1] - axis_center[0], p[2] - axis_center[1]);
    let radial = (dy * dy + dz * dz).sqrt();
    let shell = (radial - (outer - 0.5 * wall)).abs() - 0.5 * wall;
    let along = (p[0] - 0.5 * length).abs() - 0.5 * length;
    shell.max(along)
}

/// An L-shaped bracket: a base plate in z and a wall along z at x = 0.
pub fn l_bracket(p: [f64; 3], length: f64, width: f64, thickness: f64) -> f64 {
    let base = box_sdf(p, [0.0; 3], [length, width, thickness]);
    let wall = box_sdf(p, [0.0; 3], [thickness, width, length]);
    base.min(wall)
}

/// A square plate `size` on a side and `plate` thick with crossing ribs on top, `rib` wide and
/// `rib_height` tall above the plate, `pitch` apart.
pub fn ribbed_plate(
    p: [f64; 3],
    size: f64,
    plate: f64,
    rib: f64,
    rib_height: f64,
    pitch: f64,
) -> f64 {
    let mut distance = box_sdf(p, [0.0; 3], [size, size, plate]);
    let mut at = pitch;
    while at < size - 1e-9 {
        let along_x = box_sdf(
            p,
            [0.0, at - 0.5 * rib, plate - 0.5 * rib],
            [size, at + 0.5 * rib, plate + rib_height],
        );
        let along_y = box_sdf(
            p,
            [at - 0.5 * rib, 0.0, plate - 0.5 * rib],
            [at + 0.5 * rib, size, plate + rib_height],
        );
        distance = distance.min(along_x).min(along_y);
        at += pitch;
    }
    distance
}
