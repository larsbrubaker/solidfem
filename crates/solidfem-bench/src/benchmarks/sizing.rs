//! How big a volume mesh will be before it is made, and the edge length that keeps it inside a
//! budget. A caller states the budget in unknowns, because the solve's memory and time follow
//! unknowns, and describes the part by what can be read from its surface alone: enclosed volume,
//! surface area and its 5th-percentile wall thickness. [`predicted_tets`] estimates the finished
//! 10-node mesh's element count at a uniform edge length `h` meshed by `volume.rs` with two layers
//! through thin walls, and [`edge_for_unknowns`] inverts it. The corpus (`corpus/remesh_trial.rs`)
//! sizes every part with it, one edge for both the surface remesh and the volume stage.
//!
//! The estimate is three terms, each an edge-length scaling with a constant fitted, not derived
//! from ideal tetrahedra, to 73 real parts that reached a 10-node mesh in a Thingi10K corpus run
//! (`solidfem-corpus batch` on a 150-file sample, remeshed at half the 5th-percentile wall):
//! - surface: `area / (sqrt(3)/4 h^2)` ideal triangles, times [`TETS_PER_SURFACE_TRIANGLE`]: the
//!   tetrahedra surface recovery and its neighbourhood need per triangle;
//! - through-wall layers: the same triangle count times [`LAYER_TETS_PER_SURFACE_TRIANGLE`] times
//!   how far the wall falls short of two layers, `2h / wall_p5 - 1`, clamped to `0..=1` (the fit
//!   saw shortfalls up to 0.95; a thinner wall than that is outside what was measured);
//! - interior: a fill of regular tetrahedra, `6 sqrt 2 V / h^3`, times [`INTERIOR_FILL_FACTOR`].
//!
//! The fit minimised the squared relative error. Measured over predicted on those 73 parts:
//! p10 0.928, p50 0.989, p90 1.128, min 0.885, max 2.251 ([`PREDICTION_BAND`]). The upper tail is
//! parts whose surface the remesher could not keep near the ideal triangle count (curved detail
//! and creases forced up to 4.3 times as many triangles), which area alone cannot see.
//!
//! Where it is known to be off, measured on boxes: a plate thin throughout (2.0 thick at edge 1.5,
//! wall ratio 1.5) came in at 0.86, the layer term counting layers the fill already made; and a
//! thin shell far past its walls (0.62 walls at edge 1.56, wall ratio 5) came in at 2.46, the
//! layers then costing elements the clamped term does not see. The same shell at the corpus's
//! budget (wall ratio 2.4) came in at 0.98. Past a wall ratio of about 2.4 the estimate is
//! unmeasured and can be several times low.

/// Unknowns per element of a finished 10-node mesh: three per node, nodes a little over a third
/// of the elements once the mid-edge nodes are counted, less the few the supports fix. Median of
/// the 73 corpus solves (p10 4.27, p90 4.71).
pub const UNKNOWNS_PER_TET: f64 = 4.46;

/// Multigrid storage per unknown in bytes, matrix and every coarse level: median of the corpus
/// solves (p10 910, p90 981, max 1187). A caller turns an unknowns budget into memory with it.
pub const MULTIGRID_BYTES_PER_UNKNOWN: f64 = 944.0;

/// Elements per ideal surface triangle: fitted.
pub const TETS_PER_SURFACE_TRIANGLE: f64 = 0.8425;

/// Extra elements per ideal surface triangle for a wall one whole layer short: fitted.
pub const LAYER_TETS_PER_SURFACE_TRIANGLE: f64 = 0.8969;

/// The interior's elements over a fill of regular tetrahedra of the edge: fitted.
pub const INTERIOR_FILL_FACTOR: f64 = 1.253;

/// Measured over predicted elements on the fitted parts: p10, p50, p90 and the extremes.
pub const PREDICTION_BAND: PredictionBand = PredictionBand {
    min: 0.885,
    p10: 0.928,
    p50: 0.989,
    p90: 1.128,
    max: 2.251,
};

/// Elements requested through each thin wall, as every benchmark case and the corpus request.
pub const LAYERS_THROUGH_WALLS: f64 = 2.0;

/// The area of an equilateral triangle of edge 1.
const TRIANGLE_AREA_PER_EDGE_SQUARED: f64 = 0.433_012_701_892_219_3;

/// Regular tetrahedra per unit volume at edge 1: one has volume `1 / (6 sqrt 2)`.
const TETS_PER_CUBE_EDGE: f64 = 8.485_281_374_238_57;

/// Bisection steps for [`edge_for_unknowns`]: enough to pin the edge to the last bit of a
/// bracket a factor of two wide, and a fixed count so two runs give the same bits.
const BISECTION_STEPS: usize = 64;

/// The spread of measured over predicted elements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionBand {
    pub min: f64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub max: f64,
}

/// What the estimate reads from a part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartSize {
    /// Enclosed volume.
    pub volume: f64,
    /// Surface area.
    pub area: f64,
    /// 5th-percentile wall thickness; not finite or not positive when no wall could be read,
    /// which drops the layer term.
    pub wall_p5: f64,
}

/// Why no edge length could be chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizingError {
    /// The volume or area is negative or not finite.
    InvalidPart { volume: f64, area: f64 },
    /// The budget is not a positive finite number of unknowns.
    InvalidBudget(f64),
}

/// The finished element count predicted for `part` meshed at edge `h`.
pub fn predicted_tets(part: &PartSize, h: f64) -> f64 {
    let triangles = part.area / (TRIANGLE_AREA_PER_EDGE_SQUARED * h * h);
    let shortfall = if part.wall_p5.is_finite() && part.wall_p5 > 0.0 {
        (LAYERS_THROUGH_WALLS * h / part.wall_p5 - 1.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let surface = TETS_PER_SURFACE_TRIANGLE * triangles;
    let layers = LAYER_TETS_PER_SURFACE_TRIANGLE * triangles * shortfall;
    let interior = INTERIOR_FILL_FACTOR * TETS_PER_CUBE_EDGE * part.volume / (h * h * h);
    surface + layers + interior
}

/// The unknowns predicted for `part` meshed at edge `h`.
pub fn predicted_unknowns(part: &PartSize, h: f64) -> f64 {
    predicted_tets(part, h) * UNKNOWNS_PER_TET
}

/// The smallest edge length whose predicted unknowns for `part` are within `budget`; 0 for a
/// part with neither area nor volume, which any edge meets.
///
/// # Errors
/// [`SizingError::InvalidPart`] for a negative or non-finite volume or area,
/// [`SizingError::InvalidBudget`] for a budget that is not positive and finite.
pub fn edge_for_unknowns(part: &PartSize, budget: f64) -> Result<f64, SizingError> {
    let valid = |x: f64| x.is_finite() && x >= 0.0;
    if !valid(part.volume) || !valid(part.area) {
        return Err(SizingError::InvalidPart {
            volume: part.volume,
            area: part.area,
        });
    }
    if !(budget.is_finite() && budget > 0.0) {
        return Err(SizingError::InvalidBudget(budget));
    }
    if part.volume == 0.0 && part.area == 0.0 {
        return Ok(0.0);
    }
    // The prediction falls strictly as `h` grows (the layer term grows at most linearly while
    // the triangles fall with its square), so one bracket and a bisection find the edge. Start
    // from the edge either term alone would need, then widen until the budget is straddled.
    let over = |h: f64| predicted_unknowns(part, h) > budget;
    let mut high = (part.area / (TRIANGLE_AREA_PER_EDGE_SQUARED * budget))
        .sqrt()
        .max((TETS_PER_CUBE_EDGE * part.volume / budget).cbrt());
    while over(high) {
        high *= 2.0;
    }
    let mut low = high * 0.5;
    while !over(low) && low > 0.0 {
        low *= 0.5;
    }
    for _ in 0..BISECTION_STEPS {
        let middle = (low + high) * 0.5;
        if over(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(high)
}

/// How far edge `h` is from two layers through the part's 5th-percentile wall: `h` over half the
/// wall, so above 1 the thin walls are under-resolved. `None` when no wall could be read.
pub fn wall_resolution(part: &PartSize, h: f64) -> Option<f64> {
    (part.wall_p5.is_finite() && part.wall_p5 > 0.0)
        .then(|| h * LAYERS_THROUGH_WALLS / part.wall_p5)
}
