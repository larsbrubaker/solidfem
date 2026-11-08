//! Integration rules on the reference tetrahedron and triangle. Weights sum to one; the caller
//! scales by the reference measure. The 4-point rule (degree 2) integrates a straight-sided
//! quadratic element's stiffness exactly for uniform material, since its integrand is quadratic.
//! The 14-point rule (degree 5) exists for material that varies inside an element, where four
//! samples are a crude average. The tests check each rule's degree against exact monomial
//! integrals, so a wrong constant cannot hide here.

/// One integration point: natural coordinates and a weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePoint {
    /// Natural coordinates (r, s, t), the barycentrics of corners 1, 2 and 3.
    pub natural: [f64; 3],
    /// The weight, normalized so a rule's weights sum to one.
    pub weight: f64,
}

/// Which integration rule to use on each tetrahedron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quadrature {
    /// Four points, degree 2: exact stiffness for uniform material on straight elements.
    #[default]
    FourPoint,
    /// Fourteen points, degree 5: for material varying inside elements.
    FourteenPoint,
}

impl Quadrature {
    /// The rule's points, in a fixed order.
    pub fn points(self) -> &'static [QuadraturePoint] {
        match self {
            Quadrature::FourPoint => &FOUR_POINT,
            Quadrature::FourteenPoint => &FOURTEEN_POINT,
        }
    }
}

const fn point(r: f64, s: f64, t: f64, weight: f64) -> QuadraturePoint {
    QuadraturePoint {
        natural: [r, s, t],
        weight,
    }
}

const A4: f64 = 0.1381966011250105;
const B4: f64 = 0.5854101966249685;
const FOUR_POINT: [QuadraturePoint; 4] = [
    point(A4, A4, A4, 0.25),
    point(B4, A4, A4, 0.25),
    point(A4, B4, A4, 0.25),
    point(A4, A4, B4, 0.25),
];

const A14: f64 = 0.3108859192633006;
const B14: f64 = 1.0 - 3.0 * A14;
const W14A: f64 = 0.11268792571801585;
const C14: f64 = 0.09273525031089123;
const D14: f64 = 1.0 - 3.0 * C14;
const W14B: f64 = 0.07349304311636195;
const E14: f64 = 0.04550370412564965;
const F14: f64 = 0.5 - E14;
const W14C: f64 = 0.042546020777081466;
const FOURTEEN_POINT: [QuadraturePoint; 14] = [
    point(A14, A14, A14, W14A),
    point(B14, A14, A14, W14A),
    point(A14, B14, A14, W14A),
    point(A14, A14, B14, W14A),
    point(C14, C14, C14, W14B),
    point(D14, C14, C14, W14B),
    point(C14, D14, C14, W14B),
    point(C14, C14, D14, W14B),
    point(E14, E14, F14, W14C),
    point(E14, F14, E14, W14C),
    point(F14, E14, E14, W14C),
    point(E14, F14, F14, W14C),
    point(F14, E14, F14, W14C),
    point(F14, F14, E14, W14C),
];

const TA: f64 = 0.091576213509771;
const TB: f64 = 0.445948490915965;
const WA: f64 = 0.109951743655322;
const WB: f64 = 0.223381589678011;

/// A degree-4 rule on the reference triangle: `([r, s], weight)`, weights summing to one. Traction
/// loads use it; a 6-node face's shape functions are quadratic, and a curved face's area factor
/// raises the degree.
pub const TRIANGLE_POINTS: [([f64; 2], f64); 6] = [
    ([TA, TA], WA),
    ([1.0 - 2.0 * TA, TA], WA),
    ([TA, 1.0 - 2.0 * TA], WA),
    ([TB, TB], WB),
    ([1.0 - 2.0 * TB, TB], WB),
    ([TB, 1.0 - 2.0 * TB], WB),
];
