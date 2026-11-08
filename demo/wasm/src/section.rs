//! Cross-section mechanics for the demo's bars, from first principles: area, second moment,
//! Timoshenko shear coefficient and Saint-Venant torsion constant. The bar runs along x, height is
//! along z (the push direction) and width along y; `textbook.rs` builds the expected answers from
//! these numbers and `part.rs` meshes the same outline.

use std::f64::consts::PI;

/// The section outline: a solid rectangle or a closed rectangular tube with a uniform wall.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Section {
    /// Width along y, height along z.
    Solid { width: f64, height: f64 },
    /// Outer width and height and the wall thickness all round.
    Tube { width: f64, height: f64, wall: f64 },
}

impl Section {
    /// Outer width along y.
    pub fn width(&self) -> f64 {
        match *self {
            Section::Solid { width, .. } | Section::Tube { width, .. } => width,
        }
    }

    /// Outer height along z.
    pub fn height(&self) -> f64 {
        match *self {
            Section::Solid { height, .. } | Section::Tube { height, .. } => height,
        }
    }

    /// Cross-section area.
    pub fn area(&self) -> f64 {
        match *self {
            Section::Solid { width, height } => width * height,
            Section::Tube {
                width,
                height,
                wall,
            } => width * height - (width - 2.0 * wall) * (height - 2.0 * wall),
        }
    }

    /// Second moment of area about the horizontal axis through the centroid (bending under a
    /// vertical push): `(w h^3 - w_i h_i^3) / 12` for the tube, the rectangle's with no hole.
    pub fn inertia(&self) -> f64 {
        match *self {
            Section::Solid { width, height } => width * height * height * height / 12.0,
            Section::Tube {
                width,
                height,
                wall,
            } => {
                let inner_w = width - 2.0 * wall;
                let inner_h = height - 2.0 * wall;
                (width * height * height * height - inner_w * inner_h * inner_h * inner_h) / 12.0
            }
        }
    }

    /// The Timoshenko shear coefficient for a shear force along the height, derived by equating
    /// the shear strain energy of the beam (`V^2 / (2 k G A)` per length) to the integral of
    /// `tau^2 / (2 G)` over the section with `tau = V Q / (I b)`, the classical first-moment
    /// shear stress with `Q` the first moment of the area above a level and `b` the width at it:
    ///
    /// `k = I^2 / (A * integral Q(z)^2 / b(z) dz)`.
    ///
    /// For the rectangle this integral is exact and gives 5/6. For the tube the flanges and webs
    /// each give `Q^2 / b` as a polynomial of degree 4 in `z`, so a four-point Gauss-Legendre rule
    /// (exact to degree 7) on each piece gives the exact integral.
    pub fn shear_coefficient(&self) -> f64 {
        let half = 0.5 * self.height();
        // Pieces of the upper half, where Q and b take one polynomial form; symmetric below.
        let pieces: Vec<(f64, f64, f64)> = match *self {
            Section::Solid { width, .. } => vec![(0.0, half, width)],
            Section::Tube { width, wall, .. } => {
                let flange = half - wall;
                vec![(0.0, flange, 2.0 * wall), (flange, half, width)]
            }
        };
        let first_moment = |z: f64| -> f64 {
            match *self {
                Section::Solid { width, .. } => 0.5 * width * (half * half - z * z),
                Section::Tube { width, wall, .. } => {
                    let flange = half - wall;
                    if z >= flange {
                        0.5 * width * (half * half - z * z)
                    } else {
                        0.5 * width * (half * half - flange * flange)
                            + wall * (flange * flange - z * z)
                    }
                }
            }
        };
        let mut integral = 0.0;
        for (low, high, width_at) in pieces {
            for (point, weight) in GAUSS_4 {
                let z = 0.5 * (low + high) + 0.5 * (high - low) * point;
                let q = first_moment(z);
                integral += 0.5 * (high - low) * weight * q * q / width_at;
            }
        }
        // The upper half doubles for the symmetric lower half.
        integral *= 2.0;
        let inertia = self.inertia();
        inertia * inertia / (self.area() * integral)
    }

    /// The Saint-Venant torsion constant `J`, from the series for a rectangle with sides
    /// `a >= b`. Only the solid rectangle has one here: a closed thick-walled tube has no simple
    /// closed form, so the tube returns `None` and the demo reports no textbook answer.
    pub fn torsion_constant(&self) -> Option<f64> {
        match *self {
            Section::Solid { width, height } => {
                let (a, b) = if width >= height {
                    (width, height)
                } else {
                    (height, width)
                };
                // J = a b^3 / 3 * (1 - 192 b / (pi^5 a) * sum tanh(k pi a / (2 b)) / k^5), odd k.
                // Two hundred terms reach f64 precision: the terms fall as k^-5.
                let mut sum = 0.0;
                for n in 0..200 {
                    let k = (2 * n + 1) as f64;
                    sum += (k * PI * a / (2.0 * b)).tanh() / (k * k * k * k * k);
                }
                let series = 1.0 - 192.0 * b / (PI * PI * PI * PI * PI * a) * sum;
                Some(a * b * b * b / 3.0 * series)
            }
            Section::Tube { .. } => None,
        }
    }
}

/// Four-point Gauss-Legendre nodes and weights on [-1, 1]; exact to polynomial degree 7.
const GAUSS_4: [(f64, f64); 4] = [
    (-0.861_136_311_594_052_6, 0.347_854_845_137_453_85),
    (-0.339_981_043_584_856_26, 0.652_145_154_862_546_1),
    (0.339_981_043_584_856_26, 0.652_145_154_862_546_1),
    (0.861_136_311_594_052_6, 0.347_854_845_137_453_85),
];
