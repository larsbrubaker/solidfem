//! Isotropic materials and the field that supplies them. Properties are asked for at every
//! integration point by position, so a caller can describe a printed part's solid shell and
//! weaker infill core without the mesh following the shell.

/// Isotropic linear elastic properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Young's modulus: positive and finite.
    pub youngs_modulus: f64,
    /// Poisson's ratio: above -1 and below 0.5, where the stiffness becomes singular.
    pub poisson_ratio: f64,
}

impl Material {
    /// Whether these properties describe a stable solid.
    pub fn is_valid(&self) -> bool {
        self.youngs_modulus.is_finite()
            && self.youngs_modulus > 0.0
            && self.poisson_ratio > -1.0
            && self.poisson_ratio < 0.5
    }

    /// The Lamé constants (lambda, mu).
    pub fn lame(&self) -> (f64, f64) {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        (
            e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            e / (2.0 * (1.0 + nu)),
        )
    }
}

/// Material properties as a function of position.
pub trait MaterialField {
    /// The material at `position`.
    fn at(&self, position: [f64; 3]) -> Material;
}

impl MaterialField for Material {
    fn at(&self, _position: [f64; 3]) -> Material {
        *self
    }
}

impl<F: Fn([f64; 3]) -> Material> MaterialField for F {
    fn at(&self, position: [f64; 3]) -> Material {
        self(position)
    }
}
