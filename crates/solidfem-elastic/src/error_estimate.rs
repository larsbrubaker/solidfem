//! How converged a solution is: the energy-norm difference between the recovered stress
//! (`results.rs`) and each element's own stress, per element and over the whole model.
//!
//! **Why this measures the error.** The exact stress is continuous and the element stress is not;
//! the recovered field is a continuous stress fitted to the elements' most accurate values, so
//! where the mesh resolves the stress well the two agree and where it does not they differ by
//! about as much as the element stress differs from the exact one. The difference is measured in
//! the energy norm, the norm the solution itself is best in: per element
//! `e_K² = ∫_K (σ* - σ_h) : C⁻¹ : (σ* - σ_h) dV`, with `C⁻¹` the material's compliance, and
//! [`ErrorEstimate::element_errors`] holds `e_K`. The model's estimated error is
//! `‖e‖² = Σ e_K²` and its relative error is `η = ‖e‖ / sqrt(‖u_h‖² + ‖e‖²)`, where
//! `‖u_h‖² = ∫ σ_h : C⁻¹ : σ_h dV` is twice the solution's strain energy. Where every held
//! displacement is zero or exactly represented by the elements and the loads are integrated
//! exactly, the error of the solution is energy-orthogonal to it, so `‖u_h‖² + ‖e‖²` estimates
//! the exact solution's norm and `η` is the error as a fraction of the exact answer; with
//! interpolated prescribed displacements that is an approximation. `η` is below one.
//!
//! **What `η` reads, as measured** on the test fields (`tests/error_estimate/`) and on loaded
//! benchmark parts. It is an energy-norm measure of the whole model, not an accuracy of the peak
//! stress: at a plate's hole the peak von Mises read 14% low while `η` read 2.7%. On a smooth field held at every boundary node
//! it over-read the true energy-norm error by 1.5 to 1.75 times; on clamped parts under load it
//! has under-read it, down to about 0.7 times, most of the estimated error sitting in the
//! clamp's elements. Under a point load the exact energy is infinite, so `η` does not fall as the
//! mesh refines. Read it as an indicator of the order of the error, and of where it sits.
//!
//! **The recovered field inside an element** is the 10-node interpolation of the recovered nodal
//! stress: each corner takes its vertex's patch fit at the corner and each midside the mean of its
//! edge's two fits at the midside, the values [`crate::RecoveredField::nodal_stress`] reports.
//! It is continuous between elements because neighbouring elements share those nodes.
//!
//! **Integration.** The difference is quadratic inside a straight element, so its energy is of
//! degree four, which the 4-point rule (degree two) cannot integrate: it can read zero for a
//! difference that is not. Both integrals therefore use the 14-point rule (degree five), exact on
//! straight elements of uniform material whatever rule the study solved with; the element stress
//! is evaluated from the displacements at those points.
//!
//! **What it refuses.** A recovery begun before the solve converged is a preview, not an answer,
//! and gives [`ModelError::NotFinished`]. A vertex whose patch was not fitted has no recovered
//! field to compare with, so the estimate stops rather than reading the fallback value as
//! converged: [`ModelError::MixedMaterialPatch`] where the patch holds more than one material, so
//! a part of more than one material cannot be estimated yet, and [`ModelError::CoplanarPatch`]
//! where its points lie in one plane. A recovery whose mesh counts differ from the study's gives
//! [`ModelError::RecoveryMeshMismatch`]; a solution with no strain energy, which has no scale for
//! a relative error, [`ModelError::ZeroStrainEnergy`]; and sums that are not finite
//! [`ModelError::NonFiniteEnergy`].
//!
//! **Slices.** [`ErrorEstimation`] runs one element per unit of work, in element order, adding
//! each element's two integrals to running sums in that same order, so an estimate computed in any
//! slice sizes gives exactly the bits of one run.

use solidfem_mesh::QuadraticTetMesh;

use crate::element::{checked_material, point_geometry, strain_at, stress_at};
use crate::error::ModelError;
use crate::material::{Material, MaterialField};
use crate::quadrature::Quadrature;
use crate::results::{midside_mean, Recovery};
use crate::shape::{shape_values, EDGES, NODE_COUNT};

/// The rule both integrals use; see the header.
const RULE: Quadrature = Quadrature::FourteenPoint;

/// A solution's estimated discretization error.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorEstimate {
    /// Per element, in element order, the energy norm of the difference between the recovered
    /// and the element stress over that element: where the mesh is too coarse, this is large.
    pub element_errors: Vec<f64>,
    /// `‖e‖²`, the sum of the squared element errors.
    pub error_norm_squared: f64,
    /// `‖u_h‖²`, twice the solution's strain energy.
    pub solution_norm_squared: f64,
    /// `η = ‖e‖ / sqrt(‖u_h‖² + ‖e‖²)`: the estimated error in the energy norm as a fraction of
    /// the exact solution's, in `[0, 1)`. An estimate of the order of the energy error, not of
    /// the peak stress's accuracy; the module header says how it has read against the true error.
    pub relative_error: f64,
}

/// A resumable error estimate over a finished [`Recovery`], which it holds so a host with no
/// threads can keep both between frames; [`ErrorEstimation::into_recovery`] gives the recovery
/// back for [`crate::StaticStudy::finish_recovery`].
#[derive(Debug, Clone)]
pub struct ErrorEstimation {
    recovery: Recovery,
    next: usize,
    element_errors: Vec<f64>,
    error_sum: f64,
    energy_sum: f64,
}

impl ErrorEstimation {
    /// An estimate of `recovery` over a mesh of `element_count` elements, none done yet.
    pub(crate) fn new(recovery: Recovery, element_count: usize) -> Self {
        ErrorEstimation {
            recovery,
            next: 0,
            element_errors: Vec::with_capacity(element_count),
            error_sum: 0.0,
            energy_sum: 0.0,
        }
    }

    /// How many elements have been processed.
    pub fn elements_done(&self) -> usize {
        self.next
    }

    /// The recovery being measured.
    pub fn recovery(&self) -> &Recovery {
        &self.recovery
    }

    /// The recovery, to finish into a field.
    pub fn into_recovery(self) -> Recovery {
        self.recovery
    }

    /// Does up to `max_elements` more elements, reading the cancel check once per element;
    /// `Ok(true)` once every element is done. Refuses a recovery whose node, vertex or element
    /// counts differ from `mesh`'s, one with work left, and one begun before the solve converged.
    pub(crate) fn advance(
        &mut self,
        mesh: &QuadraticTetMesh,
        field: &impl MaterialField,
        max_elements: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<bool, ModelError> {
        let element_count = mesh.tets().len();
        // With these counts equal every corner index of `mesh` is a vertex of the recovery and
        // every node index one of its displacements, so nothing below can index out of range.
        let counts = (mesh.nodes().len(), mesh.vertex_count(), element_count);
        if self.recovery.mesh_counts() != counts {
            return Err(ModelError::RecoveryMeshMismatch);
        }
        if !self.recovery.is_finished(element_count) || !self.recovery.is_converged() {
            return Err(ModelError::NotFinished);
        }
        let mut budget = max_elements;
        while self.next < element_count && budget > 0 {
            budget -= 1;
            if cancel() {
                return Err(ModelError::Cancelled);
            }
            let (error, energy) = self.element_integrals(mesh, field, self.next)?;
            self.element_errors.push(error.sqrt());
            self.error_sum += error;
            self.energy_sum += energy;
            self.next += 1;
        }
        Ok(self.next == element_count)
    }

    /// The finished estimate; refuses one with elements left (or of an unconverged solve, which
    /// never advances), sums that are not finite, or a solution with no strain energy.
    pub(crate) fn estimate(&self, element_count: usize) -> Result<ErrorEstimate, ModelError> {
        if self.next != element_count || !self.recovery.is_converged() {
            return Err(ModelError::NotFinished);
        }
        if !self.energy_sum.is_finite() || !self.error_sum.is_finite() {
            return Err(ModelError::NonFiniteEnergy);
        }
        if self.energy_sum <= 0.0 {
            return Err(ModelError::ZeroStrainEnergy);
        }
        Ok(ErrorEstimate {
            element_errors: self.element_errors.clone(),
            error_norm_squared: self.error_sum,
            solution_norm_squared: self.energy_sum,
            relative_error: (self.error_sum / (self.energy_sum + self.error_sum)).sqrt(),
        })
    }

    /// One element's `(e_K², ‖u_h‖²_K)`.
    fn element_integrals(
        &self,
        mesh: &QuadraticTetMesh,
        field: &impl MaterialField,
        tet_index: usize,
    ) -> Result<(f64, f64), ModelError> {
        let element_error = |fault| ModelError::Element {
            tet: tet_index as u32,
            fault,
        };
        let tet = &mesh.tets()[tet_index];
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let displacements = self.recovery.displacements();
        let u = tet.map(|n| displacements[n as usize]);
        let recovered = self.recovered_at_nodes(mesh, tet)?;
        let mut error = 0.0;
        let mut energy = 0.0;
        for (index, q) in RULE.points().iter().enumerate() {
            let geometry = point_geometry(&nodes, q.natural, index).map_err(element_error)?;
            let material =
                checked_material(field, geometry.position, index).map_err(element_error)?;
            let stress = stress_at(&material, &strain_at(&geometry, &u));
            let shape = shape_values(q.natural);
            let mut difference = [0.0; 6];
            for (weight, value) in shape.iter().zip(&recovered) {
                for (out, component) in difference.iter_mut().zip(value) {
                    *out += weight * component;
                }
            }
            for (out, element) in difference.iter_mut().zip(&stress) {
                *out -= element;
            }
            let measure = q.weight * geometry.det_j / 6.0;
            error += measure * compliance_energy(&material, &difference);
            energy += measure * compliance_energy(&material, &stress);
        }
        Ok((error, energy))
    }

    /// The recovered stress at each of the element's nodes, as `results.rs` defines it.
    fn recovered_at_nodes(
        &self,
        mesh: &QuadraticTetMesh,
        tet: &[u32; NODE_COUNT],
    ) -> Result<[[f64; 6]; NODE_COUNT], ModelError> {
        let fits = self.recovery.fits();
        let fit = |vertex: u32| {
            fits[vertex as usize].as_ref().ok_or_else(|| {
                if self.recovery.is_mixed_material(vertex as usize) {
                    ModelError::MixedMaterialPatch { vertex }
                } else {
                    ModelError::CoplanarPatch { vertex }
                }
            })
        };
        let mut values = [[0.0; 6]; NODE_COUNT];
        for corner in 0..4 {
            values[corner] = fit(tet[corner])?.at(mesh.nodes()[tet[corner] as usize]);
        }
        for (slot, (a, b)) in EDGES.iter().enumerate() {
            let midside = mesh.nodes()[tet[4 + slot] as usize];
            values[4 + slot] = midside_mean(fit(tet[*a])?, fit(tet[*b])?, midside);
        }
        Ok(values)
    }
}

/// `s : C⁻¹ : s` for an isotropic material and a stress `[xx, yy, zz, xy, yz, zx]`: the strain
/// `C⁻¹ s` (engineering shears) dotted with the stress, written out so it is one fixed expression.
fn compliance_energy(material: &Material, s: &[f64; 6]) -> f64 {
    let nu = material.poisson_ratio;
    let normal = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
    let coupling = s[0] * s[1] + s[1] * s[2] + s[2] * s[0];
    let shear = s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
    (normal - 2.0 * nu * coupling + 2.0 * (1.0 + nu) * shear) / material.youngs_modulus
}
