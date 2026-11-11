//! Comparing two solves of one part and one load on a coarser and a finer mesh: how much the peak
//! von Mises stress, and the safety factor read from it, moved between them. The host meshes the
//! part twice and applies its supports and loads to each mesh itself; this module only reads two
//! finished [`Solution`]s, through the one peak a caller may report ([`peak_stress`] in
//! `results.rs`).
//!
//! **What the comparison is.** Two meshes show whether the number is still moving with
//! refinement. They do not bound its error: a small change says the peak has stopped moving at
//! these sizes, not that it is within that fraction of the true stress.
//!
//! **Where it never settles.** At a sharp inside corner, or at the edge of a clamp where a held
//! face meets a free one, the model's stress is unbounded. The peak there keeps rising as the
//! elements shrink, so its change between meshes does not fall toward zero. On a 4 x 1 x 1
//! cantilever clamped at one end (`crates/solidfem/tests/validation/comparison.rs`), read
//! including the clamp, the peak rose by 12.6%, 9.6% and 19.7% over three successive halvings of
//! the element size (24 to 12,288 elements); read from two depths beyond the clamp, it moved by
//! -7.5%, -2.0% and -0.49%, falling by four per halving toward beam theory's value. Leaving such a
//! spot out of the region read lets the rest settle, but the stress there then goes unassessed.
//! A clamp's edge is an artefact of an idealised support, but at a sharp inside corner of a real
//! part the high stress is real: the model needs the fillet the part has, or should have, for
//! that stress to be measured.
//!
//! **Meshes too alike.** With quadratic elements the peak's error falls as h^2, so refining the
//! element edge by `r` moves the peak by `1 - 1/r^2` of the coarse mesh's error: 0.75 of it for a
//! halving (8 times the elements), 0.60 at 4 times the elements and 0.37 at twice the elements.
//! On the cantilever above, read away from the clamp, the 1,536-element mesh (16 x 4 x 4 cells)
//! read 0.66% above beam theory; halving its element size moved the peak 0.49%, and 3,000
//! elements (20 x 5 x 5 cells, under twice the elements in the region) moved it 0.24%. Below 4
//! times the elements a small change mostly reflects the small step, so
//! [`MeshComparison::verdict`] reads [`Verdict::MeshesTooAlike`] whenever the refinement is under
//! [`MIN_ELEMENT_RATIO`], whatever the change.
//!
//! **Where the refinement is measured.** The verdict reads the refinement where each peak is,
//! [`MeshComparison::peak_element_ratio`]: the volume of the coarse peak's element over the fine
//! peak's, which in three dimensions is the ratio of element counts at that place. A ratio over a
//! whole region can be dodged: a host that holds a fillet at a floor size and halves only the
//! global size gets many more elements in the region, but the same elements at the peak, a
//! change of about zero, and a false "settled". The element counts in the region
//! ([`MeshComparison::element_ratio`]) are kept as information and for refusing a fine mesh that
//! is not finer in the region at all. A volume ratio counts a thinning in one direction as fully
//! as one in all three, so it cannot tell an element thinned along the stress gradient from one
//! thinned across it.
//!
//! The tolerance for "settled" is the host's: it chooses one and words the result.

use solidfem_mesh::{tet_volume, QuadraticTetMesh};

use crate::error::{ComparedMesh, ModelError};
use crate::results::{peak_with_holder, PeakHolder, PeakStress, Solution};

/// The smallest [`MeshComparison::peak_element_ratio`] at which a small change in the peak can be
/// read as the peak having settled: where the elements at the peak are refined in all three
/// directions, 4 times the elements there moves it by 0.60 of the coarse mesh's error, as
/// quadratic elements' error falls as h^2. A thinning in one direction reaches the same ratio
/// with less refinement (the module header's "Where the refinement is measured").
pub const MIN_ELEMENT_RATIO: f64 = 4.0;

/// The peak stress of one solve, how many of its elements the region holds, and the size of the
/// elements at the peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparedPeak {
    /// The peak von Mises stress in the region and where it is.
    pub peak: PeakStress,
    /// Elements with at least one integration point inside the region.
    pub elements: usize,
    /// The volume of the element holding the peak: its own element for an integration point,
    /// the mean volume of the elements that hold the node for a node.
    pub peak_element_volume: f64,
}

/// What a comparison says at the host's tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The peak moved by no more than the tolerance between meshes far enough apart.
    Settled,
    /// The peak moved by more than the tolerance: it is still changing with refinement.
    StillMoving,
    /// The fine mesh's elements at its peak are under [`MIN_ELEMENT_RATIO`] times smaller in
    /// volume than the coarse mesh's at its peak, so the change says little either way: refine
    /// the fine mesh further where the peak is.
    MeshesTooAlike,
}

/// How much the peak stress moved between a coarser and a finer solve of the same part and load.
/// Built only by [`MeshComparison::between`], so both peaks are finite and positive and the fine
/// mesh has more elements in the region than the coarse one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshComparison {
    coarse: ComparedPeak,
    fine: ComparedPeak,
    peak_change: f64,
    peak_distance: f64,
}

impl MeshComparison {
    /// Compares the peaks over the points `inside` accepts of two converged solves: `coarse` on
    /// `coarse_mesh` and `fine` on `fine_mesh`. Refused when a solution's arrays do not match its
    /// mesh, no integration point of one mesh is inside the region, a peak is not a finite
    /// number or is exactly zero, or the fine mesh has no more elements in the region than the
    /// coarse one. Only an exact zero is caught as unstressed: a region at rounding-level stress
    /// passes, and reads a huge safety factor.
    pub fn between(
        coarse_mesh: &QuadraticTetMesh,
        coarse: &Solution,
        fine_mesh: &QuadraticTetMesh,
        fine: &Solution,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Result<MeshComparison, ModelError> {
        check_matches(coarse_mesh, coarse, ComparedMesh::Coarse)?;
        check_matches(fine_mesh, fine, ComparedMesh::Fine)?;
        let (coarse_peak, coarse_holder) =
            region_peak(coarse_mesh, coarse, &inside, ComparedMesh::Coarse)?;
        let (fine_peak, fine_holder) = region_peak(fine_mesh, fine, &inside, ComparedMesh::Fine)?;
        let coarse_elements = elements_inside(coarse, &inside);
        let fine_elements = elements_inside(fine, &inside);
        if fine_elements <= coarse_elements {
            return Err(ModelError::FineMeshNotFiner {
                coarse_elements,
                fine_elements,
            });
        }
        let peak_change = (fine_peak.von_mises - coarse_peak.von_mises) / fine_peak.von_mises;
        let mut squared = 0.0;
        for axis in 0..3 {
            let d = fine_peak.position[axis] - coarse_peak.position[axis];
            squared += d * d;
        }
        Ok(MeshComparison {
            coarse: ComparedPeak {
                peak: coarse_peak,
                elements: coarse_elements,
                peak_element_volume: holder_volume(coarse_mesh, coarse, coarse_holder),
            },
            fine: ComparedPeak {
                peak: fine_peak,
                elements: fine_elements,
                peak_element_volume: holder_volume(fine_mesh, fine, fine_holder),
            },
            peak_change,
            peak_distance: squared.sqrt(),
        })
    }

    /// The coarse solve's peak and its elements in the region.
    pub fn coarse(&self) -> ComparedPeak {
        self.coarse
    }

    /// The fine solve's peak and its elements in the region.
    pub fn fine(&self) -> ComparedPeak {
        self.fine
    }

    /// Fine over coarse elements in the region read: information, and the measure
    /// [`ModelError::FineMeshNotFiner`] refuses on. Above 1 by construction. The verdict reads
    /// [`Self::peak_element_ratio`] instead, because a region can gain elements away from its
    /// peak.
    pub fn element_ratio(&self) -> f64 {
        // Element counts are far below 2^53, so both convert exactly.
        self.fine.elements as f64 / self.coarse.elements as f64
    }

    /// The coarse peak's element volume over the fine peak's: the refinement where each peak
    /// is, in three dimensions the ratio of element counts there. [`Self::verdict`] reads this.
    pub fn peak_element_ratio(&self) -> f64 {
        self.coarse.peak_element_volume / self.fine.peak_element_volume
    }

    /// `(fine - coarse) / fine` of the peak von Mises stress, as a signed fraction: positive when
    /// the finer mesh found more stress.
    pub fn peak_change(&self) -> f64 {
        self.peak_change
    }

    /// The distance between the two peaks' positions, in the mesh's length unit. It means
    /// something only where the peak is a distinct maximum: where several places hold nearly the
    /// same stress (uniform tension, a symmetric part) the reported position jumps between them on
    /// rounding (a bar in uniform tension reads 1.334 with nothing moved), so a host must not word
    /// it as "the peak moved".
    pub fn peak_distance(&self) -> f64 {
        self.peak_distance
    }

    /// `(coarse, fine)` safety factors, each `yield_strength / peak`. Refused for a yield
    /// strength that is not finite and positive.
    pub fn safety_factors(&self, yield_strength: f64) -> Result<(f64, f64), ModelError> {
        if !(yield_strength.is_finite() && yield_strength > 0.0) {
            return Err(ModelError::InvalidYieldStrength { yield_strength });
        }
        Ok((
            yield_strength / self.coarse.peak.von_mises,
            yield_strength / self.fine.peak.von_mises,
        ))
    }

    /// `(fine - coarse) / fine` of the safety factor, as a signed fraction: negative when the
    /// finer mesh found more stress and so a smaller margin. Refused as
    /// [`Self::safety_factors`] is.
    pub fn safety_factor_change(&self, yield_strength: f64) -> Result<f64, ModelError> {
        let (coarse, fine) = self.safety_factors(yield_strength)?;
        Ok((fine - coarse) / fine)
    }

    /// What the comparison says at `tolerance`, a fraction: [`Verdict::MeshesTooAlike`] when
    /// [`Self::peak_element_ratio`] is under [`MIN_ELEMENT_RATIO`] (or not a number), whatever
    /// the change; otherwise [`Verdict::Settled`] when `|peak_change| <= tolerance` and
    /// [`Verdict::StillMoving`] when not. A NaN tolerance is never met.
    pub fn verdict(&self, tolerance: f64) -> Verdict {
        let ratio = self.peak_element_ratio();
        if ratio.is_nan() || ratio < MIN_ELEMENT_RATIO {
            Verdict::MeshesTooAlike
        } else if self.peak_change.abs() <= tolerance {
            Verdict::Settled
        } else {
            Verdict::StillMoving
        }
    }
}

/// Refuses a solution whose arrays are not one per node of `mesh`, or whose integration points
/// do not cover every element of it the same number of times in element order.
fn check_matches(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    which: ComparedMesh,
) -> Result<(), ModelError> {
    let nodes = mesh.nodes().len();
    let tets = mesh.tets().len();
    let per_node = solution.displacements.len() == nodes
        && solution.nodal_stress.len() == nodes
        && solution.nodal_von_mises.len() == nodes;
    let points = solution.point_stresses.len();
    let covers = tets > 0
        && points > 0
        && points.is_multiple_of(tets)
        && solution
            .point_stresses
            .iter()
            .enumerate()
            .all(|(index, point)| point.tet as usize == index / (points / tets));
    if per_node && covers {
        Ok(())
    } else {
        Err(ModelError::SolutionMeshMismatch { mesh: which })
    }
}

/// Elements with at least one integration point inside the region. Points are stored in element
/// order (`check_matches`), so remembering the last element counted counts each one once.
fn elements_inside(solution: &Solution, inside: &impl Fn([f64; 3]) -> bool) -> usize {
    let mut count = 0;
    let mut last = None;
    for point in &solution.point_stresses {
        if last != Some(point.tet) && inside(point.position) {
            count += 1;
            last = Some(point.tet);
        }
    }
    count
}

/// The volume of the element holding a peak: an integration point's own element, or the mean
/// volume of the elements that hold a node, in element order. Elements have straight edges, so
/// the volume is the corner tetrahedron's.
fn holder_volume(mesh: &QuadraticTetMesh, solution: &Solution, holder: PeakHolder) -> f64 {
    let volume = |tet: &[u32; 10]| {
        let corner = |local: usize| mesh.nodes()[tet[local] as usize];
        tet_volume(corner(0), corner(1), corner(2), corner(3))
    };
    match holder {
        // `check_matches` has refused a point naming an element outside the mesh.
        PeakHolder::Point(index) => {
            volume(&mesh.tets()[solution.point_stresses[index].tet as usize])
        }
        PeakHolder::Node(node) => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for tet in mesh.tets() {
                if tet.contains(&(node as u32)) {
                    sum += volume(tet);
                    count += 1;
                }
            }
            // A node of no element gives NaN, which the verdict reads as too alike.
            sum / count as f64
        }
    }
}

/// The peak over `inside` and what holds it, refused when no integration point is inside (a
/// face or line holds no element volume to read, however fine the mesh) or the peak is not
/// finite or is zero.
fn region_peak(
    mesh: &QuadraticTetMesh,
    solution: &Solution,
    inside: &impl Fn([f64; 3]) -> bool,
    which: ComparedMesh,
) -> Result<(PeakStress, PeakHolder), ModelError> {
    if !solution
        .point_stresses
        .iter()
        .any(|point| inside(point.position))
    {
        return Err(ModelError::NothingInsideRegion { mesh: which });
    }
    // An integration point is inside, so a peak exists; `None` cannot reach here, and is mapped
    // to the same refusal rather than unwrapped.
    let (peak, holder) = peak_with_holder(
        mesh,
        &solution.nodal_von_mises,
        &solution.point_stresses,
        inside,
    )
    .ok_or(ModelError::NothingInsideRegion { mesh: which })?;
    if !peak.von_mises.is_finite() {
        Err(ModelError::NonFinitePeak {
            mesh: which,
            von_mises: peak.von_mises,
        })
    } else if peak.von_mises <= 0.0 {
        Err(ModelError::ZeroPeak { mesh: which })
    } else {
        Ok((peak, holder))
    }
}
