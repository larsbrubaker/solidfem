//! Recovery after a solve. Integration point values are the raw results; [`Solution::nodal_stress`]
//! is the one nodal stress, and everything that reads a stress at a node (`nodal_von_mises`, a
//! peak, a safety factor) reads this field.
//!
//! **How a node gets its stress.** Stress is discontinuous between elements and is most accurate
//! at the integration points, so a node's value is read from a least-squares patch fit
//! (`patch.rs`): around each mesh vertex a linear tensor field is fitted to the integration points
//! of the elements touching it, and the fit is evaluated at the vertex. A midside node takes the
//! mean of the fits from its edge's two vertices, evaluated at its own position, so corner and
//! midside values come from one field. Reading a node from each element's own extrapolation and
//! averaging is unsafe at a midside node: a midside on a free edge can belong to a single
//! element, so there is nothing to average, and the value is whatever that element's linear field
//! reads at its edge. The fit is of the tensor, never of von Mises, because von Mises is not
//! linear: a smoothed scalar is not the stress of any smoothed state.
//!
//! **Boundary nodes.** A patch at the surface is one-sided. A linear fit needs no special case
//! there: any patch holding one non-flat element determines it, and it is exact for a linear
//! stress field, which is the bending state a surface node usually sees. A quadratic fit, which
//! would address the curvature a one-sided patch cannot see, was measured and rejected: on the
//! plate-with-a-hole benchmark it read 18% high at the coarsest mesh and left some patches
//! undetermined.
//!
//! **What the field reads at a peak.** A stress concentration's fall-off is convex, so a linear
//! fit reads low by the curvature it cannot represent: against Lamé at a ring's bore
//! (`crates/solidfem/tests/validation/recovery.rs`) the bore's interior nodes read 0.2-1.3% low,
//! the error shrinking as the mesh refines. At a sharp re-entrant corner the patch wraps round
//! the corner and one linear tensor is fitted to both legs, so the fit reads far below the stress
//! the elements there computed: on a refined L angle by up to a third.
//!
//! **The peak.** [`peak_von_mises`] (and [`peak_stress`], the same peak with its position) is the
//! one peak a caller may report: over a region, the
//! larger of the recovered node values and the integration point values, each classified by its
//! own physical position. A reported peak can therefore never read below a stress the solver
//! computed in that region. The floor is applied to the peak and not to the field: a node floored
//! at its elements' integration points would invent stress at a bending neutral axis, so the
//! field and its colours stay on the fit. At a sharp corner the peak is the largest integration
//! point and keeps growing with refinement, which is the singularity showing; it is not a usable
//! stress. On the smooth cases measured the peak was above the analytic value on the ring's bore
//! (+0.9% to +1.1%, from nodes at the bore's edges, where a patch is one-sided in two directions)
//! and at a Kirsch hole (+0.3% to +0.6%), and within 1.2% either way at a real plate's hole. A
//! design margin should allow for that discretization error.
//!
//! **Mixed material.** Stress jumps across a material interface, so a patch holding more than one
//! material is not fitted; a node there reports the largest von Mises any of its elements gives at
//! the node (with its own material, not the far side's), so an interface can only read high. A
//! patch whose points are all coplanar falls back the same way.
//!
//! [`Recovery`] is the resumable form: a host with no threads runs the elements and then the patch
//! fits in bounded slices, in a fixed order, so slicing never changes a value. A finished one can
//! also be measured for how converged it is (`error_estimate.rs`) before it becomes a field.
//! [`nodal_rotations`] gives the local rotation of a displacement field, for drawing a deformed
//! shape that keeps its rotations rigid.

use solidfem_mesh::QuadraticTetMesh;

use crate::element::{
    checked_material, point_geometry, rotation_at, strain_at, stress_at, von_mises,
};
use crate::error::ModelError;
use crate::material::{Material, MaterialField};
use crate::patch::PatchFit;
use crate::quadrature::{Quadrature, QuadraturePoint};
use crate::shape::{EDGES, NODE_COUNT, NODE_NATURAL};
use crate::study::Report;

/// Stress at one integration point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointStress {
    /// The element.
    pub tet: u32,
    /// The integration point within it.
    pub point: u8,
    /// Physical position.
    pub position: [f64; 3],
    /// `[xx, yy, zz, xy, yz, zx]`.
    pub stress: [f64; 6],
    /// Von Mises equivalent stress.
    pub von_mises: f64,
}

/// A solved study's results.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Displacement of every mesh node, zero at fixed nodes.
    pub displacements: Vec<[f64; 3]>,
    /// Stress at every integration point, in (element, point) order.
    pub point_stresses: Vec<PointStress>,
    /// The recovered stress tensor at every mesh node, `[xx, yy, zz, xy, yz, zx]`.
    pub nodal_stress: Vec<[f64; 6]>,
    /// Von Mises stress of [`Self::nodal_stress`], per mesh node: the display value. A reported
    /// peak reads [`Self::peak_von_mises`], which also holds the integration points.
    pub nodal_von_mises: Vec<f64>,
    /// Sizes, iteration counts and wall times.
    pub report: Report,
}

/// A resumable stress recovery over a snapshot of nodal displacements: every element in order,
/// then a patch fit per mesh vertex in order, so a recovery resumed in slices gives exactly the
/// bits of one run; [`StaticStudy`] drives it because it owns the mesh and material it reads.
///
/// [`StaticStudy`]: crate::study::StaticStudy
#[derive(Debug, Clone)]
pub struct Recovery {
    displacements: Vec<[f64; 3]>,
    point_stresses: Vec<PointStress>,
    point_materials: Vec<Material>,
    points_per_element: usize,
    /// Per mesh node, the stress of the incident element whose von Mises at the node is largest.
    worst_at_node: Vec<([f64; 6], f64)>,
    next: usize,
    /// One entry per mesh vertex: its fit, or `None` where a patch cannot be fitted.
    fits: Vec<Option<PatchFit>>,
    /// Per mesh vertex, whether its patch was left unfitted for holding more than one material
    /// (otherwise an unfitted patch's points were coplanar).
    mixed_material: Vec<bool>,
    /// Elements in the mesh this recovery was begun for.
    element_count: usize,
    /// Whether the displacements are a converged solve's, not a preview of one in progress.
    converged: bool,
    next_vertex: usize,
    /// Vertex patches: `incident[offsets[v]..offsets[v + 1]]` are the elements touching vertex `v`.
    offsets: Vec<usize>,
    incident: Vec<u32>,
}

/// Displacements and the stresses recovered from them: a live estimate while a solve runs, the
/// answer once it has converged.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveredField {
    /// Displacement of every mesh node, zero at fixed nodes.
    pub displacements: Vec<[f64; 3]>,
    /// Stress at every integration point, in (element, point) order.
    pub point_stresses: Vec<PointStress>,
    /// The recovered stress tensor at every mesh node, `[xx, yy, zz, xy, yz, zx]`.
    pub nodal_stress: Vec<[f64; 6]>,
    /// Von Mises stress of [`Self::nodal_stress`], per mesh node.
    pub nodal_von_mises: Vec<f64>,
}

/// A peak von Mises stress and where it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakStress {
    /// The von Mises stress at the peak.
    pub von_mises: f64,
    /// Physical position of the node or integration point that holds it.
    pub position: [f64; 3],
}

/// The peak von Mises stress over the points `inside` accepts: the larger of the recovered value
/// at the mesh nodes and the value at the integration points, each classified by its physical
/// position. `None` when neither a node nor a point is inside. A non-finite value is kept, never
/// dropped, so it cannot hide a broken result behind a finite peak. Read from any recovered
/// field; [`Solution::peak_von_mises`] and [`RecoveredField::peak_von_mises`] call this, and it
/// is [`peak_stress`]'s value, so the two never disagree.
pub fn peak_von_mises(
    mesh: &QuadraticTetMesh,
    nodal_von_mises: &[f64],
    point_stresses: &[PointStress],
    inside: impl Fn([f64; 3]) -> bool,
) -> Option<f64> {
    peak_stress(mesh, nodal_von_mises, point_stresses, inside).map(|peak| peak.von_mises)
}

/// [`peak_von_mises`] with the position that holds it. Candidates are taken in node order, then
/// in integration point order; a later candidate replaces the peak only when it is strictly
/// larger, so of equal values the first is reported. The first NaN met becomes the peak and stays
/// it, with its own position, so a broken value cannot be hidden behind a finite one.
pub fn peak_stress(
    mesh: &QuadraticTetMesh,
    nodal_von_mises: &[f64],
    point_stresses: &[PointStress],
    inside: impl Fn([f64; 3]) -> bool,
) -> Option<PeakStress> {
    peak_with_holder(mesh, nodal_von_mises, point_stresses, inside).map(|(peak, _)| peak)
}

/// What holds a peak: a mesh node or an integration point, by its index in its array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeakHolder {
    Node(usize),
    Point(usize),
}

/// [`peak_stress`] with what holds the peak, for a caller that needs the elements around it.
pub(crate) fn peak_with_holder(
    mesh: &QuadraticTetMesh,
    nodal_von_mises: &[f64],
    point_stresses: &[PointStress],
    inside: impl Fn([f64; 3]) -> bool,
) -> Option<(PeakStress, PeakHolder)> {
    let nodes = mesh
        .nodes()
        .iter()
        .zip(nodal_von_mises)
        .enumerate()
        .filter(|(_, (position, _))| inside(**position))
        .map(|(node, (position, value))| {
            let peak = PeakStress {
                von_mises: *value,
                position: *position,
            };
            (peak, PeakHolder::Node(node))
        });
    let points = point_stresses
        .iter()
        .enumerate()
        .filter(|(_, point)| inside(point.position))
        .map(|(index, point)| {
            let peak = PeakStress {
                von_mises: point.von_mises,
                position: point.position,
            };
            (peak, PeakHolder::Point(index))
        });
    nodes.chain(points).reduce(|peak, candidate| {
        if peak.0.von_mises.is_nan() {
            peak
        } else if candidate.0.von_mises.is_nan() || candidate.0.von_mises > peak.0.von_mises {
            candidate
        } else {
            peak
        }
    })
}

impl Solution {
    /// The peak von Mises stress over the points `inside` accepts; see [`peak_von_mises`].
    pub fn peak_von_mises(
        &self,
        mesh: &QuadraticTetMesh,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Option<f64> {
        peak_von_mises(mesh, &self.nodal_von_mises, &self.point_stresses, inside)
    }

    /// The peak von Mises stress over the points `inside` accepts, with its position; see
    /// [`peak_stress`].
    pub fn peak_stress(
        &self,
        mesh: &QuadraticTetMesh,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Option<PeakStress> {
        peak_stress(mesh, &self.nodal_von_mises, &self.point_stresses, inside)
    }
}

impl RecoveredField {
    /// The peak von Mises stress over the points `inside` accepts; see [`peak_von_mises`].
    pub fn peak_von_mises(
        &self,
        mesh: &QuadraticTetMesh,
        inside: impl Fn([f64; 3]) -> bool,
    ) -> Option<f64> {
        peak_von_mises(mesh, &self.nodal_von_mises, &self.point_stresses, inside)
    }
}

/// The local rotation vector (half the curl of the displacement), in radians, at every mesh node
/// of `displacements`.
///
/// Each element contributes the 10-node shape-function gradients evaluated at the node, and a node
/// shared by several elements takes the average weighted by each element's Jacobian determinant at
/// that node: on this mesh's straight-edged elements that is six times the element's volume.
/// Gradients are discontinuous between elements, so unlike the displacement itself this is a
/// smoothed value: exact on a rigid
/// rotation, an average where the rotation varies across the node. `displacements` must hold one
/// vector per mesh node.
pub fn nodal_rotations(
    mesh: &QuadraticTetMesh,
    displacements: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, ModelError> {
    let node_count = mesh.nodes().len();
    if displacements.len() != node_count {
        return Err(ModelError::DisplacementFieldLength {
            nodes: node_count,
            values: displacements.len(),
        });
    }
    let mut weighted = vec![[0.0f64; 3]; node_count];
    let mut weights = vec![0.0f64; node_count];
    for (tet_index, tet) in mesh.tets().iter().enumerate() {
        let nodes = tet.map(|n| mesh.nodes()[n as usize]);
        let u = tet.map(|n| displacements[n as usize]);
        for node in 0..NODE_COUNT {
            let geometry = point_geometry(&nodes, NODE_NATURAL[node], node).map_err(|fault| {
                ModelError::Element {
                    tet: tet_index as u32,
                    fault,
                }
            })?;
            let rotation = rotation_at(&geometry, &u);
            let global = tet[node] as usize;
            for axis in 0..3 {
                weighted[global][axis] += geometry.det_j * rotation[axis];
            }
            weights[global] += geometry.det_j;
        }
    }
    // Every mesh node belongs to a tetrahedron (`TetMesh::new`), so every weight is positive.
    Ok(weighted
        .iter()
        .zip(&weights)
        .map(|(sum, weight)| sum.map(|c| c / weight))
        .collect())
}

impl Recovery {
    /// A recovery of `displacements` over a mesh of `element_count` elements, none done yet;
    /// `converged` says whether they are a converged solve's.
    pub(crate) fn new(
        displacements: Vec<[f64; 3]>,
        node_count: usize,
        vertex_count: usize,
        element_count: usize,
        points_per_element: usize,
        converged: bool,
    ) -> Self {
        Recovery {
            displacements,
            point_stresses: Vec::with_capacity(element_count * points_per_element),
            point_materials: Vec::with_capacity(element_count * points_per_element),
            points_per_element,
            worst_at_node: vec![([0.0; 6], f64::MIN); node_count],
            next: 0,
            fits: vec![None; vertex_count],
            mixed_material: vec![false; vertex_count],
            element_count,
            converged,
            next_vertex: 0,
            offsets: Vec::new(),
            incident: Vec::new(),
        }
    }

    /// How many elements have been processed.
    pub fn elements_done(&self) -> usize {
        self.next
    }

    /// The displacements being recovered.
    pub fn displacements(&self) -> &[[f64; 3]] {
        &self.displacements
    }

    /// Each mesh vertex's patch fit, `None` where its patch was not fitted.
    pub(crate) fn fits(&self) -> &[Option<PatchFit>] {
        &self.fits
    }

    /// Whether `vertex`'s patch was left unfitted for holding more than one material.
    pub(crate) fn is_mixed_material(&self, vertex: usize) -> bool {
        self.mixed_material[vertex]
    }

    /// The node, vertex and element counts of the mesh this recovery was begun for.
    pub(crate) fn mesh_counts(&self) -> (usize, usize, usize) {
        (
            self.displacements.len(),
            self.fits.len(),
            self.element_count,
        )
    }

    /// Whether the displacements are a converged solve's rather than a preview's.
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Whether every element and every patch fit is done, so the field can be read.
    pub(crate) fn is_finished(&self, element_count: usize) -> bool {
        self.next == element_count && self.next_vertex == self.fits.len()
    }

    /// The finished field. Every mesh node belongs to a tetrahedron (`TetMesh::new`), so every
    /// node has a fit or a worst incident value.
    pub(crate) fn into_field(self, mesh: &QuadraticTetMesh) -> RecoveredField {
        let nodes = mesh.nodes();
        let vertices = self.fits.len();
        let mut nodal_stress = vec![[0.0; 6]; nodes.len()];
        for (vertex, fit) in self.fits.iter().enumerate() {
            nodal_stress[vertex] = match fit {
                Some(fit) => fit.at(nodes[vertex]),
                None => self.worst_at_node[vertex].0,
            };
        }
        // A midside takes the two end patches' mean at its own position; a shared midside is
        // written once, from the first element that reaches it, which fixes the order.
        let mut midside_done = vec![false; nodes.len().saturating_sub(vertices)];
        for tet in mesh.tets() {
            for (slot, (a, b)) in EDGES.iter().enumerate() {
                let midside = tet[4 + slot] as usize;
                if midside_done[midside - vertices] {
                    continue;
                }
                midside_done[midside - vertices] = true;
                let ends = (&self.fits[tet[*a] as usize], &self.fits[tet[*b] as usize]);
                nodal_stress[midside] = match ends {
                    (Some(first), Some(second)) => midside_mean(first, second, nodes[midside]),
                    _ => self.worst_at_node[midside].0,
                };
            }
        }
        let nodal_von_mises = nodal_stress.iter().map(von_mises).collect();
        RecoveredField {
            displacements: self.displacements,
            point_stresses: self.point_stresses,
            nodal_stress,
            nodal_von_mises,
        }
    }

    /// The elements touching each vertex, in element order, as a compressed list. One pass over
    /// the tetrahedra's four corners, comparable to a single slice of element recovery.
    fn build_patches(&mut self, mesh: &QuadraticTetMesh) {
        let mut counts = vec![0usize; self.fits.len() + 1];
        for tet in mesh.tets() {
            for corner in &tet[..4] {
                counts[*corner as usize + 1] += 1;
            }
        }
        for index in 1..counts.len() {
            counts[index] += counts[index - 1];
        }
        let mut next = counts.clone();
        let mut incident = vec![0u32; counts[self.fits.len()]];
        for (tet_index, tet) in mesh.tets().iter().enumerate() {
            for corner in &tet[..4] {
                let slot = &mut next[*corner as usize];
                incident[*slot] = tet_index as u32;
                *slot += 1;
            }
        }
        self.offsets = counts;
        self.incident = incident;
    }

    /// Fits the patch at the next vertex; a patch of more than one material is left unfitted.
    fn fit_next_vertex(&mut self, mesh: &QuadraticTetMesh) {
        let vertex = self.next_vertex;
        let elements = &self.incident[self.offsets[vertex]..self.offsets[vertex + 1]];
        let mut samples = Vec::with_capacity(elements.len() * self.points_per_element);
        let mut first_material = None;
        let mut uniform = true;
        for element in elements {
            let start = *element as usize * self.points_per_element;
            for point in start..start + self.points_per_element {
                let material = self.point_materials[point];
                match first_material {
                    None => first_material = Some(material),
                    Some(m) if same_bits(m, material) => {}
                    Some(_) => uniform = false,
                }
                samples.push((
                    self.point_stresses[point].position,
                    self.point_stresses[point].stress,
                ));
            }
        }
        self.fits[vertex] = if uniform {
            PatchFit::fit(mesh.nodes()[vertex], &samples)
        } else {
            None
        };
        self.mixed_material[vertex] = !uniform;
        self.next_vertex += 1;
    }

    /// Does up to `max_work` more units of work (an element, then a patch fit), reading the cancel
    /// check once per unit.
    pub(crate) fn advance(
        &mut self,
        mesh: &QuadraticTetMesh,
        field: &impl MaterialField,
        rule: Quadrature,
        max_work: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<(), ModelError> {
        let element_error = |tet: usize, fault| ModelError::Element {
            tet: tet as u32,
            fault,
        };
        let points = rule.points();
        let nearest_point = NODE_NATURAL.map(|node| nearest_natural_point(points, node));
        let mut budget = max_work;
        while self.next < mesh.tets().len() && budget > 0 {
            budget -= 1;
            let tet_index = self.next;
            if cancel() {
                return Err(ModelError::Cancelled);
            }
            let tet = &mesh.tets()[tet_index];
            let nodes = tet.map(|n| mesh.nodes()[n as usize]);
            let u = tet.map(|n| self.displacements[n as usize]);
            for (index, q) in points.iter().enumerate() {
                let geometry = point_geometry(&nodes, q.natural, index)
                    .map_err(|f| element_error(tet_index, f))?;
                let material = checked_material(field, geometry.position, index)
                    .map_err(|f| element_error(tet_index, f))?;
                let stress = stress_at(&material, &strain_at(&geometry, &u));
                self.point_materials.push(material);
                self.point_stresses.push(PointStress {
                    tet: tet_index as u32,
                    point: index as u8,
                    position: geometry.position,
                    stress,
                    von_mises: von_mises(&stress),
                });
            }
            let element_materials =
                &self.point_materials[self.point_materials.len() - points.len()..];
            for node in 0..NODE_COUNT {
                let geometry = point_geometry(&nodes, NODE_NATURAL[node], node)
                    .map_err(|f| element_error(tet_index, f))?;
                // The element's own material: a node on a material interface would otherwise
                // read the far side's stiffness against this element's strain.
                let material = element_materials[nearest_point[node]];
                let stress = stress_at(&material, &strain_at(&geometry, &u));
                let vm = von_mises(&stress);
                let worst = &mut self.worst_at_node[tet[node] as usize];
                if vm > worst.1 {
                    *worst = (stress, vm);
                }
            }
            self.next += 1;
        }
        if self.next == mesh.tets().len() && self.next_vertex < self.fits.len() && budget > 0 {
            if self.offsets.is_empty() {
                self.build_patches(mesh);
            }
            while self.next_vertex < self.fits.len() && budget > 0 {
                budget -= 1;
                if cancel() {
                    return Err(ModelError::Cancelled);
                }
                self.fit_next_vertex(mesh);
            }
        }
        Ok(())
    }
}

/// A midside node's recovered stress: the mean of its edge's two vertex fits at its own position.
/// The sum is commutative in f64, so either end order gives the same bits.
pub(crate) fn midside_mean(first: &PatchFit, second: &PatchFit, at: [f64; 3]) -> [f64; 6] {
    let (p, q) = (first.at(at), second.at(at));
    let mut mean = [0.0; 6];
    for k in 0..6 {
        mean[k] = 0.5 * (p[k] + q[k]);
    }
    mean
}

/// The integration point of a rule nearest a node in natural coordinates; a tie takes the earlier
/// point, so the choice is fixed.
fn nearest_natural_point(points: &[QuadraturePoint], node: [f64; 3]) -> usize {
    let mut best = (f64::MAX, 0);
    for (index, point) in points.iter().enumerate() {
        let distance: f64 = point
            .natural
            .iter()
            .zip(node.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if distance < best.0 {
            best = (distance, index);
        }
    }
    best.1
}

/// Whether two materials are the same bits: the patch-uniformity question, so any variation at all
/// in a graded material counts as a patch to leave unfitted.
fn same_bits(a: Material, b: Material) -> bool {
    a.youngs_modulus.to_bits() == b.youngs_modulus.to_bits()
        && a.poisson_ratio.to_bits() == b.poisson_ratio.to_bits()
}
