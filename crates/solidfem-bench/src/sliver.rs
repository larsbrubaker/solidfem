//! How much a bad boundary triangle costs a 10-node stress result: a known cantilever with chosen
//! surface triangles made sliver-shaped, each case solved and compared with the unperturbed bar.
//!
//! Construction: a [`LENGTH`] x [`DEPTH`] x [`DEPTH`] bar from `box_tet_mesh` (cells a quarter of
//! the depth), upgraded to straight-edged 10-node tetrahedra. Boundary nodes are moved only within
//! their face's plane, so the part's shape, supports and loads do not change and the unperturbed
//! solve stays the exact reference. A [`Shape::Cap`] pulls a triangle's right-angle corner toward
//! the midpoint of its long edge (one triangle, two small angles and one near 180 degrees).
//! A [`Shape::Needle`] shortens one edge (necessarily the two triangles sharing it, since an edge
//! of a closed surface always has two). A cluster repeats the move at ten nodes that share no
//! edge, so each bad triangle keeps the construction above and they stay comparable to the single
//! case.
//!
//! What this is not: the rest of each tet stays where the box put it, so the solid under a bad
//! triangle is a sliver a real mesher would likely not produce in the same shape. A real mesher
//! decides where those nodes go and curves nothing here; these elements are straight-sided, so
//! their Jacobian is constant inside each element and the within-element ratio is 1 by
//! construction (it is measured, and the scaled Jacobian is what actually varies).
//!
//! `src/bin/sliver.rs` runs every case and prints the table.

use solidfem_elastic::{
    point_geometry, shape_values, strain_at, stress_at, von_mises, Material, Model, ModelError,
    Solution, StaticStudy, StepOutcome, StudyOptions, NODE_NATURAL,
};
use solidfem_mesh::{box_tet_mesh, tet_volume, MeshError, QuadraticTetMesh, TetMesh};

/// Bar length.
pub const LENGTH: f64 = 10.0;
/// Section side.
pub const DEPTH: f64 = 1.0;
/// Cells along the length, the width and the depth: the cantilever validation's finest mesh.
pub const DIVISIONS: [usize; 3] = [40, 4, 4];
/// Total transverse tip force.
pub const FORCE: f64 = 1.0;
/// Young's modulus.
pub const YOUNGS: f64 = 1000.0;

/// How the bad triangle is bad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Two small angles and one near 180 degrees: one vertex near the middle of the opposite edge.
    Cap,
    /// One short edge: a small angle at the opposite vertex, on both triangles sharing the edge.
    Needle,
}

/// Where on the part the bad triangles are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// The outer tensile face, next to the clamped root, where bending stress is highest.
    NearRoot,
    /// The side face at mid-span on the neutral axis, where the normal stress is near zero.
    NeutralAxis,
}

/// How many bad triangles a case has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    /// One moved node.
    Single,
    /// Ten moved nodes that share no edge.
    Cluster,
}

/// One case: what kind of bad triangle, where, how many and how bad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spec {
    /// The triangle's shape.
    pub shape: Shape,
    /// Where it is.
    pub location: Location,
    /// How many.
    pub extent: Extent,
    /// The smallest corner asked for, in degrees.
    pub target_degrees: f64,
}

/// The node index of grid point `(i, j, k)` in the box's x-then-y-then-z ordering.
fn grid_node(i: usize, j: usize, k: usize) -> u32 {
    let nx = DIVISIONS[0] + 1;
    let ny = DIVISIONS[1] + 1;
    (i + nx * (j + ny * k)) as u32
}

impl Location {
    /// A grid node in the location's face, from its in-plane indices.
    fn node(self, a: usize, b: usize) -> u32 {
        match self {
            Location::NearRoot => grid_node(a, b, DIVISIONS[2]),
            Location::NeutralAxis => grid_node(a, 0, b),
        }
    }

    /// The in-plane indices of the nodes to move. Every node moved here is strictly inside its
    /// face, so none lies on an edge of the part, a support or a loaded face. A near-root corner
    /// stays one cell off the clamp so the bad element is not also in the clamp's singularity.
    fn moved_nodes(self, extent: Extent) -> Vec<(usize, usize)> {
        match (self, extent) {
            (Location::NearRoot, Extent::Single) => vec![(2, 2)],
            (Location::NeutralAxis, Extent::Single) => vec![(20, 2)],
            (Location::NearRoot, Extent::Cluster) => spaced_nodes(2),
            (Location::NeutralAxis, Extent::Cluster) => spaced_nodes(16),
        }
    }
}

/// Five columns, two cells apart, at the two interior rows that are two cells apart: ten nodes,
/// no two of them joined by an edge.
fn spaced_nodes(first_column: usize) -> Vec<(usize, usize)> {
    let mut nodes = Vec::with_capacity(10);
    for row in [1, 3] {
        for column in 0..5 {
            nodes.push((first_column + 2 * column, row));
        }
    }
    nodes
}

/// A mesh with bad boundary triangles, and which triangles and nodes they are.
pub struct Distorted {
    /// The four-node mesh, validated with every tetrahedron positive.
    pub linear: TetMesh,
    /// The same mesh with midpoint nodes.
    pub quadratic: QuadraticTetMesh,
    /// The corner nodes of each bad boundary triangle.
    pub bad_triangles: Vec<[u32; 3]>,
    /// The nodes that were moved.
    pub moved: Vec<u32>,
}

/// The unperturbed bar.
pub fn unperturbed() -> Result<QuadraticTetMesh, MeshError> {
    Ok(QuadraticTetMesh::from_linear(&unperturbed_linear()?))
}

fn unperturbed_linear() -> Result<TetMesh, MeshError> {
    box_tet_mesh([0.0; 3], [LENGTH, DEPTH, DEPTH], DIVISIONS)
}

/// Builds the mesh of `spec`; `Err` when a tetrahedron would have no positive volume, which is the
/// answer "this angle cannot be built with this construction", reported rather than hidden.
pub fn build(spec: Spec) -> Result<Distorted, MeshError> {
    let base = unperturbed_linear()?;
    let mut positions = base.positions().to_vec();
    let mut bad_triangles = Vec::new();
    let mut moved = Vec::new();
    let tangent = libm::tan(spec.target_degrees.to_radians());
    for (a, b) in spec.location.moved_nodes(spec.extent) {
        let c = spec.location.node(a, b);
        let c_at = positions[c as usize];
        match spec.shape {
            Shape::Cap => {
                let from = spec.location.node(a - 1, b);
                let to = spec.location.node(a, b + 1);
                let mid = midpoint(positions[from as usize], positions[to as usize]);
                // The undistorted triangle is right-angled and isosceles, so its height over the
                // long edge equals half that edge and the small corners are 45 degrees: scaling
                // the height by tan(target) makes them the target.
                positions[c as usize] = along(mid, c_at, tangent);
                bad_triangles.push([from, to, c]);
            }
            Shape::Needle => {
                let near = spec.location.node(a, b - 1);
                let across = spec.location.node(a - 1, b - 1);
                let beyond = spec.location.node(a + 1, b);
                // Moving C toward B along the edge shortens it. The tightest corner is at the
                // vertex past the edge on the far side, whose tangent is s / (2 - s) of the move
                // fraction s, so s = 2 tan / (1 + tan) gives that corner the target.
                let fraction = 2.0 * tangent / (1.0 + tangent);
                positions[c as usize] = along(positions[near as usize], c_at, fraction);
                bad_triangles.push([across, near, c]);
                bad_triangles.push([near, beyond, c]);
            }
        }
        moved.push(c);
    }
    let linear = TetMesh::new(positions, base.tets().to_vec())?;
    let quadratic = QuadraticTetMesh::from_linear(&linear);
    Ok(Distorted {
        linear,
        quadratic,
        bad_triangles,
        moved,
    })
}

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

/// The point `fraction` of the way from `from` to `to`.
fn along(from: [f64; 3], to: [f64; 3], fraction: f64) -> [f64; 3] {
    [
        from[0] + fraction * (to[0] - from[0]),
        from[1] + fraction * (to[1] - from[1]),
        from[2] + fraction * (to[2] - from[2]),
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The angle between two vectors, in degrees.
fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    libm::atan2(length(cross(a, b)), dot(a, b)).to_degrees()
}

/// The smallest corner of a triangle, in degrees.
pub fn triangle_min_corner(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let at_a = angle_between(sub(b, a), sub(c, a));
    let at_b = angle_between(sub(a, b), sub(c, b));
    let at_c = angle_between(sub(a, c), sub(b, c));
    at_a.min(at_b).min(at_c)
}

/// The shape of what a case built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quality {
    /// The smallest corner of any boundary triangle, in degrees.
    pub min_corner: f64,
    /// Boundary triangles with a corner under 1.2 times the target.
    pub low_triangles: usize,
    /// The smallest interior dihedral of any tetrahedron, in degrees.
    pub min_dihedral: f64,
    /// The largest interior dihedral of any tetrahedron, in degrees.
    pub max_dihedral: f64,
    /// The smallest scaled Jacobian of any tetrahedron: 1 for a regular one, 0 for a flat one.
    pub scaled_jacobian: f64,
    /// The smallest ratio of an element's smallest to largest Jacobian determinant, over its nodes
    /// and integration points.
    pub jacobian_ratio: f64,
}

/// Measures the shape of a mesh; `target` sets the threshold for `low_triangles`.
pub fn measure_quality(linear: &TetMesh, quadratic: &QuadraticTetMesh, target: f64) -> Quality {
    let positions = linear.positions();
    let mut min_corner = f64::INFINITY;
    let mut low_triangles = 0;
    for face in quadratic.boundary_faces() {
        let [a, b, c] =
            [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| positions[n as usize]);
        let corner = triangle_min_corner(a, b, c);
        min_corner = min_corner.min(corner);
        if corner < 1.2 * target {
            low_triangles += 1;
        }
    }
    let mut min_dihedral = f64::INFINITY;
    let mut max_dihedral = f64::NEG_INFINITY;
    let mut scaled_jacobian = f64::INFINITY;
    for tet in linear.tets() {
        let corners = tet.map(|n| positions[n as usize]);
        let (low, high) = tet_dihedrals(&corners);
        min_dihedral = min_dihedral.min(low);
        max_dihedral = max_dihedral.max(high);
        scaled_jacobian = scaled_jacobian.min(tet_scaled_jacobian(&corners));
    }
    let mut jacobian_ratio = f64::INFINITY;
    for tet in quadratic.tets() {
        jacobian_ratio = jacobian_ratio.min(element_jacobian_ratio(
            &tet.map(|n| quadratic.nodes()[n as usize]),
        ));
    }
    Quality {
        min_corner,
        low_triangles,
        min_dihedral,
        max_dihedral,
        scaled_jacobian,
        jacobian_ratio,
    }
}

/// The smallest and largest interior dihedral of a tetrahedron, in degrees. Each edge's dihedral
/// is the angle between the other two corners seen along that edge.
fn tet_dihedrals(c: &[[f64; 3]; 4]) -> (f64, f64) {
    const EDGES: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ];
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for (a, b, p, q) in EDGES {
        let axis = sub(c[b], c[a]);
        let scale = dot(axis, axis);
        let across = |v: [f64; 3]| {
            let along_axis = dot(v, axis) / scale;
            [
                v[0] - along_axis * axis[0],
                v[1] - along_axis * axis[1],
                v[2] - along_axis * axis[2],
            ]
        };
        let angle = angle_between(across(sub(c[p], c[a])), across(sub(c[q], c[a])));
        low = low.min(angle);
        high = high.max(angle);
    }
    (low, high)
}

/// The tetrahedron's smallest corner value of `volume product / edge lengths`, scaled so a regular
/// tetrahedron scores 1.
fn tet_scaled_jacobian(c: &[[f64; 3]; 4]) -> f64 {
    let mut worst = f64::INFINITY;
    for corner in 0..4 {
        let others: Vec<usize> = (0..4).filter(|i| *i != corner).collect();
        let (e1, e2, e3) = (
            sub(c[others[0]], c[corner]),
            sub(c[others[1]], c[corner]),
            sub(c[others[2]], c[corner]),
        );
        // A valid mesh's tetrahedra are positively oriented, so the magnitude of the corner's
        // determinant is its positive value whatever the order of its three edges.
        let determinant = dot(e1, cross(e2, e3)).abs();
        let scale = length(e1) * length(e2) * length(e3);
        worst = worst.min(determinant / scale * std::f64::consts::SQRT_2);
    }
    worst
}

/// An element's smallest Jacobian determinant over its largest, over its nodes and the default
/// integration points. One for a straight-sided element, since its Jacobian is constant.
fn element_jacobian_ratio(nodes: &[[f64; 3]; 10]) -> f64 {
    let mut low = f64::INFINITY;
    let mut high: f64 = 0.0;
    let integration = solidfem_elastic::Quadrature::default().points();
    let naturals = NODE_NATURAL
        .iter()
        .copied()
        .chain(integration.iter().map(|q| q.natural));
    for (index, natural) in naturals.enumerate() {
        match point_geometry(nodes, natural, index) {
            Ok(geometry) => {
                low = low.min(geometry.det_j);
                high = high.max(geometry.det_j);
            }
            // An inverted point counts as the worst possible ratio; the study refuses it too.
            Err(_) => return 0.0,
        }
    }
    low / high
}

/// The material, with `poisson` chosen by the run.
pub fn material(poisson: f64) -> Material {
    Material {
        youngs_modulus: YOUNGS,
        poisson_ratio: poisson,
    }
}

/// Runs the cantilever on `mesh` to the end, returning the solved fields.
pub fn solve(mesh: &QuadraticTetMesh, poisson: f64) -> Result<Solution, ModelError> {
    let mut model = Model::new(mesh.clone(), material(poisson));
    let on_plane = |x: f64, plane: f64| (x - plane).abs() < 1e-9;
    let fixed: Vec<u32> = (0..mesh.nodes().len() as u32)
        .filter(|n| on_plane(mesh.nodes()[*n as usize][0], 0.0))
        .collect();
    for node in fixed {
        model.fix_node(node);
    }
    let traction = FORCE / (DEPTH * DEPTH);
    for face in mesh.boundary_faces() {
        if face.nodes[..3]
            .iter()
            .all(|n| on_plane(mesh.nodes()[*n as usize][0], LENGTH))
        {
            model.add_traction(face, [0.0, 0.0, -traction]);
        }
    }
    let mut study = StaticStudy::new(model, StudyOptions::default(), &mut || false)?;
    loop {
        match study.step(1000, &mut || false)? {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => return Err(ModelError::Cancelled),
        }
    }
    study.solution(&mut || false)
}

/// The mean z displacement of the loaded end.
pub fn tip_deflection(mesh: &QuadraticTetMesh, solution: &Solution) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for (node, p) in mesh.nodes().iter().enumerate() {
        if (p[0] - LENGTH).abs() < 1e-9 {
            sum += solution.displacements[node][2];
            count += 1.0;
        }
    }
    -sum / count
}

/// Timoshenko deflection for the bar (bending plus shear, with coefficient 5/6). The clamp's
/// Poisson restraint and warping restraint are not in it.
pub fn timoshenko(poisson: f64) -> f64 {
    let area = DEPTH * DEPTH;
    let inertia = DEPTH.powi(4) / 12.0;
    let shear_modulus = YOUNGS / (2.0 * (1.0 + poisson));
    FORCE * LENGTH.powi(3) / (3.0 * YOUNGS * inertia)
        + FORCE * LENGTH / (5.0 / 6.0 * shear_modulus * area)
}

/// What the case measured, relative to the unperturbed bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// Tip deflection change, percent.
    pub tip_change: f64,
    /// Peak integration-point von Mises change over the whole part, percent.
    pub peak_change: f64,
    /// Whether that peak is in an element touching a bad triangle.
    pub peak_in_bad_element: bool,
    /// The x position of the whole part's peak point.
    pub peak_x: f64,
    /// Peak change over the elements not touching a bad triangle, percent, against the same
    /// elements in the reference.
    pub peak_excluded_change: f64,
    /// Peak per-node display value change over the whole part, percent.
    pub nodal_peak_change: f64,
    /// Largest change of the per-node value at any node of an element touching a bad triangle,
    /// percent of the reference value at that point.
    pub near_change: f64,
    /// That largest change in stress units.
    pub near_absolute: f64,
    /// The largest change at a node that kept its position, percent of the reference value at
    /// that same node: no interpolation of the reference field is involved.
    pub near_unmoved_change: f64,
    /// Largest change of the raw integration-point von Mises inside elements touching a bad
    /// triangle, percent of the unperturbed element stress at that same physical point.
    pub element_change: f64,
    /// That largest change in stress units.
    pub element_absolute: f64,
    /// Conjugate gradient iterations.
    pub iterations: u64,
    /// Final relative residual.
    pub residual: f64,
    /// Whether the solve stopped at the rounding floor of the system.
    pub at_floor: bool,
}

/// The unperturbed bar's solved results, the base of every comparison.
pub struct Reference {
    /// The mesh.
    pub mesh: QuadraticTetMesh,
    /// The solve.
    pub solution: Solution,
    /// Mean tip deflection.
    pub tip: f64,
    /// Peak integration-point von Mises over the part, and its x position.
    pub peak: (f64, f64),
    /// Peak per-node display value over the part.
    pub nodal_peak: f64,
    /// The material the bar was solved with.
    pub material: Material,
}

/// Solves the unperturbed bar.
pub fn reference(poisson: f64) -> Result<Reference, Box<dyn std::error::Error>> {
    let mesh = unperturbed()?;
    let solution = solve(&mesh, poisson)?;
    let tip = tip_deflection(&mesh, &solution);
    let mut peak = (f64::NEG_INFINITY, 0.0);
    for point in &solution.point_stresses {
        if point.von_mises > peak.0 {
            peak = (point.von_mises, point.position[0]);
        }
    }
    let nodal_peak = solution
        .nodal_von_mises
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(Reference {
        mesh,
        solution,
        tip,
        peak,
        nodal_peak,
        material: material(poisson),
    })
}

/// The tetrahedron of the unperturbed bar containing a physical point, as its index and the
/// point's natural coordinates; `None` if no tetrahedron holds the point.
fn locate(reference: &Reference, point: [f64; 3]) -> Option<(usize, [f64; 3])> {
    let nodes = reference.mesh.nodes();
    for (index, tet) in reference.mesh.tets().iter().enumerate() {
        let [c0, c1, c2, c3] = [tet[0], tet[1], tet[2], tet[3]].map(|n| nodes[n as usize]);
        let volume = tet_volume(c0, c1, c2, c3);
        let r = tet_volume(c0, point, c2, c3) / volume;
        let s = tet_volume(c0, c1, point, c3) / volume;
        let t = tet_volume(c0, c1, c2, point) / volume;
        let first = 1.0 - r - s - t;
        const INSIDE: f64 = -1e-9;
        if first >= INSIDE && r >= INSIDE && s >= INSIDE && t >= INSIDE {
            return Some((index, [r, s, t]));
        }
    }
    None
}

/// The reference's per-node display field at a physical point, interpolated with the quadratic
/// shape functions of the element containing it.
fn reference_field_at(reference: &Reference, point: [f64; 3]) -> Option<f64> {
    let (index, natural) = locate(reference, point)?;
    let tet = reference.mesh.tets()[index];
    let values = shape_values(natural);
    let mut value = 0.0;
    for (slot, shape) in values.iter().enumerate() {
        value += shape * reference.solution.nodal_von_mises[tet[slot] as usize];
    }
    Some(value)
}

/// The reference's raw element von Mises stress at a physical point: the unperturbed element's
/// own strain, not the averaged display field.
fn reference_element_stress(reference: &Reference, point: [f64; 3]) -> Option<f64> {
    let (index, natural) = locate(reference, point)?;
    let tet = reference.mesh.tets()[index];
    let nodes = tet.map(|n| reference.mesh.nodes()[n as usize]);
    let u = tet.map(|n| reference.solution.displacements[n as usize]);
    let geometry = point_geometry(&nodes, natural, 0).ok()?;
    Some(von_mises(&stress_at(
        &reference.material,
        &strain_at(&geometry, &u),
    )))
}

/// The elements with any corner on a bad triangle.
fn touching_elements(distorted: &Distorted) -> Vec<bool> {
    let mut corner = vec![false; distorted.linear.positions().len()];
    for triangle in &distorted.bad_triangles {
        for node in triangle {
            corner[*node as usize] = true;
        }
    }
    distorted
        .linear
        .tets()
        .iter()
        .map(|tet| tet.iter().any(|n| corner[*n as usize]))
        .collect()
}

/// Compares a solved case with the unperturbed bar.
pub fn measure(reference: &Reference, distorted: &Distorted, solution: &Solution) -> Measured {
    let touching = touching_elements(distorted);
    let percent = |value: f64, base: f64| 100.0 * (value - base) / base;

    let tip_change = percent(
        tip_deflection(&distorted.quadratic, solution),
        reference.tip,
    );

    let mut peak = (f64::NEG_INFINITY, 0.0, false);
    let mut excluded = f64::NEG_INFINITY;
    for point in &solution.point_stresses {
        let bad = touching[point.tet as usize];
        if point.von_mises > peak.0 {
            peak = (point.von_mises, point.position[0], bad);
        }
        if !bad {
            excluded = excluded.max(point.von_mises);
        }
    }
    let excluded_reference = reference
        .solution
        .point_stresses
        .iter()
        .filter(|p| !touching[p.tet as usize])
        .map(|p| p.von_mises)
        .fold(f64::NEG_INFINITY, f64::max);
    let nodal_peak = solution
        .nodal_von_mises
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut near_change = 0.0f64;
    let mut near_absolute = 0.0f64;
    let mut near_unmoved_change = 0.0f64;
    let floor = 1e-6 * reference.peak.0;
    let quadratic = &distorted.quadratic;
    let mut seen = vec![false; quadratic.nodes().len()];
    for (tet, touches) in quadratic.tets().iter().zip(&touching) {
        if !touches {
            continue;
        }
        for node in tet {
            let at = *node as usize;
            if seen[at] {
                continue;
            }
            seen[at] = true;
            if let Some(base) = reference_field_at(reference, quadratic.nodes()[at]) {
                let difference = (solution.nodal_von_mises[at] - base).abs();
                // Near the neutral axis the local value is small and most of a change is real;
                // the floor only keeps a value of exactly zero from dividing by zero.
                let change = 100.0 * difference / base.abs().max(floor);
                near_change = near_change.max(change);
                near_absolute = near_absolute.max(difference);
                // Quadratic interpolation of the V-shaped von Mises field across the neutral axis
                // is what `near_change` picks up at a moved node, so the like-for-like comparison
                // is the nodes that stayed put, where no interpolation is involved.
                if quadratic.nodes()[at] == reference.mesh.nodes()[at] {
                    let same = reference.solution.nodal_von_mises[at];
                    let change =
                        100.0 * (solution.nodal_von_mises[at] - same).abs() / same.abs().max(floor);
                    near_unmoved_change = near_unmoved_change.max(change);
                }
            }
        }
    }

    let mut element_change = 0.0f64;
    let mut element_absolute = 0.0f64;
    for point in &solution.point_stresses {
        if touching[point.tet as usize] {
            if let Some(base) = reference_element_stress(reference, point.position) {
                let difference = (point.von_mises - base).abs();
                element_change = element_change.max(100.0 * difference / base.abs().max(floor));
                element_absolute = element_absolute.max(difference);
            }
        }
    }

    Measured {
        tip_change,
        peak_change: percent(peak.0, reference.peak.0),
        peak_in_bad_element: peak.2,
        peak_x: peak.1,
        peak_excluded_change: percent(excluded, excluded_reference),
        nodal_peak_change: percent(nodal_peak, reference.nodal_peak),
        near_change,
        near_absolute,
        near_unmoved_change,
        element_change,
        element_absolute,
        iterations: solution.report.iterations,
        residual: solution.report.relative_residual,
        at_floor: solution.report.at_rounding_floor,
    }
}
