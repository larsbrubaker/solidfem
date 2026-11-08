// A 3D-printed part as a solid shell around a weaker infill core: a square-section cantilever under
// an end shear, with the material chosen per integration point from depth below the surface. How
// much does that cost, on a mesh that does not follow the shell?
//
// Why this beam. The section is a bar of height and width 1 with a uniform shell of thickness t all
// around, and core modulus E_shell / ratio. Poisson's ratio is the same in both, so under pure
// bending a plane section stays plane and both materials are in the same uniaxial state with
// strain_xx = kappa z and identical lateral strain: the transformed-section rigidity EI below is
// then the exact bending rigidity, not an approximation. At L/h = 20 the Timoshenko shear term is
// about a percent of the deflection, so the Euler-Bernoulli bending term carries the answer and the
// shear term is a correction.
//
// The closed form is Timoshenko's composite beam: delta = P L^3 / (3 EI) + P L / K_shear, with
// K_shear from `Section::shear_compliance`. That term is an energy-based estimate rather than an
// exact result; the bending term is the part that is exact. Nothing below relies on the shear term
// being better than its own size, so every limit carries the whole shear fraction as its modelling
// allowance.
//
// The depth-lookup material uses `Section::material`, the depth to the nearest of the four long
// faces. A real part's shell also wraps the end faces; here the ends are excluded so the section
// stays prismatic and the beam formula stays exact.
//
// What is asserted and what is only printed. A limit appears only where it can be derived (each
// states how). The non-conforming error against depth is measured data with no derivable bound, so
// `depth_lookup_error_table` prints it (`cargo test --release --test validation -- --ignored
// shell_core --nocapture`). Each printed row also gives `predicted`, the EI error expected from
// sampling the step in material with the rule's own points, so the measured error can be judged
// against what the sampling alone explains.
//
// What the table shows. A depth lookup behaves like a shell of the wrong thickness: the rule sees
// the shell only through its sample depths, so the thickness it finds is off by up to a fixed
// fraction of a cell, the figure `a_shell_thinner_than_the_first_sample_disappears_from_the_model`
// prints for each rule. The error depends on where the interface falls among those depths, not
// smoothly on the element count, and falls only as one over the elements through the shell. A
// shell shallower than the first sample disappears altogether, with no warning.

use solidfem::elastic::{
    Clock, Material, MaterialField, Model, PcgOptions, Quadrature, Solution, StaticStudy,
    StepOutcome, StudyOptions,
};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh, TetMesh};

use crate::support::{faces_at, fix_root, nodes_at};

const LENGTH: f64 = 20.0;
const HEIGHT: f64 = 1.0;
const WIDTH: f64 = 1.0;
const LOAD: f64 = 1.0;
const SHELL_MODULUS: f64 = 1000.0;
const POISSON: f64 = 0.3;
/// The distance within which a node counts as on a plane.
const ON_PLANE: f64 = 1e-9;
/// Mid-span stress points are read within one section height of the middle: St Venant end effects
/// decay over about a section depth, and these points are nine or more depths from either end.
const STRESS_BAND: f64 = HEIGHT;

/// One beam: the shell thickness and the core's modulus.
#[derive(Clone, Copy)]
struct Section {
    shell: f64,
    core_modulus: f64,
}

impl Section {
    fn with_ratio(shell: f64, ratio: f64) -> Self {
        Section {
            shell,
            core_modulus: SHELL_MODULUS / ratio,
        }
    }

    fn inner_height(&self) -> f64 {
        HEIGHT - 2.0 * self.shell
    }

    fn inner_width(&self) -> f64 {
        WIDTH - 2.0 * self.shell
    }

    /// Transformed-section bending rigidity about the y axis: the shell's second moment is the
    /// full rectangle less the inner one, and the core's is the inner rectangle.
    fn rigidity(&self) -> f64 {
        self.rigidity_with_shell(self.shell)
    }

    fn rigidity_with_shell(&self, shell: f64) -> f64 {
        let (inner_h, inner_w) = (HEIGHT - 2.0 * shell, WIDTH - 2.0 * shell);
        let outer = WIDTH * HEIGHT.powi(3) / 12.0;
        let inner = inner_w * inner_h.powi(3) / 12.0;
        SHELL_MODULUS * (outer - inner) + self.core_modulus * inner
    }

    /// Energy-based shear compliance per unit length. A section's shear flow at height z is
    /// `V Q_E(z) / EI`, with Q_E the modulus-weighted first moment above z; with no shear stress
    /// across the width the shear strain is uniform there, so that flow divides by the width
    /// integral of the shear modulus. Integrated over the height: 1 / K = int (Q_E / EI)^2 /
    /// (int G dy) dz. Composite Simpson: Q_E is piecewise quadratic, so 40000 intervals leave
    /// only rounding.
    fn shear_compliance(&self) -> f64 {
        let ei = self.rigidity();
        let half_core = 0.5 * self.inner_height();
        let shear_factor = 1.0 / (2.0 * (1.0 + POISSON));
        let (core_width, shell_side) = (self.inner_width(), 2.0 * self.shell);
        let core_row = SHELL_MODULUS * shell_side + self.core_modulus * core_width;
        let shell_row = SHELL_MODULUS * WIDTH;
        let first_moment = |z: f64| -> f64 {
            let z = z.abs();
            let half = 0.5 * HEIGHT;
            if z >= half_core {
                0.5 * shell_row * (half * half - z * z)
            } else {
                0.5 * core_row * (half_core * half_core - z * z)
                    + 0.5 * shell_row * (half * half - half_core * half_core)
            }
        };
        let integrand = |z: f64| -> f64 {
            let row = if z.abs() >= half_core {
                shell_row
            } else {
                core_row
            };
            let flow = first_moment(z) / ei;
            flow * flow / (row * shear_factor)
        };
        let intervals = 40000usize;
        let step = HEIGHT / intervals as f64;
        let mut sum = integrand(-0.5 * HEIGHT) + integrand(0.5 * HEIGHT);
        for i in 1..intervals {
            let z = -0.5 * HEIGHT + step * i as f64;
            sum += integrand(z) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * step / 3.0
    }

    /// (bending deflection, shear deflection) at the tip under the end load.
    fn tip_parts(&self) -> (f64, f64) {
        (
            LOAD * LENGTH.powi(3) / (3.0 * self.rigidity()),
            LOAD * LENGTH * self.shear_compliance(),
        )
    }

    fn tip(&self) -> f64 {
        let (bending, shear) = self.tip_parts();
        bending + shear
    }

    /// The shear term's share of the whole deflection: the closed form's modelling allowance.
    fn shear_fraction(&self) -> f64 {
        let (bending, shear) = self.tip_parts();
        shear / (bending + shear)
    }

    /// Bending curvature at station x under the end load.
    fn curvature(&self, x: f64) -> f64 {
        LOAD * (LENGTH - x) / self.rigidity()
    }

    fn depth(position: [f64; 3]) -> f64 {
        position[1]
            .min(WIDTH - position[1])
            .min(position[2])
            .min(HEIGHT - position[2])
    }

    /// Whether a position lies in the shell.
    fn in_shell(&self, position: [f64; 3]) -> bool {
        Self::depth(position) < self.shell
    }

    /// The depth lookup, the thing under test: shell above the interface, core below it.
    fn material(self) -> impl Fn([f64; 3]) -> Material + Copy {
        move |position| Material {
            youngs_modulus: if self.in_shell(position) {
                SHELL_MODULUS
            } else {
                self.core_modulus
            },
            poisson_ratio: POISSON,
        }
    }

    /// Bending stress magnitude at a point according to the transformed section.
    fn bending_stress(&self, position: [f64; 3], modulus: f64) -> f64 {
        modulus * self.curvature(position[0]) * (position[2] - 0.5 * HEIGHT).abs()
    }
}

/// A box mesh whose grid planes are the given coordinates, through the production box builder:
/// it only makes uniform cells, so positions are remapped, and a monotone map keeps every
/// tetrahedron positive and every plane flat.
fn graded_bar(x: &[f64], y: &[f64], z: &[f64]) -> QuadraticTetMesh {
    let divisions = [x.len() - 1, y.len() - 1, z.len() - 1];
    let uniform = box_tet_mesh([0.0; 3], [1.0; 3], divisions).expect("box");
    let mapped = uniform
        .positions()
        .iter()
        .map(|p| {
            let index = |a: usize| (p[a] * divisions[a] as f64).round() as usize;
            [x[index(0)], y[index(1)], z[index(2)]]
        })
        .collect();
    QuadraticTetMesh::from_linear(&TetMesh::new(mapped, uniform.tets().to_vec()).expect("graded"))
}

fn uniform_planes(length: f64, cells: usize) -> Vec<f64> {
    (0..=cells)
        .map(|i| length * i as f64 / cells as f64)
        .collect()
}

/// Section planes for a mesh that does NOT follow the shell: cells of depth `cell_depth` from each
/// face, enough to span the interface and one more cell, then the middle in cells at most
/// `max_middle` thick. When the outer bands would meet, as in the coarsest cases, they stop short
/// and the leftover becomes a single middle cell.
fn lookup_planes(cell_depth: f64, shell: f64, max_middle: f64) -> Vec<f64> {
    let wanted = (shell / cell_depth).ceil() as usize + 1;
    let fit = (0.5 * HEIGHT / cell_depth).floor() as usize;
    let outer = wanted.min(fit).max(1);
    let band = cell_depth * outer as f64;
    let mut planes: Vec<f64> = (0..=outer).map(|i| cell_depth * i as f64).collect();
    let middle = HEIGHT - 2.0 * band;
    if middle > ON_PLANE {
        let cells = (middle / max_middle).ceil() as usize;
        for i in 1..cells {
            planes.push(band + middle * i as f64 / cells as f64);
        }
        planes.push(HEIGHT - band);
    }
    for i in (0..outer).rev() {
        planes.push(HEIGHT - cell_depth * i as f64);
    }
    planes
}

/// Section planes that DO follow the shell: `through` cells across the shell and `core` cells
/// across the middle, the interface exactly on a grid plane so no element is cut.
fn fitted_planes(shell: f64, through: usize, core: usize) -> Vec<f64> {
    let mut planes: Vec<f64> = (0..=through)
        .map(|i| shell * i as f64 / through as f64)
        .collect();
    let middle = HEIGHT - 2.0 * shell;
    for i in 1..core {
        planes.push(shell + middle * i as f64 / core as f64);
    }
    planes.extend(
        (0..=through)
            .rev()
            .map(|i| HEIGHT - shell * i as f64 / through as f64),
    );
    planes
}

/// What one run measured.
struct Measured {
    /// Tip deflection from the work the traction does: the load's energy conjugate.
    tip: f64,
    /// Peak surface nodal von Mises at mid-span, as a result viewer shows it.
    shell_nodal: f64,
    /// (largest von Mises in the band, closed form there, how many shell points the band holds).
    shell_point: Option<(f64, f64, usize)>,
    /// (largest |stress_xx| among core points in the band, closed form there, core point count).
    core_point: Option<(f64, f64, usize)>,
    free_dofs: usize,
    iterations: u64,
    solve_seconds: f64,
}

/// Solves one model and takes every measurement.
fn solve_and_measure(
    section: Section,
    mesh: &QuadraticTetMesh,
    rule: Quadrature,
    field: impl Fn([f64; 3]) -> Material,
) -> (Measured, Solution) {
    let mut model = Model::new(mesh.clone(), field);
    model.set_quadrature(rule);
    fix_root(&mut model);
    let traction = LOAD / (WIDTH * HEIGHT);
    let tip_faces = faces_at(mesh, LENGTH);
    for face in &tip_faces {
        model.add_traction(*face, [0.0, 0.0, -traction]);
    }
    let solution = run_coarse_tolerance(model);
    // On a straight 6-node triangle the corner shape functions integrate to zero and each midpoint
    // to a third of the area, so the integral of u over a face is area/3 times the midpoint sum.
    // Dividing by the section area gives the deflection a uniform traction does work through.
    let mut work = 0.0;
    for face in &tip_faces {
        let corner = |n: usize| mesh.nodes()[face.nodes[n] as usize];
        let (a, b, c) = (corner(0), corner(1), corner(2));
        let u = [b[1] - a[1], b[2] - a[2]];
        let v = [c[1] - a[1], c[2] - a[2]];
        let area = 0.5 * (u[0] * v[1] - u[1] * v[0]).abs();
        let mean: f64 = face.nodes[3..]
            .iter()
            .map(|n| solution.displacements[*n as usize][2])
            .sum::<f64>()
            / 3.0;
        work += area * mean;
    }
    let tip = -work / (WIDTH * HEIGHT);

    let mid = 0.5 * LENGTH;
    let shell_nodal = nodes_at(mesh, mid)
        .into_iter()
        .filter(|n| {
            let z = mesh.nodes()[*n][2];
            z.abs() < ON_PLANE || (z - HEIGHT).abs() < ON_PLANE
        })
        .map(|n| solution.nodal_von_mises[n])
        .fold(0.0, f64::max);

    let mut shell_point: Option<(f64, f64, usize)> = None;
    let mut core_point: Option<(f64, f64, usize)> = None;
    let (mut shell_count, mut core_count) = (0usize, 0usize);
    for point in &solution.point_stresses {
        if (point.position[0] - mid).abs() > STRESS_BAND {
            continue;
        }
        if section.in_shell(point.position) {
            shell_count += 1;
            let closed = section.bending_stress(point.position, SHELL_MODULUS);
            if shell_point.is_none_or(|(best, _, _)| point.von_mises > best) {
                shell_point = Some((point.von_mises, closed, 0));
            }
        } else {
            core_count += 1;
            let value = point.stress[0].abs();
            let closed = section.bending_stress(point.position, section.core_modulus);
            if core_point.is_none_or(|(best, _, _)| value > best) {
                core_point = Some((value, closed, 0));
            }
        }
    }
    let r = solution.report;
    (
        Measured {
            tip,
            shell_nodal,
            shell_point: shell_point.map(|(v, c, _)| (v, c, shell_count)),
            core_point: core_point.map(|(v, c, _)| (v, c, core_count)),
            free_dofs: r.free_dofs,
            iterations: r.iterations,
            solve_seconds: r.solve_seconds,
        },
        solution,
    )
}

fn relative(value: f64, expected: f64) -> f64 {
    (value - expected) / expected
}

/// The mid-span surface bending stress of the shell, from the transformed section.
fn shell_surface_stress(section: Section) -> f64 {
    SHELL_MODULUS * section.curvature(0.5 * LENGTH) * 0.5 * HEIGHT
}

/// The fraction of a rule's sampled points that lie shallower than `fraction` of a cell, over the
/// six tetrahedra a box cell is cut into. Each tetrahedron walks one axis per edge from the cell's
/// low corner to its high one, so along the depth axis its corners sit at 0 until the walk takes
/// the step along that axis, and at 1 after: a point's depth fraction is the sum of its
/// barycentrics on the corners past that step. Of the six tetrahedra two each take depth first,
/// second and third.
fn fraction_of_samples_shallower(rule: Quadrature, fraction: f64) -> f64 {
    let mut below = 0.0;
    for point in rule.points() {
        let [r, s, t] = point.natural;
        let corners = [1.0 - r - s - t, r, s, t];
        for step in 1..=3 {
            let depth: f64 = corners[step..].iter().sum();
            if depth < fraction {
                below += point.weight / 3.0;
            }
        }
    }
    below
}

/// The shell thickness a depth lookup effectively sees on a uniform cell depth: whole cells above
/// the interface, plus the sampled fraction of the cell it cuts.
fn sampled_shell_thickness(rule: Quadrature, cell_depth: f64, shell: f64) -> f64 {
    let in_cells = shell / cell_depth;
    let whole = in_cells.floor();
    let partial = in_cells - whole;
    if partial <= ON_PLANE {
        // The interface is on a grid plane: no cell is cut, whatever the rule.
        return shell;
    }
    cell_depth * (whole + fraction_of_samples_shallower(rule, partial))
}

/// Runs a study with a residual tolerance of 1e-7 rather than the default 1e-9, because the
/// graded, high-contrast models here take many thousands of Jacobi iterations and every measurement
/// is read to about 0.1%. The solver's own options give the reason this is enough: the error in
/// strain energy shrinks with the square of the residual, so 1e-7 leaves a relative compliance
/// error of order 1e-14 times the condition number; at the 1e9 these models can reach, 1e-5.
fn run_coarse_tolerance(model: Model<impl MaterialField>) -> Solution {
    let start = std::time::Instant::now();
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    let options = StudyOptions {
        pcg: PcgOptions {
            tolerance: 1e-7,
            max_iterations: None,
        },
        clock: Some(clock),
    };
    let mut study = StaticStudy::new(model, options, &mut || false).expect("a valid model");
    loop {
        match study.step(1000, &mut || false).expect("the solve succeeds") {
            StepOutcome::Converged(_) => break,
            StepOutcome::Running(_) => {}
            StepOutcome::Cancelled(_) => panic!("nothing cancelled this solve"),
        }
    }
    study.solution(&mut || false).expect("a converged solve")
}

/// A body-fitted mesh for a section: `through` cells across the shell, `core` across the middle,
/// `length_cells` along the bar.
fn fitted_bar(
    section: Section,
    through: usize,
    core: usize,
    length_cells: usize,
) -> QuadraticTetMesh {
    let planes = fitted_planes(section.shell, through, core);
    graded_bar(&uniform_planes(LENGTH, length_cells), &planes, &planes)
}

/// Cells along the bar in the sweep: long cells are cheap and the deflection error they add is
/// common to the lookup and to the reference it is measured against, so it cancels.
const SWEEP_LENGTH_CELLS: usize = 12;
/// The sweep's reference section mesh: two cells through the shell and four across the core.
const REFERENCE_THROUGH: usize = 2;
const REFERENCE_CORE: usize = 4;
/// Core cell cap in the sweep's lookup meshes: their cost is in the outer band, the middle is
/// smooth.
const SWEEP_MIDDLE_CELL: f64 = 0.25;

fn rule_name(rule: Quadrature) -> &'static str {
    match rule {
        Quadrature::FourPoint => "4pt",
        Quadrature::FourteenPoint => "14pt",
    }
}

#[test]
fn a_body_fitted_mesh_agrees_with_the_closed_form() {
    // Bending is exact for a prismatic composite with a shared Poisson ratio, so the closed form's
    // only inexact part is its shear term S (a fraction of the deflection, printed); the shear
    // term cannot be wrong by more than itself. A full root clamp also restrains warping and the
    // Poisson contraction there. The cantilever case carries 0.5% for root effects and
    // discretization at nu = 0, and this limit keeps that figure, set before any result was read.
    // It is not a bound on the Poisson restraint at nu = 0.3: a first-order energy estimate (a
    // restrained zone about h / 2 long holds 3 a / L of the compliance, stiffened by
    // nu^2 / (1 - nu^2)) puts that near 0.7%, on the stiff side of the closed form. The limit
    // stays as first written rather than being fitted to the measurement. Tip limit: S + 0.5%.
    //
    // The transformed-section surface stress is M c / EI, so it inherits the deflection's EI gap,
    // and shear stress vanishes at the surface. Stress is read at a node evaluated in its
    // elements, where a 10-node tet's linear strain contains the bending strain exactly, so what
    // remains is the coarse-mesh strain error: another 1%. Shell limit: S + 1.5%.
    //
    // The core's stress is read at its outermost integration point, away from the surface but
    // still in an element whose whole depth is core: the same EI gap, and 1.5% for the same
    // discretization allowance plus the shear strain's contribution there, which the closed form
    // omits. Core limit: S + 2%.
    for ratio in [10.0, 30.0] {
        let section = Section::with_ratio(0.1, ratio);
        let mesh = fitted_bar(section, 1, 4, 20);
        let (m, _) = solve_and_measure(section, &mesh, Quadrature::FourPoint, section.material());
        let allowance = section.shear_fraction();
        let tip = relative(m.tip, section.tip()).abs();
        let shell = relative(m.shell_nodal, shell_surface_stress(section)).abs();
        let (core_value, core_closed, _) = m.core_point.expect("core points in the band");
        let core = relative(core_value, core_closed).abs();
        println!(
            "SHELLCORE fitted ratio={ratio}: shear={allowance:.4} tip={:.4}% shell={:.4}% core={:.4}% dofs={} iters={} s={:.1}",
            tip * 100.0,
            shell * 100.0,
            core * 100.0,
            m.free_dofs,
            m.iterations,
            m.solve_seconds
        );
        assert!(tip < allowance + 0.005, "ratio {ratio}: tip error {tip}");
        assert!(
            shell < allowance + 0.015,
            "ratio {ratio}: shell stress error {shell}"
        );
        assert!(
            core < allowance + 0.02,
            "ratio {ratio}: core stress error {core}"
        );
    }
}

#[test]
fn a_depth_lookup_lies_between_its_all_shell_and_all_core_bounds() {
    // The one bound a depth lookup obeys on any mesh, whatever the rule. Each integration point's
    // modulus lies between the core's and the shell's, so the assembled stiffness is ordered
    // between the same mesh's two uniform-material stiffnesses, and for a fixed load compliance
    // falls as stiffness rises: the tip of an all-shell bar is a lower bound and of an all-core
    // bar an upper bound, exactly. A lookup outside the bracket would be a bug. The interface is
    // placed at an awkward 0.37 of a cell deep and the cell is coarser than the shell.
    let section = Section::with_ratio(0.1, 30.0);
    let planes = lookup_planes(section.shell / 0.37, section.shell, SWEEP_MIDDLE_CELL);
    let mesh = graded_bar(
        &uniform_planes(LENGTH, SWEEP_LENGTH_CELLS),
        &planes,
        &planes,
    );
    let (all_shell, _) = solve_and_measure(
        section,
        &mesh,
        Quadrature::FourPoint,
        uniform(SHELL_MODULUS),
    );
    let (all_core, _) = solve_and_measure(
        section,
        &mesh,
        Quadrature::FourPoint,
        uniform(section.core_modulus),
    );
    for rule in [Quadrature::FourPoint, Quadrature::FourteenPoint] {
        let (lookup, _) = solve_and_measure(section, &mesh, rule, section.material());
        assert!(
            all_shell.tip <= lookup.tip && lookup.tip <= all_core.tip,
            "{}: lookup tip {} outside [{}, {}]",
            rule_name(rule),
            lookup.tip,
            all_shell.tip,
            all_core.tip
        );
    }
}

/// One material everywhere, for the runs a depth lookup is compared with.
fn uniform(modulus: f64) -> impl Fn([f64; 3]) -> Material {
    move |_| Material {
        youngs_modulus: modulus,
        poisson_ratio: POISSON,
    }
}

/// Every depth, as a fraction of a cell, at which a rule samples a box cell. A cell's tetrahedra
/// walk the depth axis at their first, second or third step, measured from either face.
fn sample_depths(rule: Quadrature) -> Vec<f64> {
    let mut depths = Vec::new();
    for point in rule.points() {
        let [r, s, t] = point.natural;
        let corners = [1.0 - r - s - t, r, s, t];
        for step in 1..=3 {
            depths.push(corners[step..].iter().sum());
            depths.push(corners[..step].iter().sum());
        }
    }
    depths
}

/// The most the shell thickness a rule samples can differ from the true one, in cells: the
/// largest gap between the sampled fraction and the true fraction, just either side of each
/// sample depth, which is where that step function jumps.
fn worst_thickness_error(rule: Quadrature) -> f64 {
    let gap = |f: f64| (fraction_of_samples_shallower(rule, f) - f).abs();
    sample_depths(rule)
        .into_iter()
        .map(|f| gap(f - 1e-12).max(gap(f + 1e-12)))
        .fold(0.0, f64::max)
}

#[test]
fn a_shell_thinner_than_the_first_sample_disappears_from_the_model() {
    // Exact, not a tolerance. Every integration point sits at least the rule's shallowest sample
    // depth below a cell's faces, in both section directions, so a shell nine tenths of that deep
    // holds no integration point at all. Every sample then reads the core, the assembled system
    // is the all-core system operation for operation, and the two tips must be equal to the bit.
    // A thin wall on a coarse mesh therefore reports the part as if it had no shell, and nothing
    // in the model refuses it.
    for rule in [Quadrature::FourPoint, Quadrature::FourteenPoint] {
        let shallowest = sample_depths(rule).into_iter().fold(1.0, f64::min);
        let cell = 0.4;
        let section = Section::with_ratio(0.9 * shallowest * cell, 30.0);
        let planes = lookup_planes(cell, section.shell, SWEEP_MIDDLE_CELL);
        let mesh = graded_bar(
            &uniform_planes(LENGTH, SWEEP_LENGTH_CELLS),
            &planes,
            &planes,
        );
        let (lookup, _) = solve_and_measure(section, &mesh, rule, section.material());
        let (core, _) = solve_and_measure(section, &mesh, rule, uniform(section.core_modulus));
        println!(
            "SHELLCORE {}: shallowest sample {shallowest:.4} of a cell, worst thickness error {:.4} of a cell",
            rule_name(rule),
            worst_thickness_error(rule)
        );
        assert_eq!(lookup.tip, core.tip, "{}", rule_name(rule));
    }
}

/// Deflection, surface stress and core stress for a depth lookup, one row of the printed table.
fn lookup_row(section: Section, ratio: f64, depth: f64, rule: Quadrature, reference: f64) {
    let cell = section.shell / depth;
    let planes = lookup_planes(cell, section.shell, SWEEP_MIDDLE_CELL);
    let mesh = graded_bar(
        &uniform_planes(LENGTH, SWEEP_LENGTH_CELLS),
        &planes,
        &planes,
    );
    let (m, _) = solve_and_measure(section, &mesh, rule, section.material());
    let sampled = sampled_shell_thickness(rule, cell, section.shell);
    let predicted = section.rigidity() / section.rigidity_with_shell(sampled) - 1.0;
    let point = |value: Option<(f64, f64, usize)>| match value {
        Some((measured, closed, count)) => {
            format!("{:+.2}%({count})", relative(measured, closed) * 100.0)
        }
        None => "none".to_string(),
    };
    println!(
        "SHELLCORE ratio={ratio} t={} depth={depth} cell={cell:.4} across={} rule={} tip={:+.2}% predicted={:+.2}% surface={:+.2}% shell_point={} core={} dofs={} iters={} s={:.1}",
        section.shell,
        planes.len() - 1,
        rule_name(rule),
        relative(m.tip, reference) * 100.0,
        predicted * 100.0,
        relative(m.shell_nodal, shell_surface_stress(section)) * 100.0,
        point(m.shell_point),
        point(m.core_point),
        m.free_dofs,
        m.iterations,
        m.solve_seconds
    );
}

/// Measures the depth lookup on non-conforming meshes. Printed, with no limit: nothing derivable
/// bounds these errors (`predicted` is a first-order estimate, not a bound), so the table is the
/// result, run in release with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement table taking minutes; run in release with --nocapture"]
fn depth_lookup_error_table() {
    let cases: [(f64, f64, &[f64]); 3] = [
        (0.1, 30.0, &[0.25, 0.37, 0.5, 0.63, 1.0, 1.37, 2.0]),
        (0.1, 10.0, &[0.25, 0.5, 1.0, 1.37]),
        (0.04, 30.0, &[0.25, 0.5]),
    ];
    for (shell, ratio, depths) in cases {
        let section = Section::with_ratio(shell, ratio);
        let mesh = fitted_bar(
            section,
            REFERENCE_THROUGH,
            REFERENCE_CORE,
            SWEEP_LENGTH_CELLS,
        );
        let (reference, _) =
            solve_and_measure(section, &mesh, Quadrature::FourPoint, section.material());
        println!(
            "SHELLCORE reference ratio={ratio} t={shell}: tip={} closed={} shear={:.4} dofs={} iters={} s={:.1}",
            reference.tip,
            section.tip(),
            section.shear_fraction(),
            reference.free_dofs,
            reference.iterations,
            reference.solve_seconds
        );
        for depth in depths {
            for rule in [Quadrature::FourPoint, Quadrature::FourteenPoint] {
                lookup_row(section, ratio, *depth, rule, reference.tip);
            }
        }
    }
}

#[test]
#[ignore = "the 34k-unknown fine mesh takes minutes; run in release"]
fn refining_a_body_fitted_mesh_only_ever_raises_the_compliance() {
    // A conforming mesh's stiffness is ordered by inclusion: doubling every cell of the structured
    // grid splits each tetrahedron along its own planes, and each element's material is uniform
    // here, so the 4-point rule is exact and the coarse solution is in the fine mesh's space. The
    // compliance, measured as the traction's work, therefore rises with refinement. The levels
    // double all three directions, which is what makes them nested.
    for ratio in [10.0, 30.0] {
        let section = Section::with_ratio(0.1, ratio);
        let mut previous = f64::NEG_INFINITY;
        for (through, core, length_cells) in [(1, 2, 10), (2, 4, 20)] {
            let mesh = fitted_bar(section, through, core, length_cells);
            let (m, _) =
                solve_and_measure(section, &mesh, Quadrature::FourPoint, section.material());
            println!(
                "SHELLCORE nested ratio={ratio} through={through} core={core} x={length_cells}: tip={} closed={} dofs={} iters={} s={:.1}",
                m.tip,
                section.tip(),
                m.free_dofs,
                m.iterations,
                m.solve_seconds
            );
            assert!(
                m.tip > previous,
                "ratio {ratio}: tip {} did not rise from {previous}",
                m.tip
            );
            previous = m.tip;
        }
    }
}
