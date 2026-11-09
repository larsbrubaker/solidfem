//! The end-of-mesher gate: a 100 x 100 x 0.8 mm plate, meshed by this workspace's mesher with two
//! element layers through the wall, solved through the 10-node elements, and compared with beam
//! theory. Everything the verdict uses is measured here, on the mesh's public arrays:
//! an independent ray walk along columns across the wall, the mesher's contracts, and the free-end
//! deflection against Euler-Bernoulli plus Timoshenko shear. A capped solve is reported as not
//! assessed, never as a pass. The solve uses the multigrid preconditioner: diagonal scaling cannot
//! converge on a plate this thin in any reasonable time.
//!
//! `solidfem-gate structured <cells> <nu> <cap-seconds> [width]` builds the plate from a structured
//! grid (a width other than 100 is the narrower strip reported as additional information);
//! `solidfem-gate unaligned <cells> <nu> <cap-seconds> [width]` moves the structured plate's
//! interior skin vertices differently on the two skins, a small and quick stand-in for a remeshed
//! surface; `solidfem-gate file <path> <edge> <nu> <cap-seconds>` reads a triangle surface from a
//! text file (`vertex-count face-count`, then `x y z` lines, then `a b c` lines). Run in release.
//! The cap is looked at every few iterations. `MG_INSPECT` prints the levels and how much of the
//! wall the finest aggregates span, `GATE_TRACE=<n>` the residual every n iterations, and
//! `MG_REACH`, `MG_RANGE`, `MG_DEGREE` and `MG_COARSE` override the multigrid options.
//!
//! **The reference.** The plate is clamped at x = 0 and loaded by a uniform traction over the x =
//! L end, total force F. With Poisson's ratio zero there is no anticlastic or clamp restraint, so
//! the plate is a beam: `d = F L^3 / (3 E I) + F L / (k G A)`, `I = W t^3 / 12`, `A = W t`,
//! `k = 5/6`, `G = E / 2`. The shear term is about 0.6 (t / L)^2 of the bending term, 3.8e-5 here,
//! so the choice of `k` and any shear-warping effect at the clamp or the loaded end, each a fraction
//! of that term, are far below the 1% limit; the uniform traction versus the parabolic shear
//! distribution is the same order. Mid-span bending stress is `F (L/2) (t/2) / I`, compared
//! at top-surface corner nodes near x = L/2 away from the side edges.

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use robust::{orient3d, Coord3D};
use solidfem::elastic::{
    Clock, Material, Model, PreconditionerChoice, Quadrature, Report, Solution, StaticStudy,
    StepOutcome, StudyOptions,
};
use solidfem::meshing::quadratic_from_solid;
use solidfem::tetra::{RefineOptions, Refiner, SolidMesh, StepStatus, SurfaceMesher, HULL};
use solidfem_bench::gate_diagnostics::{inspect_aggregates, multigrid_options};

const LENGTH: f64 = 100.0;
const DEFAULT_WIDTH: f64 = 100.0;
const THICK: f64 = 0.8;
const LOAD: f64 = 1.0;
const YOUNGS_MODULUS: f64 = 200_000.0;

/// Solver iterations between looks at the clock. A cycle of multigrid on a large model takes about
/// a second, so a cap is met to within a few seconds.
const SOLVE_SLICE: usize = 5;

type Surface = (Vec<[f64; 3]>, Vec<[u32; 3]>);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize| args.get(i).map(String::as_str).unwrap_or("");
    let parsed = match get(0) {
        "structured" => get(1).parse::<usize>().ok().and_then(|cells| {
            let edge = LENGTH / cells as f64;
            let nu = get(2).parse::<f64>().ok()?;
            let cap = get(3).parse::<f64>().ok()?;
            Some((
                structured_plate(cells, get(4).parse::<f64>().unwrap_or(DEFAULT_WIDTH)),
                edge,
                nu,
                cap,
                get(4).parse::<f64>().unwrap_or(DEFAULT_WIDTH),
                format!("structured {cells} cells"),
            ))
        }),
        "unaligned" => get(1).parse::<usize>().ok().and_then(|cells| {
            let nu = get(2).parse::<f64>().ok()?;
            let cap = get(3).parse::<f64>().ok()?;
            let width = get(4).parse::<f64>().unwrap_or(DEFAULT_WIDTH);
            Some((
                unaligned_plate(cells, width),
                LENGTH / cells as f64,
                nu,
                cap,
                width,
                format!("unaligned skins, {cells} cells"),
            ))
        }),
        "file" => (|| {
            let surface = read_surface(get(1))?;
            let edge = get(2).parse::<f64>().ok()?;
            let nu = get(3).parse::<f64>().ok()?;
            let cap = get(4).parse::<f64>().ok()?;
            Some((
                surface,
                edge,
                nu,
                cap,
                DEFAULT_WIDTH,
                format!("remeshed surface {}", get(1)),
            ))
        })(),
        _ => None,
    };
    let Some(((points, triangles), edge, nu, cap, width, name)) = parsed else {
        eprintln!("usage: solidfem-gate structured <cells> <nu> <cap-seconds> [width]");
        eprintln!("       solidfem-gate unaligned <cells> <nu> <cap-seconds> [width]");
        eprintln!("       solidfem-gate file <path> <edge> <nu> <cap-seconds>");
        std::process::exit(2);
    };
    println!("GATE {name}: width {width}, edge {edge}, nu {nu}, solve cap {cap} s");
    run_gate(points, triangles, edge, nu, cap, width);
}

/// The plate with square cells of `LENGTH / cells` on each face and one quad up the rim per boundary
/// cell; `width` sets the number of rows.
fn structured_plate(cells: usize, width: f64) -> Surface {
    let size = LENGTH / cells as f64;
    let rows = (width / size).round().max(1.0) as usize;
    let (nx, ny) = (cells + 1, rows + 1);
    let at = |i: usize, j: usize, layer: usize| ((layer * nx + i) * ny + j) as u32;
    let mut points = Vec::with_capacity(2 * nx * ny);
    for layer in 0..2 {
        for i in 0..nx {
            for j in 0..ny {
                points.push([
                    LENGTH * i as f64 / cells as f64,
                    width * j as f64 / rows as f64,
                    THICK * layer as f64,
                ]);
            }
        }
    }
    let mut triangles = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, d: u32| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    };
    for i in 0..cells {
        for j in 0..rows {
            quad(
                at(i, j, 1),
                at(i + 1, j, 1),
                at(i + 1, j + 1, 1),
                at(i, j + 1, 1),
            );
            quad(
                at(i, j, 0),
                at(i, j + 1, 0),
                at(i + 1, j + 1, 0),
                at(i + 1, j, 0),
            );
        }
        quad(at(i, 0, 0), at(i + 1, 0, 0), at(i + 1, 0, 1), at(i, 0, 1));
        quad(
            at(i + 1, rows, 0),
            at(i, rows, 0),
            at(i, rows, 1),
            at(i + 1, rows, 1),
        );
    }
    for j in 0..rows {
        quad(at(0, j + 1, 0), at(0, j, 0), at(0, j, 1), at(0, j + 1, 1));
        quad(
            at(cells, j, 0),
            at(cells, j + 1, 0),
            at(cells, j + 1, 1),
            at(cells, j, 1),
        );
    }
    (points, triangles)
}

/// The plate with square cells, each interior skin vertex moved within its cell by a different,
/// sign-alternating amount on the two skins, so the top and bottom triangulations share no column
/// away from the rim: the small, quick stand-in for a remeshed surface.
fn unaligned_plate(cells: usize, width: f64) -> Surface {
    let (mut points, triangles) = structured_plate(cells, width);
    let size = LENGTH / cells as f64;
    let rows = (width / size).round().max(1.0) as usize;
    for p in &mut points {
        let (i, j) = (
            (p[0] / size).round() as usize,
            (p[1] / (width / rows as f64)).round() as usize,
        );
        if i == 0 || j == 0 || i == cells || j == rows {
            continue;
        }
        let sign = if p[2] > 0.0 { 1.0 } else { -1.0 };
        p[0] += 0.31 * size * sign * if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 };
        p[1] += 0.23 * size * sign * if i.is_multiple_of(2) { 1.0 } else { -1.0 };
    }
    (points, triangles)
}

fn read_surface(path: &str) -> Option<Surface> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut tokens = text.split_ascii_whitespace();
    let vertex_count: usize = tokens.next()?.parse().ok()?;
    let face_count: usize = tokens.next()?.parse().ok()?;
    let mut points = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut p = [0.0; 3];
        for value in &mut p {
            *value = tokens.next()?.parse().ok()?;
        }
        points.push(p);
    }
    let mut triangles = Vec::with_capacity(face_count);
    for _ in 0..face_count {
        let mut t = [0u32; 3];
        for value in &mut t {
            *value = tokens.next()?.parse().ok()?;
        }
        triangles.push(t);
    }
    Some((points, triangles))
}

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

fn run_gate(
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    edge: f64,
    nu: f64,
    cap: f64,
    width: f64,
) {
    let surface_points = points.clone();
    let clock_start = Instant::now();
    let t0 = Instant::now();
    let mut mesher = match SurfaceMesher::new(points, triangles.clone()) {
        Ok(mesher) => mesher,
        Err(error) => return fail(&format!("surface refused: {error}")),
    };
    match mesher.step(usize::MAX, &mut || false) {
        Ok(StepStatus::Done) => {}
        other => return fail(&format!("surface recovery did not finish: {other:?}")),
    }
    let solid = match mesher.result() {
        Ok(solid) => solid,
        Err(error) => return fail(&format!("surface recovery failed: {error}")),
    };
    let recovery_seconds = t0.elapsed().as_secs_f64();
    println!(
        "phase surface recovery: {recovery_seconds:.3} s, {} tets, {} points, flips and {} added points",
        solid.tets.len(),
        solid.points.len(),
        mesher.report().steiner_points,
    );

    let t0 = Instant::now();
    let options = RefineOptions {
        layers_through_walls: 2,
        ..RefineOptions::default()
    };
    let size = 1.5 * edge;
    let mut refiner = match Refiner::new(solid, Box::new(move |_| size), options) {
        Ok(refiner) => refiner,
        Err(error) => return fail(&format!("refiner refused: {error}")),
    };
    match refiner.step(usize::MAX, &mut || false) {
        Ok(StepStatus::Done) => {}
        other => return fail(&format!("refinement did not finish: {other:?}")),
    }
    let refined = refiner.result().expect("finished refinement has a result");
    let report = refiner.report();
    println!(
        "phase refinement: {:.3} s, {} tets, {} points; layer points {}, dropped {}, thin-wall vertices {}, too thin {}, through edges split {}, refused {}, size points {}; min dihedral before {:.2} after {:.2}, mean after {:.2}",
        t0.elapsed().as_secs_f64(),
        refined.tets.len(),
        refined.points.len(),
        report.layer_points,
        report.layer_points_dropped,
        report.thin_wall_vertices,
        report.walls_too_thin_for_rounding,
        report.through_edges_split,
        report.through_edges_refused,
        report.size_points,
        report.before.min_degrees,
        report.after.min_degrees,
        report.after.mean_degrees,
    );

    // A refused through edge means an element can span the wall where a split was meant to remove
    // it, which the verdict below cannot see from columns it did not sample.
    if report.through_edges_refused != 0 {
        return fail(&format!(
            "{} through edges were refused; an element may span the wall",
            report.through_edges_refused
        ));
    }

    // The mesher's contracts, checked on the arrays alone.
    if let Err(message) = check_contracts(&refined, &surface_points, width) {
        return fail(&format!("mesher contract broken: {message}"));
    }
    println!("contracts: surface bits identical, every tet exactly positive, volume matches, added points strictly inside: PASS");

    // Layers through the wall, by a walk along columns.
    // Measured on the wall this surface really has: a file may carry the thickness rounded to f32.
    let wall = surface_points.iter().map(|p| p[2]).fold(f64::MIN, f64::max);
    let layers = measure_layers(&refined, edge, wall, width);
    println!(
        "layers away from the rim ({} columns, margin {:.1} mm): at least two tets everywhere = {}, none crossing both skins = {} ({} columns have one), fewest tets {}, largest span {:.4} of a {:.1} wall; columns missed {}",
        layers.away_columns,
        layers.margin,
        layers.away_min_tets >= 2,
        layers.away_crossing == 0,
        layers.away_crossing,
        layers.away_min_tets,
        layers.away_largest_span,
        THICK,
        layers.away_missing,
    );
    println!(
        "layers at the rim ({} columns): fewest tets {}, columns with a tet crossing both skins {}, largest span {:.4}; columns missed {}",
        layers.rim_columns, layers.rim_min_tets, layers.rim_crossing, layers.rim_largest_span, layers.rim_missing,
    );
    let layers_pass =
        layers.away_min_tets >= 2 && layers.away_crossing == 0 && layers.away_missing == 0;

    let t0 = Instant::now();
    let mesh = match quadratic_from_solid(&refined, Quadrature::FourPoint) {
        Ok(mesh) => mesh,
        Err(error) => return fail(&format!("10-node upgrade refused: {error}")),
    };
    println!(
        "phase 10-node upgrade: {:.3} s, {} nodes, {} elements",
        t0.elapsed().as_secs_f64(),
        mesh.nodes().len(),
        mesh.tets().len(),
    );
    let material = Material {
        youngs_modulus: YOUNGS_MODULUS,
        poisson_ratio: nu,
    };
    let mut model = Model::new(mesh.clone(), material);
    for node in 0..mesh.nodes().len() as u32 {
        if mesh.nodes()[node as usize][0].abs() < 1e-9 {
            model.fix_node(node);
        }
    }
    let end_faces: Vec<_> = mesh
        .boundary_faces()
        .into_iter()
        .filter(|face| {
            face.nodes[..3]
                .iter()
                .all(|n| (mesh.nodes()[*n as usize][0] - LENGTH).abs() < 1e-9)
        })
        .collect();
    let area: f64 = end_faces
        .iter()
        .map(|face| {
            let [a, b, c] =
                [face.nodes[0], face.nodes[1], face.nodes[2]].map(|n| mesh.nodes()[n as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
        })
        .sum();
    if area <= 0.0 {
        return fail("no boundary face at x = L");
    }
    println!("loaded end area {area:.6}, expected {}", width * THICK);
    for face in end_faces {
        model.add_traction(face, [0.0, 0.0, -LOAD / area]);
    }
    let start = Instant::now();
    let clock: Clock = Box::new(move || start.elapsed().as_secs_f64());
    let mut study = match StaticStudy::new(
        model,
        StudyOptions {
            clock: Some(clock),
            preconditioner: PreconditionerChoice::Multigrid(multigrid_options()),
            ..StudyOptions::default()
        },
        &mut || false,
    ) {
        Ok(study) => study,
        Err(error) => return fail(&format!("study refused: {error}")),
    };
    let report_of = |r: Report| {
        let hierarchy = r.multigrid.map_or(String::new(), |m| {
            format!(
                "; multigrid {} levels, {} coarse values, operator complexity {:.2}, memory {:.0} MB",
                m.levels,
                m.coarse_dofs,
                m.operator_complexity,
                m.memory_bytes as f64 / 1e6
            )
        });
        format!(
            "total DOF {}, free DOF {}, nonzeros {}, iterations {}, residual {:.3e}, rounding floor {}; assembly {:.3} s, setup {:.3} s, solve {:.3} s, recovery {:.3} s{hierarchy}",
            r.total_dofs, r.free_dofs, r.nonzeros, r.iterations, r.relative_residual, r.at_rounding_floor, r.assembly_seconds, r.preconditioner_seconds, r.solve_seconds, r.recovery_seconds
        )
    };
    if std::env::var("MG_INSPECT").is_ok() {
        if let Some(multigrid) = study.multigrid() {
            inspect_aggregates(multigrid, &mesh);
        }
    }
    let trace_every = std::env::var("GATE_TRACE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let solve_start = Instant::now();
    let mut converged = false;
    let mut last_print = 0.0_f64;
    let mut next_trace = trace_every.unwrap_or(u64::MAX);
    loop {
        match study.step(SOLVE_SLICE, &mut || false) {
            Ok(StepOutcome::Converged(_)) => {
                converged = true;
                break;
            }
            Ok(StepOutcome::Running(_)) => {}
            Ok(StepOutcome::Cancelled(_)) => unreachable!("nothing cancels this solve"),
            Err(error) => return fail(&format!("solve failed: {error}")),
        }
        let spent = solve_start.elapsed().as_secs_f64();
        if let Some(every) = trace_every {
            let r = study.report();
            if r.iterations >= next_trace {
                println!(
                    "  iteration {}: residual {:.3e}, {spent:.1} s",
                    r.iterations, r.relative_residual
                );
                next_trace = r.iterations + every;
            }
        }
        if spent - last_print >= 120.0 {
            let r = study.report();
            println!(
                "  ... solving: {spent:.0} s, iterations {}, residual {:.3e}",
                r.iterations, r.relative_residual
            );
            last_print = spent;
        }
        if spent >= cap {
            break;
        }
    }
    if !converged {
        let r = study.report();
        println!("VERDICT benchmark NOT ASSESSED: the solve hit the {cap} s cap unconverged.");
        println!("solver: {}", report_of(r));
        println!(
            "(total wall clock {:.1} s)",
            clock_start.elapsed().as_secs_f64()
        );
        return;
    }
    let solution = match study.solution(&mut || false) {
        Ok(solution) => solution,
        Err(error) => return fail(&format!("recovery failed: {error}")),
    };
    println!("solver: {}", report_of(solution.report));
    let verdict = benchmark(&mesh, &solution, nu, width);
    println!(
        "VERDICT contracts PASS; layers away from rim {}; benchmark {}",
        if layers_pass { "PASS" } else { "FAIL" },
        if nu != 0.0 {
            "information only (nu != 0)"
        } else if verdict < 0.01 {
            "PASS"
        } else {
            "FAIL"
        },
    );
    println!(
        "(total wall clock {:.1} s)",
        clock_start.elapsed().as_secs_f64()
    );
}

/// Prints the free-end deflection and mid-span stress against the references, returning the
/// deflection error.
fn benchmark(
    mesh: &solidfem::mesh::QuadraticTetMesh,
    solution: &Solution,
    nu: f64,
    width: f64,
) -> f64 {
    let area = width * THICK;
    let inertia = width * THICK * THICK * THICK / 12.0;
    let e = YOUNGS_MODULUS;
    let g = e / (2.0 * (1.0 + nu));
    let kappa = 5.0 / 6.0;
    let bending = LOAD * LENGTH.powi(3) / (3.0 * e * inertia);
    let shear = LOAD * LENGTH / (kappa * g * area);
    let expected = bending + shear;
    let (mut sum, mut count) = (0.0, 0usize);
    for (node, position) in mesh.nodes().iter().enumerate() {
        if (position[0] - LENGTH).abs() < 1e-9 {
            sum += -solution.displacements[node][2];
            count += 1;
        }
    }
    let deflection = sum / count as f64;
    let error = (deflection - expected).abs() / expected;
    println!(
        "reference free-end deflection {expected:.6e} (bending {bending:.6e}, shear {shear:.6e}, {:.4}% of total); measured mean over {count} end nodes {deflection:.6e}, error {:.3}%",
        100.0 * shear / expected,
        error * 100.0
    );
    if nu != 0.0 {
        let wide = LOAD * LENGTH.powi(3) * (1.0 - nu * nu) / (3.0 * e * inertia) + shear;
        println!(
            "nu {nu}: information only. Plate-strip (anticlastic restrained) reference {wide:.6e}, measured against it {:+.3}%; beam reference above",
            100.0 * (deflection - wide) / wide
        );
    }
    // Mid-span top-surface bending stress, from nodal von Mises at corner nodes.
    let expected_stress = LOAD * (LENGTH / 2.0) * (THICK / 2.0) / inertia;
    let (mut stress, mut samples) = (0.0, 0usize);
    let corners = mesh.vertex_count();
    let top = mesh.nodes().iter().map(|n| n[2]).fold(f64::MIN, f64::max);
    for (node, position) in mesh.nodes()[..corners].iter().enumerate() {
        if (position[2] - top).abs() < 1e-9
            && (position[0] - LENGTH / 2.0).abs() < 1.5
            && position[1] > 0.1 * width
            && position[1] < 0.9 * width
        {
            stress += solution.nodal_von_mises[node];
            samples += 1;
        }
    }
    if samples > 0 {
        let measured = stress / samples as f64;
        println!(
            "mid-span top-surface bending stress: reference {expected_stress:.6e}, measured mean over {samples} corner nodes {measured:.6e}, error {:.3}%",
            100.0 * (measured - expected_stress).abs() / expected_stress
        );
    }
    error
}

fn fail(message: &str) {
    println!("VERDICT FAIL: {message}");
}

/// The mesher's contracts, read only from the arrays: surface vertices bit-identical, every tet
/// exactly positive, the volume the plate's, every added point strictly inside the box.
fn check_contracts(mesh: &SolidMesh, surface: &[[f64; 3]], width: f64) -> Result<(), String> {
    if &mesh.points[..surface.len()] != surface {
        return Err("a surface vertex changed".to_string());
    }
    let mut boundary_vertices = BTreeSet::new();
    for (t, tet) in mesh.tets.iter().enumerate() {
        let [a, b, c, d] = tet.map(|v| coord(mesh.points[v as usize]));
        if orient3d(a, b, c, d) >= 0.0 {
            return Err(format!("tet {t} {tet:?} is not exactly positive"));
        }
        for i in 0..4 {
            if mesh.neighbors[t][i] == HULL {
                for j in (0..4).filter(|&j| j != i) {
                    boundary_vertices.insert(tet[j]);
                }
            }
        }
    }
    if boundary_vertices.len() != surface.len()
        || boundary_vertices
            .iter()
            .any(|&v| v as usize >= surface.len())
    {
        return Err("boundary faces use a point that is not a surface vertex".to_string());
    }
    let volume: f64 = mesh
        .tets
        .iter()
        .map(|tet| {
            let [a, b, c, d] = tet.map(|v| mesh.points[v as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                + u[2] * (v[0] * w[1] - v[1] * w[0]))
                / 6.0
        })
        .sum();
    let thickness = surface.iter().map(|p| p[2]).fold(0.0, f64::max);
    let expected = LENGTH * width * thickness;
    // Each tet determinant accumulates at most a few dozen rounding units of the part size cubed.
    let tolerance = mesh.tets.len() as f64 * 64.0 * f64::EPSILON * (2.0 * LENGTH).powi(3);
    if (volume - expected).abs() > tolerance + 1e-9 * expected {
        return Err(format!("volume {volume} against the plate's {expected}"));
    }
    for (index, p) in mesh.points.iter().enumerate().skip(surface.len()) {
        if !(p[0] > 0.0
            && p[0] < LENGTH
            && p[1] > 0.0
            && p[1] < width
            && p[2] > 0.0
            && p[2] < thickness)
        {
            return Err(format!(
                "added point {index} at {p:?} is not strictly inside the plate"
            ));
        }
    }
    Ok(())
}

struct Layers {
    margin: f64,
    away_columns: usize,
    away_min_tets: usize,
    away_crossing: usize,
    away_largest_span: f64,
    away_missing: usize,
    rim_columns: usize,
    rim_min_tets: usize,
    rim_crossing: usize,
    rim_largest_span: f64,
    rim_missing: usize,
}

/// Walks columns across the wall on a jittered grid, measuring tets traversed and their span in z.
fn measure_layers(mesh: &SolidMesh, edge: f64, thick: f64, width: f64) -> Layers {
    let margin = 2.0 * edge;
    let bottom = bottom_faces(mesh, edge);
    let mut layers = Layers {
        margin,
        away_columns: 0,
        away_min_tets: usize::MAX,
        away_crossing: 0,
        away_largest_span: 0.0,
        away_missing: 0,
        rim_columns: 0,
        rim_min_tets: usize::MAX,
        rim_crossing: 0,
        rim_largest_span: 0.0,
        rim_missing: 0,
    };
    let columns = 60usize;
    let tolerance = 0.5 * thick * (1.0 + 1e-9);
    for i in 0..columns {
        for j in 0..columns {
            let x = (i as f64 + (0.31 + 0.617 * j as f64).fract() * 0.8 + 0.1) * LENGTH
                / columns as f64;
            let y =
                (j as f64 + (0.27 + 0.617 * i as f64).fract() * 0.8 + 0.1) * width / columns as f64;
            let at_rim = x < margin || x > LENGTH - margin || y < margin || y > width - margin;
            match column_walk(mesh, &bottom, x, y, thick) {
                None => {
                    if at_rim {
                        layers.rim_missing += 1;
                    } else {
                        layers.away_missing += 1;
                    }
                }
                Some(walk) => {
                    let crossing = walk.iter().any(|&span| span > tolerance);
                    let largest = walk.iter().fold(0.0_f64, |a, &b| a.max(b));
                    if at_rim {
                        layers.rim_columns += 1;
                        layers.rim_min_tets = layers.rim_min_tets.min(walk.len());
                        layers.rim_crossing += usize::from(crossing);
                        layers.rim_largest_span = layers.rim_largest_span.max(largest);
                    } else {
                        layers.away_columns += 1;
                        layers.away_min_tets = layers.away_min_tets.min(walk.len());
                        layers.away_crossing += usize::from(crossing);
                        layers.away_largest_span = layers.away_largest_span.max(largest);
                    }
                }
            }
        }
    }
    layers
}

type Buckets = HashMap<(i32, i32), Vec<(usize, usize)>>;

/// The tets on the z = 0 face of the plate, indexed by the cells their faces overlap.
fn bottom_faces(mesh: &SolidMesh, edge: f64) -> (Buckets, f64) {
    let cell = 2.0 * edge;
    let mut buckets: Buckets = HashMap::new();
    let key = |v: f64| (v / cell).floor() as i32;
    for (t, tet) in mesh.tets.iter().enumerate() {
        for i in 0..4 {
            if mesh.neighbors[t][i] != HULL {
                continue;
            }
            let face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
            if face.iter().any(|&v| mesh.points[v as usize][2] != 0.0) {
                continue;
            }
            let (xs, ys): (Vec<f64>, Vec<f64>) = face
                .iter()
                .map(|&v| (mesh.points[v as usize][0], mesh.points[v as usize][1]))
                .unzip();
            let (x0, x1) = (
                xs.iter().copied().fold(f64::MAX, f64::min),
                xs.iter().copied().fold(f64::MIN, f64::max),
            );
            let (y0, y1) = (
                ys.iter().copied().fold(f64::MAX, f64::min),
                ys.iter().copied().fold(f64::MIN, f64::max),
            );
            for ix in key(x0)..=key(x1) {
                for iy in key(y0)..=key(y1) {
                    buckets.entry((ix, iy)).or_default().push((t, i));
                }
            }
        }
    }
    (buckets, cell)
}

/// The z-span of each tetrahedron a vertical line through `(x, y)` crosses, from the bottom skin
/// to the top, by following neighbour links. `None` when the line is not entered from the bottom.
fn column_walk(
    mesh: &SolidMesh,
    bottom: &(Buckets, f64),
    x: f64,
    y: f64,
    thick: f64,
) -> Option<Vec<f64>> {
    let (buckets, cell) = bottom;
    let low = [x, y, -1.0];
    let high = [x, y, thick + 1.0];
    let key = ((x / cell).floor() as i32, (y / cell).floor() as i32);
    let &(mut t, mut through) = buckets
        .get(&key)?
        .iter()
        .find(|&&(t, i)| crosses(mesh, mesh.tets[t], i, low, high))?;
    let mut spans = Vec::new();
    for _ in 0..mesh.tets.len() {
        let tet = mesh.tets[t];
        let zs = tet.map(|v| mesh.points[v as usize][2]);
        let span = zs.iter().copied().fold(f64::MIN, f64::max)
            - zs.iter().copied().fold(f64::MAX, f64::min);
        spans.push(span);
        let exit = (0..4).find(|&i| i != through && crosses(mesh, tet, i, low, high))?;
        let next = mesh.neighbors[t][exit];
        if next == HULL {
            return Some(spans);
        }
        let back = (0..4).find(|&i| !tet.contains(&mesh.tets[next as usize][i]))?;
        t = next as usize;
        through = back;
    }
    None
}

fn crosses(mesh: &SolidMesh, tet: [u32; 4], skip: usize, low: [f64; 3], high: [f64; 3]) -> bool {
    let v: Vec<[f64; 3]> = (0..4)
        .filter(|&i| i != skip)
        .map(|i| mesh.points[tet[i] as usize])
        .collect();
    let (a, b, c) = (coord(v[0]), coord(v[1]), coord(v[2]));
    let (s1, s2) = (
        orient3d(a, b, c, coord(low)),
        orient3d(a, b, c, coord(high)),
    );
    if s1 * s2 >= 0.0 {
        return false;
    }
    let sides = [
        orient3d(coord(low), coord(high), a, b),
        orient3d(coord(low), coord(high), b, c),
        orient3d(coord(low), coord(high), c, a),
    ];
    sides.iter().all(|&s| s > 0.0) || sides.iter().all(|&s| s < 0.0)
}
