//! Prints what bad boundary triangles do to a 10-node stress result: every shape, location and
//! extent of `solidfem_bench::sliver` at each smallest-corner angle, against the unperturbed bar.
//! `solidfem-sliver [poisson]`, default 0.3. The thresholds the summary uses are stated in
//! the source and printed; nothing here asserts them.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::error::Error;

    use solidfem_bench::sliver::{
        build, measure, measure_quality, reference, solve, timoshenko, Extent, Location, Measured,
        Quality, Reference, Shape, Spec, DEPTH, DIVISIONS, FORCE, LENGTH, YOUNGS,
    };

    /// The smallest corners tried, in degrees.
    const ANGLES: [f64; 9] = [30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1];
    /// A measure is materially affected past these: tip deflection within 0.5%, peak stresses
    /// within 5%, iterations within 25%.
    const TIP_THRESHOLD: f64 = 0.5;
    const STRESS_THRESHOLD: f64 = 5.0;
    const ITERATION_THRESHOLD: f64 = 25.0;

    /// One case's outcome.
    enum Status {
        Solved(Box<(Quality, Measured)>),
        Invalid(String),
    }

    struct Row {
        spec: Spec,
        status: Status,
    }

    fn shape_name(shape: Shape) -> &'static str {
        match shape {
            Shape::Cap => "cap",
            Shape::Needle => "needle",
        }
    }

    fn location_name(location: Location) -> &'static str {
        match location {
            Location::NearRoot => "root",
            Location::NeutralAxis => "neutral",
        }
    }

    fn extent_name(extent: Extent) -> &'static str {
        match extent {
            Extent::Single => "single",
            Extent::Cluster => "cluster10",
        }
    }

    fn run_case(spec: Spec, poisson: f64, reference: &Reference) -> Row {
        let distorted = match build(spec) {
            Ok(distorted) => distorted,
            Err(error) => {
                return Row {
                    spec,
                    status: Status::Invalid(format!("mesh refused: {error}")),
                };
            }
        };
        let quality = measure_quality(&distorted.linear, &distorted.quadratic, spec.target_degrees);
        match solve(&distorted.quadratic, poisson) {
            Ok(solution) => Row {
                spec,
                status: Status::Solved(Box::new((
                    quality,
                    measure(reference, &distorted, &solution),
                ))),
            },
            Err(error) => Row {
                spec,
                status: Status::Invalid(format!("solve refused: {error}")),
            },
        }
    }

    fn print_row(row: &Row, iterations: u64) {
        let s = row.spec;
        let name = format!(
            "{:<6} {:<7} {:<9} {:>5}",
            shape_name(s.shape),
            location_name(s.location),
            extent_name(s.extent),
            s.target_degrees
        );
        match &row.status {
            Status::Invalid(why) => println!("{name} | INVALID: {why}"),
            Status::Solved(boxed) => {
                let (q, m) = (boxed.0, boxed.1);
                let flags = flags(m, iterations);
                let peak_at = if m.peak_in_bad_element {
                    "bad".to_string()
                } else {
                    format!("x={:.2}", m.peak_x)
                };
                println!(
                    "{name} | {:>7.3} {:>3} | {:>6.2} {:>6.2} | {:>6.4} {:>6.4} | {:>+8.3} | {:>+9.2} {:>7} | {:>+7.2} {:>+9.2} | {:>7.2} {:>7.2} {:>6.3} | {:>7.2} {:>6.2} | {:>5} {:>8.2e} {} | {}",
                    q.min_corner,
                    q.low_triangles,
                    q.min_dihedral,
                    q.max_dihedral,
                    q.scaled_jacobian,
                    q.jacobian_ratio,
                    m.tip_change,
                    m.peak_change,
                    peak_at,
                    m.peak_excluded_change,
                    m.nodal_peak_change,
                    m.near_unmoved_change,
                    m.near_change,
                    m.near_absolute,
                    m.element_change,
                    m.element_absolute,
                    m.iterations,
                    m.residual,
                    if m.at_floor { "F" } else { " " },
                    flags,
                );
            }
        }
    }

    /// Letters for each threshold passed: T tip, P whole-part peak, E peak excluding bad elements,
    /// D display peak, N per-node at unmoved nodes near the bad triangles, X raw element stress in
    /// the touching elements, I iterations.
    fn flags(m: Measured, iterations: u64) -> String {
        let mut out = String::new();
        if m.tip_change.abs() > TIP_THRESHOLD {
            out.push('T');
        }
        if m.peak_change.abs() > STRESS_THRESHOLD {
            out.push('P');
        }
        if m.peak_excluded_change.abs() > STRESS_THRESHOLD {
            out.push('E');
        }
        if m.nodal_peak_change.abs() > STRESS_THRESHOLD {
            out.push('D');
        }
        if m.near_unmoved_change > STRESS_THRESHOLD {
            out.push('N');
        }
        if m.element_change > STRESS_THRESHOLD {
            out.push('X');
        }
        let growth = 100.0 * (m.iterations as f64 - iterations as f64) / iterations as f64;
        if growth > ITERATION_THRESHOLD {
            out.push('I');
        }
        if out.is_empty() {
            out.push('-');
        }
        out
    }

    /// The largest angle at which `crosses` was true, walking down the angles, and whether a case
    /// that crossed then came back under the threshold at a smaller angle (no monotone statement
    /// holds then).
    fn first_crossing(rows: &[&Row], crosses: impl Fn(&Row) -> bool) -> String {
        let mut largest: Option<f64> = None;
        let mut any_clear_after = false;
        for row in rows {
            if crosses(row) {
                if largest.is_none() {
                    largest = Some(row.spec.target_degrees);
                }
            } else if largest.is_some() {
                any_clear_after = true;
            }
        }
        match largest {
            Some(a) if any_clear_after => format!("{a} (not monotone)"),
            Some(a) => format!("{a}"),
            None => "never".to_string(),
        }
    }

    fn print_summary(rows: &[Row], iterations: u64) {
        println!(
            "\nthresholds: tip {TIP_THRESHOLD}%, stress {STRESS_THRESHOLD}%, iterations +{ITERATION_THRESHOLD}%; \
             each row is the largest target angle at which the measure crossed (angles fall from \
             left to right in the table), or 'never'; invalid cases count as crossed everywhere"
        );
        println!(
            "{:<6} {:<7} {:<9} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
            "shape",
            "where",
            "extent",
            "tip",
            "peak",
            "peak excl",
            "display",
            "near",
            "element",
            "iters"
        );
        for shape in [Shape::Cap, Shape::Needle] {
            for location in [Location::NearRoot, Location::NeutralAxis] {
                for extent in [Extent::Single, Extent::Cluster] {
                    let group: Vec<&Row> = rows
                        .iter()
                        .filter(|r| {
                            r.spec.shape == shape
                                && r.spec.location == location
                                && r.spec.extent == extent
                        })
                        .collect();
                    let value = |crosses: &dyn Fn(Measured) -> bool| {
                        first_crossing(&group, |row| match &row.status {
                            Status::Solved(boxed) => crosses(boxed.1),
                            Status::Invalid(_) => true,
                        })
                    };
                    let tip = value(&|m| m.tip_change.abs() > TIP_THRESHOLD);
                    let peak = value(&|m| m.peak_change.abs() > STRESS_THRESHOLD);
                    let excluded = value(&|m| m.peak_excluded_change.abs() > STRESS_THRESHOLD);
                    let display = value(&|m| m.nodal_peak_change.abs() > STRESS_THRESHOLD);
                    let near = value(&|m| m.near_unmoved_change > STRESS_THRESHOLD);
                    let element = value(&|m| m.element_change > STRESS_THRESHOLD);
                    let iters = value(&|m| {
                        100.0 * (m.iterations as f64 - iterations as f64) / iterations as f64
                            > ITERATION_THRESHOLD
                    });
                    println!(
                        "{:<6} {:<7} {:<9} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                        shape_name(shape),
                        location_name(location),
                        extent_name(extent),
                        tip,
                        peak,
                        excluded,
                        display,
                        near,
                        element,
                        iters
                    );
                }
            }
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let poisson: f64 = match std::env::args().nth(1) {
            Some(text) => text.parse()?,
            None => 0.3,
        };
        let start = std::time::Instant::now();
        let reference = reference(poisson)?;
        let closed_form = timoshenko(poisson);
        let peak_formula = FORCE * LENGTH * (0.5 * DEPTH) / (DEPTH.powi(4) / 12.0);
        let base_iterations = reference.solution.report.iterations;
        println!(
            "unperturbed bar {LENGTH} x {DEPTH} x {DEPTH}, {} x {} x {} cells, E={YOUNGS}, nu={poisson}: {} tets, {} free DOF, {} CG iterations, residual {:.2e}",
            DIVISIONS[0],
            DIVISIONS[1],
            DIVISIONS[2],
            reference.mesh.tets().len(),
            reference.solution.report.free_dofs,
            base_iterations,
            reference.solution.report.relative_residual,
        );
        println!(
            "tip deflection {:.6} (Timoshenko {:.6}, {:+.3}%); peak integration-point von Mises {:.3} at x={:.3} (beam Mc/I at root {:.3}, the clamp is a singularity); peak per-node {:.3}",
            reference.tip,
            closed_form,
            100.0 * (reference.tip - closed_form) / closed_form,
            reference.peak.0,
            reference.peak.1,
            peak_formula,
            reference.nodal_peak,
        );
        println!(
            "\ncolumns: shape where extent target | min corner (deg), triangles under 1.2*target | worst tet dihedral min, max (deg) | scaled Jacobian, element detJ min/max | tip % | whole-part peak %, where | peak excluding touching elements %, display peak % | per-node change near the bad triangle: at unmoved nodes %, with the reference interpolated at every node %, that as stress | raw integration-point stress in the touching elements against the unperturbed element stress at the same point %, as stress | CG iters, residual, F=at rounding floor | flags"
        );
        println!(
            "{:<6} {:<7} {:<9} {:>5} | {:>7} {:>3} | {:>6} {:>6} | {:>6} {:>6} | {:>8} | {:>9} {:>7} | {:>7} {:>9} | {:>7} {:>7} {:>6} | {:>7} {:>6} | {:>5} {:>8}   | flags",
            "shape", "where", "extent", "angle", "min deg", "low", "dihmin", "dihmax", "sJ", "detJr", "tip %", "peak %", "at", "excl %", "display%", "nearU %", "nearI %", "nearI", "elem %", "elem", "iters", "residual"
        );
        let mut rows = Vec::new();
        for shape in [Shape::Cap, Shape::Needle] {
            for location in [Location::NearRoot, Location::NeutralAxis] {
                for extent in [Extent::Single, Extent::Cluster] {
                    for angle in ANGLES {
                        let row = run_case(
                            Spec {
                                shape,
                                location,
                                extent,
                                target_degrees: angle,
                            },
                            poisson,
                            &reference,
                        );
                        print_row(&row, base_iterations);
                        rows.push(row);
                    }
                }
            }
        }
        print_summary(&rows, base_iterations);
        eprintln!(
            "{} cases in {:.1}s",
            rows.len(),
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = native::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
