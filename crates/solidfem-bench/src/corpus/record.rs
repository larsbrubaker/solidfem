//! The one-line records of a measured part (its input, its remesh and its volume mesh), and the
//! tables and summaries the binary prints from them. A record is a `key=value` list on one line,
//! so a child process can hand it to the parent that runs the corpus and a crashed child costs
//! only its own row. The input record's `sample` says whether the part was remeshed (`in`) or
//! left out of the sample and why (`input:<reason>`, `measure::sample_exclusion`); the remesh
//! record's `status` is `ok` or the remesher's refusal as `remesh:<variant>`, and its `walls`
//! whether two layers fit through the 5th-percentile wall at the edge chosen (`resolved`,
//! `under-resolved` with `wall_ratio` over 1 when the budget forced a coarser edge, or `unread`),
//! beside `pred_tets`, the element count predicted for that edge; the volume and
//! solve records' `status` is their stage's outcome name (`volume_trial::outcome_name`,
//! `solve_trial::solve_outcome_name`), or `child:<how>` when the child process was lost in that
//! stage. A solve record carries its mesh's worst dihedral angle so iterations can be read
//! against it (`by_dihedral_bucket`).

use std::collections::BTreeMap;

use super::measure::{sample_exclusion, InputMetrics};
use super::remesh_trial::{refusal_name, RemeshTrial, RemeshedPart, TargetSource};
use super::solve_trial::{solve_outcome_name, SolveFailure, SolveTrial};
use super::volume_trial::{outcome_name, VolumeTrial};

/// A parsed record: field name to text.
pub type Record = BTreeMap<String, String>;

/// Shortest text that still reads back to the same value; never a locale-dependent format.
fn num(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "nan".to_string()
    }
}

/// The `key=value` fields of a measured input.
pub fn input_fields(name: &str, m: &InputMetrics) -> Vec<(String, String)> {
    let t = &m.topology;
    let mut fields = vec![
        ("name", name.replace(' ', "_")),
        ("tris", m.triangles.to_string()),
        ("comps", t.components.to_string()),
        ("closed", t.is_closed().to_string()),
        ("oriented", t.is_manifold_and_oriented().to_string()),
        ("bedges", t.boundary_edges.to_string()),
        ("nmedges", t.nonmanifold_edges.to_string()),
        ("misedges", t.misoriented_edges.to_string()),
        ("xpairs", m.self_intersection.pairs.to_string()),
        ("xcapped", m.self_intersection.capped.to_string()),
        ("folds", m.folds.to_string()),
        ("degen", m.quality.degenerate.to_string()),
        ("armed", num(m.quality.aspect.median)),
        ("ar95", num(m.quality.aspect.p95)),
        ("ar99", num(m.quality.aspect.p99)),
        ("arworst", num(m.quality.aspect.worst)),
        ("minang", num(m.quality.min_angle_degrees)),
        ("emin", num(m.edge_min)),
        ("emed", num(m.edge_median)),
        ("emax", num(m.edge_max)),
        ("sharp", t.sharp_edges.len().to_string()),
        ("wallp5", num(m.wall.p5)),
        ("wallmin", num(m.wall.minimum)),
        ("diag", num(m.diagonal)),
        ("diag_wallp5", num(m.diagonal / m.wall.p5)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect::<Vec<_>>();
    fields.push(("kind".to_string(), "input".to_string()));
    let sample = match sample_exclusion(m) {
        Some(reason) => format!("input:{reason}"),
        None => "in".to_string(),
    };
    fields.push(("sample".to_string(), sample));
    fields
}

/// The `key=value` fields of a remesh trial.
pub fn remesh_fields(name: &str, trial: &RemeshTrial) -> Vec<(String, String)> {
    let source = match trial.target_source {
        TargetSource::ThinnestWall => "wall/2",
        TargetSource::Budget => "budget",
        TargetSource::CoarsestAllowed => "coarsest",
        TargetSource::NoWall => "no-wall",
    };
    let mut fields = vec![
        ("name".to_string(), name.replace(' ', "_")),
        ("kind".to_string(), "remesh".to_string()),
        ("h".to_string(), num(trial.target_edge_length)),
        ("hsrc".to_string(), source.to_string()),
        (
            "pred_tets".to_string(),
            num(trial.predicted_tets.round_ties_even()),
        ),
        (
            "walls".to_string(),
            walls_name(trial.wall_ratio).to_string(),
        ),
        (
            "wall_ratio".to_string(),
            trial.wall_ratio.map_or_else(|| "-".to_string(), num),
        ),
        ("dev".to_string(), num(trial.deviation_bound)),
        ("sec".to_string(), num(trial.seconds)),
    ];
    match &trial.outcome {
        Ok(part) => append_remeshed(&mut fields, part),
        Err(refusal) => {
            fields.push((
                "status".to_string(),
                format!("remesh:{}", refusal_name(refusal)),
            ));
            // The refusal's measured values, one token so the line still splits on spaces.
            let detail = format!("{refusal:?}").replace(char::is_whitespace, "");
            fields.push(("detail".to_string(), detail));
        }
    }
    fields
}

/// Whether two layers fit through the part's thin walls at the target: `resolved`,
/// `under-resolved` (the budget forced a coarser edge, a result on it is suspect), or `unread`
/// when no wall could be measured or the ratio is not a finite number.
fn walls_name(ratio: Option<f64>) -> &'static str {
    match ratio {
        None => "unread",
        // A ratio of exactly 1 is the wall-chosen edge itself; the tolerance only absorbs the
        // rounding of halving the wall and doubling it back.
        // A ratio that is not a number (the edge itself was not) says nothing about the walls.
        Some(r) if !r.is_finite() => "unread",
        Some(r) if r <= 1.0 + 1e-12 => "resolved",
        Some(_) => "under-resolved",
    }
}

fn append_remeshed(fields: &mut Vec<(String, String)>, part: &RemeshedPart) {
    fields.extend(
        [
            ("status", "ok".to_string()),
            ("tris_after", part.triangles.to_string()),
            ("closed_after", part.closed.to_string()),
            ("oriented_after", part.manifold_and_oriented.to_string()),
            ("vol_pct", num(part.volume_change_percent)),
            ("dev_out", num(part.deviation_output_to_input)),
            ("dev_in", num(part.deviation_input_to_output)),
            ("sharp_in", part.sharp_edges_in.to_string()),
            ("sharp_held", part.sharp_edges_held.to_string()),
            ("sharp_after", part.sharp_edges_after.to_string()),
            ("sharp_pct", num(part.sharp_survival_percent)),
            ("ar_med_after", num(part.quality.aspect.median)),
            ("ar_worst_after", num(part.quality.aspect.worst)),
            ("min_ang_after", num(part.quality.min_angle_degrees)),
            ("degen_after", part.quality.degenerate.to_string()),
            ("xpairs_after", part.self_intersection.pairs.to_string()),
            ("xcapped_after", part.self_intersection.capped.to_string()),
            ("folds_after", part.folds.to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v)),
    );
}

/// The `key=value` fields of a volume trial.
pub fn volume_fields(name: &str, trial: &VolumeTrial) -> Vec<(String, String)> {
    let mut fields = vec![
        ("name".to_string(), name.replace(' ', "_")),
        ("kind".to_string(), "volume".to_string()),
        ("vh".to_string(), num(trial.edge_length)),
        ("status".to_string(), outcome_name(&trial.outcome)),
        ("sec".to_string(), num(trial.seconds)),
    ];
    if let Ok(part) = &trial.outcome {
        fields.extend(
            [
                ("tets", part.tets.to_string()),
                ("nodes", part.nodes.to_string()),
                ("mindih", num(part.min_dihedral_degrees)),
                ("lt1", part.under_one_degree.to_string()),
                ("lt5", part.under_five_degrees.to_string()),
                (
                    "size_stopped",
                    if part.size_stopped { "yes" } else { "no" }.to_string(),
                ),
                ("refused_moves", part.refused_moves.to_string()),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v)),
        );
    }
    fields
}

/// The volume record of a part whose child process ended during the volume stage, `how` saying
/// how (`timeout` or `crashed`): the earlier records were already written, so the loss is the
/// volume stage's.
pub fn volume_lost_fields(name: &str, how: &str) -> Vec<(String, String)> {
    stage_lost_fields(name, "volume", how)
}

/// The record of kind `kind` for a part whose child process ended during that stage.
pub fn stage_lost_fields(name: &str, kind: &str, how: &str) -> Vec<(String, String)> {
    vec![
        ("name".to_string(), name.replace(' ', "_")),
        ("kind".to_string(), kind.to_string()),
        ("status".to_string(), format!("child:{how}")),
    ]
}

/// The `key=value` fields of a solve trial on a mesh whose worst dihedral angle is
/// `min_dihedral_degrees`.
pub fn solve_fields(
    name: &str,
    trial: &SolveTrial,
    min_dihedral_degrees: f64,
) -> Vec<(String, String)> {
    let mut fields = vec![
        ("name".to_string(), name.replace(' ', "_")),
        ("kind".to_string(), "solve".to_string()),
        ("status".to_string(), solve_outcome_name(&trial.outcome)),
        ("unknowns".to_string(), trial.unknowns.to_string()),
        ("mindih".to_string(), num(min_dihedral_degrees)),
        ("sec".to_string(), num(trial.seconds)),
    ];
    if let Some(report) = &trial.report {
        fields.extend([
            ("iters".to_string(), report.iterations.to_string()),
            ("resid".to_string(), num(report.relative_residual)),
            (
                "setup_sec".to_string(),
                num(report.assembly_seconds + report.preconditioner_seconds),
            ),
            ("solve_sec".to_string(), num(report.solve_seconds)),
        ]);
        if let Some(multigrid) = &report.multigrid {
            fields.push(("mg_bytes".to_string(), multigrid.memory_bytes.to_string()));
        }
    }
    let measured = match &trial.outcome {
        Ok(part) | Err(SolveFailure::Unphysical(part) | SolveFailure::Imbalanced(part)) => {
            Some(part)
        }
        Err(_) => None,
    };
    if let Some(part) = measured {
        fields.push(("imbalance".to_string(), num(part.imbalance)));
        fields.push((
            "umax_diag".to_string(),
            num(part.max_displacement_over_diagonal),
        ));
    }
    if let Err(failure @ (SolveFailure::Model(_) | SolveFailure::Reactions(_))) = &trial.outcome {
        // The error's own values, one token so the line still splits on spaces.
        let detail = format!("{failure:?}").replace(char::is_whitespace, "");
        fields.push(("detail".to_string(), detail));
    }
    fields
}

/// The worst-dihedral buckets solve iterations are read by, in degrees: `[low, high)`.
pub const DIHEDRAL_BUCKETS: [(&str, f64, f64); 4] = [
    ("< 0.1 deg", 0.0, 0.1),
    ("0.1 to 1 deg", 0.1, 1.0),
    ("1 to 5 deg", 1.0, 5.0),
    (">= 5 deg", 5.0, f64::INFINITY),
];

/// For each of [`DIHEDRAL_BUCKETS`], the `records` whose `mindih` falls in it, in order. A record
/// with no readable `mindih` is in none.
pub fn by_dihedral_bucket(records: &[Record]) -> Vec<(&'static str, Vec<Record>)> {
    DIHEDRAL_BUCKETS
        .iter()
        .map(|(label, low, high)| {
            let inside = records
                .iter()
                .filter(|r| field_f64(r, "mindih").is_some_and(|d| d >= *low && d < *high))
                .cloned()
                .collect();
            (*label, inside)
        })
        .collect()
}

/// How many `records` have each value of `key`, most frequent first, ties by value.
pub fn counts_by(records: &[Record], key: &str) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for record in records {
        *counts.entry(field(record, key)).or_insert(0) += 1;
    }
    let mut sorted: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(value, count)| (value.to_string(), count))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sorted
}

/// The minimum, 10th, 50th and 90th percentile and maximum of the numeric field `key`.
pub fn distribution(records: &[Record], key: &str) -> String {
    let mut values: Vec<f64> = records.iter().filter_map(|r| field_f64(r, key)).collect();
    values.sort_unstable_by(f64::total_cmp);
    let (Some(first), Some(last)) = (values.first(), values.last()) else {
        return "no values".to_string();
    };
    let at = |q: f64| super::measure::percentile(&values, q);
    format!(
        "min {first:.4}  p10 {:.4}  p50 {:.4}  p90 {:.4}  max {last:.4}  (n={})",
        at(0.1),
        at(0.5),
        at(0.9),
        values.len()
    )
}

/// [`distribution`] in scientific notation, for values such as residuals and imbalances whose
/// size is their exponent and that four decimals would print as zero.
pub fn distribution_sci(records: &[Record], key: &str) -> String {
    let mut values: Vec<f64> = records.iter().filter_map(|r| field_f64(r, key)).collect();
    values.sort_unstable_by(f64::total_cmp);
    let (Some(first), Some(last)) = (values.first(), values.last()) else {
        return "no values".to_string();
    };
    let at = |q: f64| super::measure::percentile(&values, q);
    format!(
        "min {first:.3e}  p10 {:.3e}  p50 {:.3e}  p90 {:.3e}  max {last:.3e}  (n={})",
        at(0.1),
        at(0.5),
        at(0.9),
        values.len()
    )
}

/// One `key=value` line from `fields`.
pub fn to_line(fields: &[(String, String)]) -> String {
    fields
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parses a `key=value` line.
pub fn parse_line(line: &str) -> Record {
    line.split_whitespace()
        .filter_map(|token| {
            let (k, v) = token.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

/// A numeric field, `None` when absent or not a finite number.
pub fn field_f64(record: &Record, key: &str) -> Option<f64> {
    record
        .get(key)
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

/// The value of `key`, or `-` when the record has none.
pub fn field<'a>(record: &'a Record, key: &str) -> &'a str {
    record.get(key).map_or("-", String::as_str)
}

/// `records` as a fixed-width table of `columns`. A cell is cut at 18 characters, except a
/// `status`, which is a failure class's name: cut short, two classes would read alike.
pub fn table(records: &[Record], columns: &[(&str, &str)]) -> String {
    let widths: Vec<usize> = columns
        .iter()
        .map(|(key, title)| {
            let cap = if *key == "status" { 48 } else { 18 };
            records
                .iter()
                .map(|r| field(r, key).len())
                .chain([title.len()])
                .max()
                .unwrap_or(0)
                .min(cap)
        })
        .collect();
    let mut out = String::new();
    let cells = columns
        .iter()
        .zip(&widths)
        .map(|((_, title), w)| format!("{title:<w$}"));
    out.push_str(&cells.collect::<Vec<_>>().join(" "));
    out.push('\n');
    for record in records {
        let cells = columns.iter().zip(&widths).map(|((key, _), w)| {
            let text = field(record, key);
            let text = text.get(..text.len().min(*w)).unwrap_or(text);
            format!("{text:<w$}")
        });
        out.push_str(&cells.collect::<Vec<_>>().join(" "));
        out.push('\n');
    }
    out
}

/// The share of `records` where `key` is `true`, as a percentage with its count.
pub fn share_true(records: &[Record], key: &str) -> String {
    let hits = records.iter().filter(|r| field(r, key) == "true").count();
    format!(
        "{hits}/{} ({:.0}%)",
        records.len(),
        percent(hits, records.len())
    )
}

/// The share of `records` where the numeric field `key` is above zero.
pub fn share_nonzero(records: &[Record], key: &str) -> String {
    let hits = records
        .iter()
        .filter(|r| field_f64(r, key).is_some_and(|v| v > 0.0))
        .count();
    format!(
        "{hits}/{} ({:.0}%)",
        records.len(),
        percent(hits, records.len())
    )
}

fn percent(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

/// The 10th, 50th and 90th percentile of the numeric field `key` over `records`.
pub fn spread(records: &[Record], key: &str) -> String {
    let mut values: Vec<f64> = records.iter().filter_map(|r| field_f64(r, key)).collect();
    values.sort_unstable_by(f64::total_cmp);
    let at = |q: f64| super::measure::percentile(&values, q);
    format!(
        "p10 {:.4e}  p50 {:.4e}  p90 {:.4e}",
        at(0.1),
        at(0.5),
        at(0.9)
    )
}
