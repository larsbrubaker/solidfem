//! The generator's table agrees with the benchmark cases that read its files: every file a case
//! names is one the generator writes, with the same `Sizing`, and nothing is generated that no
//! case reads. A size function differing between the surface and the volume mesh would quietly
//! change every result it touched. Also holds what `select` promises a user on the command line.

use solidfem_bench::benchmarks::cases::all_cases;
use solidfem_bench::surfaces::{all_parts, select, SurfaceError};

#[test]
fn every_case_file_is_generated_with_the_same_sizing() {
    let parts = all_parts();
    let generated: Vec<_> = parts
        .iter()
        .flat_map(|part| part.densities.iter())
        .collect();
    for case in all_cases() {
        for density in &case.densities {
            let stem = density
                .surface_file
                .strip_suffix(".txt")
                .expect("a case's surface file is named <stem>.txt");
            let source = generated
                .iter()
                .find(|d| d.stem == stem)
                .unwrap_or_else(|| panic!("case {} reads {stem}, which is not generated", case.id));
            assert_eq!(
                source.size, density.size,
                "case {}, {stem}: the surface's size function differs from the volume mesh's",
                case.id
            );
        }
    }
}

#[test]
fn nothing_is_generated_that_no_case_reads() {
    let cases = all_cases();
    let read: Vec<String> = cases
        .iter()
        .flat_map(|case| case.densities.iter().map(|d| d.surface_file.clone()))
        .collect();
    for part in all_parts() {
        for density in &part.densities {
            assert!(
                read.contains(&format!("{}.txt", density.stem)),
                "{} is generated but no benchmark case reads it",
                density.stem
            );
        }
    }
}

#[test]
fn every_stem_is_unique_and_starts_with_its_part() {
    let mut stems = Vec::new();
    for part in all_parts() {
        assert_eq!(part.densities.len(), 3, "{} densities", part.name);
        for density in &part.densities {
            assert!(
                density.stem.starts_with(&format!("{}_", part.name)),
                "{} is not named after {}",
                density.stem,
                part.name
            );
            stems.push(density.stem.clone());
        }
    }
    let count = stems.len();
    stems.sort();
    stems.dedup();
    assert_eq!(stems.len(), count, "two densities share a file name");
}

#[test]
fn no_names_selects_every_density() {
    let everything: usize = all_parts().iter().map(|p| p.densities.len()).sum();
    let selected: usize = select(&[])
        .expect("selecting nothing is valid")
        .iter()
        .map(|p| p.densities.len())
        .sum();
    assert_eq!(selected, everything);
}

#[test]
fn a_part_name_is_exact_so_le1_does_not_pull_in_le10() {
    let parts = select(&["le1".to_string()]).expect("le1 is a part");
    let names: Vec<_> = parts.iter().map(|p| p.name).collect();
    assert_eq!(names, ["le1"]);
    assert_eq!(parts[0].densities.len(), 3);
}

#[test]
fn a_stem_selects_only_that_density() {
    let parts = select(&["gear_n0.6".to_string()]).expect("gear_n0.6 is a density");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].name, "gear");
    let stems: Vec<_> = parts[0].densities.iter().map(|d| d.stem.as_str()).collect();
    assert_eq!(stems, ["gear_n0.6"]);
}

#[test]
fn an_unknown_name_is_refused_and_lists_the_real_ones() {
    let Err(error) = select(&["gearbox".to_string()]) else {
        panic!("an unknown name must be refused");
    };
    assert!(matches!(&error, SurfaceError::UnknownName { name, .. } if name == "gearbox"));
    let text = error.to_string();
    assert!(text.contains("gearbox"), "{text}");
    assert!(text.contains("gear_n0.6"), "{text}");
    assert!(text.contains("le10"), "{text}");
}
