//! What the demo offers and only that: the materials, sections, restraints and load kinds, the
//! parameter ranges the tests prove accurate, and the [`StudyRequest`] a page sends. `plan.rs`
//! turns a request into a mesh and a textbook comparison; `tests/demo/accuracy.rs` enumerates
//! every combination from here, so what cannot be proved accurate cannot be offered.

use crate::section::Section;

/// The demo's working title, in one place so a rename touches one line.
pub const DISPLAY_NAME: &str = "solidfem";

/// One material of the small table: typical values, not a data sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoMaterial {
    pub name: &'static str,
    /// Young's modulus in MPa.
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    /// Yield strength in MPa.
    pub yield_strength: f64,
}

/// Typical properties of common materials: printed plastics are process dependent, so these are
/// labelled typical, and the safety factor is an estimate.
pub const MATERIALS: [DemoMaterial; 6] = [
    DemoMaterial {
        name: "PLA (typical)",
        youngs_modulus: 3500.0,
        poisson_ratio: 0.36,
        yield_strength: 50.0,
    },
    DemoMaterial {
        name: "PETG (typical)",
        youngs_modulus: 2100.0,
        poisson_ratio: 0.38,
        yield_strength: 45.0,
    },
    DemoMaterial {
        name: "ABS (typical)",
        youngs_modulus: 2200.0,
        poisson_ratio: 0.35,
        yield_strength: 40.0,
    },
    DemoMaterial {
        name: "Nylon (typical)",
        youngs_modulus: 1800.0,
        poisson_ratio: 0.39,
        yield_strength: 50.0,
    },
    DemoMaterial {
        name: "Aluminium 6061 (typical)",
        youngs_modulus: 69_000.0,
        poisson_ratio: 0.33,
        yield_strength: 275.0,
    },
    DemoMaterial {
        name: "Steel, mild (typical)",
        youngs_modulus: 200_000.0,
        poisson_ratio: 0.29,
        yield_strength: 250.0,
    },
];

/// Which section outline a visitor chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    Solid,
    Tube,
}

/// How the part is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restraint {
    FixedOneEnd,
    FixedBothEnds,
}

/// What is applied. Both act at the free end of a one-end-fixed bar and at mid-span of a
/// both-ends-fixed bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    /// A transverse force (N) along -z.
    Push,
    /// A torque (N mm) about the bar axis.
    Twist,
}

/// Element density: more cells through each wall and along the bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    Standard,
    Fine,
}

/// One closed range a parameter must lie in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
}

impl Range {
    /// The low, middle and high values the accuracy test visits.
    pub fn points(&self) -> [f64; 3] {
        [self.min, 0.5 * (self.min + self.max), self.max]
    }

    fn contains(&self, value: f64) -> bool {
        // A NaN fails both comparisons and is refused by the caller first.
        value >= self.min && value <= self.max
    }
}

/// The allowed ranges for a combination, in mm, N and N mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranges {
    pub length: Range,
    pub width: Range,
    pub height: Range,
    /// `None` for a solid section.
    pub wall_fraction: Option<Range>,
    pub magnitude: Range,
}

/// The ranges allowed for `(kind, restraint, load)`. These are restricted to where the accuracy
/// test passes: a bar at least 6 section depths long held at one end and at least 12 held at
/// both, where a beam formula applies, and a wall between 0.2 and 0.3 of the smaller side, so
/// each wall gets two square-ish elements through its thickness in the standard mesh.
pub fn ranges(kind: SectionKind, restraint: Restraint, load: LoadKind) -> Ranges {
    let length = match restraint {
        Restraint::FixedOneEnd => Range {
            min: 120.0,
            max: 240.0,
        },
        Restraint::FixedBothEnds => Range {
            min: 240.0,
            max: 360.0,
        },
    };
    // The tube's wall is at most 0.3 of the smaller side, so a narrower tube would need a finer
    // mesh per wall cell and exceed the size cap on the bar's length.
    let side = Range {
        min: if kind == SectionKind::Tube {
            14.0
        } else {
            10.0
        },
        max: 20.0,
    };
    Ranges {
        length,
        width: side,
        height: side,
        wall_fraction: (kind == SectionKind::Tube).then_some(Range { min: 0.2, max: 0.3 }),
        magnitude: match load {
            LoadKind::Push => Range {
                min: 1.0,
                max: 2000.0,
            },
            LoadKind::Twist => Range {
                min: 100.0,
                max: 200_000.0,
            },
        },
    }
}

/// A full request from the page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudyRequest {
    pub section: SectionKind,
    pub length: f64,
    pub width: f64,
    pub height: f64,
    /// Wall thickness as a fraction of the smaller side; ignored for a solid section.
    pub wall_fraction: f64,
    /// Index into [`MATERIALS`].
    pub material: usize,
    pub restraint: Restraint,
    pub load: LoadKind,
    pub magnitude: f64,
    pub detail: Detail,
}

/// Why a request was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestError {
    /// A number is NaN or infinite.
    NotANumber(&'static str),
    /// A value lies outside the range this combination allows; `allowed` names it.
    OutOfRange {
        parameter: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    /// The material index is not in [`MATERIALS`].
    UnknownMaterial(usize),
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::NotANumber(name) => write!(f, "{name} must be a finite number"),
            RequestError::OutOfRange {
                parameter,
                value,
                min,
                max,
            } => write!(
                f,
                "{parameter} {value} is outside the allowed range {min} to {max} for this choice"
            ),
            RequestError::UnknownMaterial(index) => {
                write!(f, "material {index} is not one of the offered materials")
            }
        }
    }
}

impl std::error::Error for RequestError {}

impl StudyRequest {
    /// The section this request builds, assuming [`Self::validate`] passed.
    pub fn section_shape(&self) -> Section {
        match self.section {
            SectionKind::Solid => Section::Solid {
                width: self.width,
                height: self.height,
            },
            SectionKind::Tube => Section::Tube {
                width: self.width,
                height: self.height,
                wall: self.wall_fraction * self.width.min(self.height),
            },
        }
    }

    /// Refuses non-finite and out-of-range values, naming the parameter so the page can say what
    /// to change.
    pub fn validate(&self) -> Result<(), RequestError> {
        let allowed = ranges(self.section, self.restraint, self.load);
        let check = |name: &'static str, value: f64, range: Range| {
            if !value.is_finite() {
                return Err(RequestError::NotANumber(name));
            }
            if !range.contains(value) {
                return Err(RequestError::OutOfRange {
                    parameter: name,
                    value,
                    min: range.min,
                    max: range.max,
                });
            }
            Ok(())
        };
        check("length", self.length, allowed.length)?;
        check("width", self.width, allowed.width)?;
        check("height", self.height, allowed.height)?;
        check("magnitude", self.magnitude, allowed.magnitude)?;
        if let Some(range) = allowed.wall_fraction {
            check("wall_fraction", self.wall_fraction, range)?;
        }
        if self.material >= MATERIALS.len() {
            return Err(RequestError::UnknownMaterial(self.material));
        }
        Ok(())
    }
}

/// One offered `(section, restraint, load)` combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combination {
    pub section: SectionKind,
    pub restraint: Restraint,
    pub load: LoadKind,
}

/// Every combination the page may ask for, whether it has a textbook answer or not. The accuracy
/// test walks this list.
pub fn offered_combinations() -> Vec<Combination> {
    let mut all = Vec::new();
    for section in [SectionKind::Solid, SectionKind::Tube] {
        for restraint in [Restraint::FixedOneEnd, Restraint::FixedBothEnds] {
            for load in [LoadKind::Push, LoadKind::Twist] {
                all.push(Combination {
                    section,
                    restraint,
                    load,
                });
            }
        }
    }
    all
}
