//! The demo page's WebAssembly bindings: a visitor picks a part, holds it and loads it, and sees
//! the stress and how close the result is to the textbook while the solve runs.
//!
//! The page drives one [`Demo`] object from a web worker or the main thread: [`Demo::start`]
//! plans and begins a study (a new start cancels the running one at once, since nothing runs
//! between calls), [`Demo::step`] advances it by a small budget per frame, and the getters copy
//! the current estimate out as typed arrays and rows. Everything visible to JavaScript is here;
//! `catalog.rs` holds what is offered, `section.rs` and `plan.rs` the textbook, `part.rs` the
//! mesh, `engine.rs` the slicing, `view.rs` the render arrays and `validity.rs` whether a result
//! is inside what the linear model can describe, so the native tests run the same code the
//! browser does.

pub mod catalog;
pub mod engine;
pub mod part;
pub mod plan;
pub mod section;
pub mod validity;
pub mod view;

use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use catalog::{LoadKind, Restraint, SectionKind, StudyRequest, DISPLAY_NAME, MATERIALS};
use engine::{DemoStudy, StudyState};
use plan::{PlanError, Textbook};

/// Installs the panic hook once.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// The display name, for the page title.
#[wasm_bindgen(js_name = displayName)]
pub fn display_name() -> String {
    DISPLAY_NAME.to_string()
}

/// JSON text: the materials, section kinds, restraints, loads and the allowed ranges for each
/// combination, so the page can build its controls from what the engine offers.
#[wasm_bindgen(js_name = catalogJson)]
pub fn catalog_json() -> String {
    let mut materials = Vec::new();
    for m in MATERIALS {
        materials.push(format!(
            "{{\"name\":{},\"youngsModulus\":{},\"poissonRatio\":{},\"yieldStrength\":{}}}",
            json_string(m.name),
            m.youngs_modulus,
            m.poisson_ratio,
            m.yield_strength
        ));
    }
    let mut combinations = Vec::new();
    for c in catalog::offered_combinations() {
        let r = catalog::ranges(c.section, c.restraint, c.load);
        let wall = r.wall_fraction.map_or("null".to_string(), |w| {
            format!("{{\"min\":{},\"max\":{}}}", w.min, w.max)
        });
        let textbook = c.section == SectionKind::Solid || c.load == LoadKind::Push;
        combinations.push(format!(
            "{{\"section\":{},\"restraint\":{},\"load\":{},\"hasTextbookAnswer\":{},\
             \"length\":{{\"min\":{},\"max\":{}}},\"width\":{{\"min\":{},\"max\":{}}},\
             \"height\":{{\"min\":{},\"max\":{}}},\"wallFraction\":{},\
             \"magnitude\":{{\"min\":{},\"max\":{}}}}}",
            json_string(section_name(c.section)),
            json_string(restraint_name(c.restraint)),
            json_string(load_name(c.load)),
            textbook,
            r.length.min,
            r.length.max,
            r.width.min,
            r.width.max,
            r.height.min,
            r.height.max,
            wall,
            r.magnitude.min,
            r.magnitude.max
        ));
    }
    format!(
        "{{\"displayName\":{},\"maxElements\":{},\"materials\":[{}],\"combinations\":[{}]}}",
        json_string(DISPLAY_NAME),
        plan::MAX_ELEMENTS,
        materials.join(","),
        combinations.join(",")
    )
}

/// The study the page is driving.
#[wasm_bindgen]
pub struct Demo {
    study: Option<DemoStudy>,
    last_refusal: Option<String>,
}

impl Default for Demo {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Demo {
    /// A demo with no study yet.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Demo {
        Demo {
            study: None,
            last_refusal: None,
        }
    }

    /// Starts a new study, cancelling any running one at once. Section is `"solid"` or `"tube"`;
    /// restraint `"one_end"` or `"both_ends"`; load `"push"` or `"twist"`; `wall_fraction` is the
    /// wall as a fraction of the smaller side (solid: ignored).
    ///
    /// Returns the state name: `"assembling"` when it began, `"too_large"` when the mesh would
    /// exceed the size the demo can solve in time (read [`Demo::refusal`] for the sentence). Bad
    /// numbers or an unknown name return an error naming the value.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        section: &str,
        length: f64,
        width: f64,
        height: f64,
        wall_fraction: f64,
        material: usize,
        restraint: &str,
        load: &str,
        magnitude: f64,
    ) -> Result<String, JsValue> {
        self.study = None;
        self.last_refusal = None;
        let request = StudyRequest {
            section: parse_section(section)?,
            length,
            width,
            height,
            wall_fraction,
            material,
            restraint: parse_restraint(restraint)?,
            load: parse_load(load)?,
            magnitude,
        };
        match DemoStudy::start(request) {
            Ok(study) => {
                let state = state_name(study.state());
                self.study = Some(study);
                Ok(state.to_string())
            }
            Err(error @ PlanError::TooLarge { .. }) => {
                self.last_refusal = Some(error.to_string());
                Ok("too_large".to_string())
            }
            Err(error) => Err(JsValue::from_str(&error.to_string())),
        }
    }

    /// Runs one slice with `budget` units of work (see `engine.rs`); returns the state name:
    /// `"assembling"`, `"solving"`, `"recovering"`, `"done"`, `"cancelled"`, `"failed"`, or
    /// `"idle"` with no study.
    pub fn step(&mut self, budget: u32) -> String {
        self.study.as_mut().map_or("idle".to_string(), |s| {
            state_name(s.step(budget)).to_string()
        })
    }

    /// Cancels the running study; a new `start` replaces it at any time anyway.
    pub fn cancel(&mut self) {
        if let Some(study) = self.study.as_mut() {
            study.cancel();
        }
    }

    /// The reason for the last `"too_large"` state, if any.
    pub fn refusal(&self) -> Option<String> {
        self.last_refusal.clone()
    }

    /// JSON text of the current estimate's numbers: state, iterations, residual, element and node
    /// counts, min/max stress, max displacement, peak stress outside the clamps, safety factor
    /// (`null` before there is stress), `validity` (whether the result is inside what the linear
    /// model can describe: `withinModel`, `exceedsYield`, `excessiveDeflection`, `excessiveTwist`,
    /// and `deflectionLimit`, `twistRadians`, `twistLimitRadians`, each `null` where it does not
    /// apply; see `validity.rs`) and either `rows` (quantity, unit, place, fea, textbook,
    /// percentDifference) or `textbook: "none"` with `explanation` when the combination has no
    /// textbook answer. Failure gives `failure` with the message.
    #[wasm_bindgen(js_name = summaryJson)]
    pub fn summary_json(&self) -> String {
        let Some(study) = self.study.as_ref() else {
            return "{\"state\":\"idle\"}".to_string();
        };
        let view = study.view();
        let (elements, nodes) = study.mesh_size();
        let safety = view.safety_factor.map_or("null".to_string(), json_number);
        let optional = |value: Option<f64>| value.map_or("null".to_string(), json_number);
        let validity = study.validity(&view);
        let validity = format!(
            "{{\"withinModel\":{},\"exceedsYield\":{},\"excessiveDeflection\":{},\
             \"excessiveTwist\":{},\"deflectionLimit\":{},\"twistRadians\":{},\
             \"twistLimitRadians\":{}}}",
            validity.within_model(),
            validity.exceeds_yield,
            validity.excessive_deflection,
            validity.excessive_twist,
            optional(validity.deflection_limit),
            optional(validity.twist_radians),
            optional(validity.twist_limit_radians),
        );
        let textbook = match study.textbook() {
            Textbook::Answers(_) => {
                let rows: Vec<String> = study
                    .rows()
                    .iter()
                    .map(|row| {
                        format!(
                            "{{\"quantity\":{},\"unit\":{},\"place\":{},\"fea\":{},\"textbook\":{},\"percentDifference\":{}}}",
                            json_string(row.quantity),
                            json_string(row.unit),
                            json_string(&row.place),
                            json_number(row.fea),
                            json_number(row.textbook),
                            json_number(row.percent_difference)
                        )
                    })
                    .collect();
                format!("\"textbook\":\"compared\",\"rows\":[{}]", rows.join(","))
            }
            Textbook::NoTextbook(reason) => {
                format!(
                    "\"textbook\":\"none\",\"explanation\":{}",
                    json_string(reason.explanation())
                )
            }
        };
        format!(
            "{{\"state\":{},\"iterations\":{},\"residual\":{},\"elements\":{elements},\"nodes\":{nodes},\
             \"minStress\":{},\"maxStress\":{},\"maxDisplacement\":{},\"peakStressOutsideClamps\":{},\
             \"safetyFactor\":{safety},\"validity\":{validity},\"failure\":{},{textbook}}}",
            json_string(state_name(study.state())),
            study.iterations(),
            json_number(study.residual()),
            json_number(view.min_stress),
            json_number(view.max_stress),
            json_number(view.max_displacement),
            json_number(view.peak_stress_outside_clamps),
            study.failure().map_or("null".to_string(), json_string),
        )
    }

    /// Surface vertex positions, three per vertex, in mm.
    pub fn positions(&self) -> Float32Array {
        self.array(|v| &v.positions)
    }

    /// Triangle vertex indices, three per triangle.
    pub fn indices(&self) -> Uint32Array {
        let view = self.study.as_ref().map(|s| s.view());
        let empty = Vec::new();
        Uint32Array::from(view.as_ref().map_or(empty.as_slice(), |v| &v.indices[..]))
    }

    /// Von Mises stress in MPa per vertex (the latest finished recovery of the estimate).
    #[wasm_bindgen(js_name = vonMises)]
    pub fn von_mises(&self) -> Float32Array {
        self.array(|v| &v.von_mises)
    }

    /// Displacement in mm, three per vertex, at the current estimate.
    pub fn displacements(&self) -> Float32Array {
        self.array(|v| &v.displacement)
    }

    fn array(&self, pick: impl Fn(&view::ViewData) -> &Vec<f32>) -> Float32Array {
        let view = self.study.as_ref().map(|s| s.view());
        let empty = Vec::new();
        Float32Array::from(view.as_ref().map_or(empty.as_slice(), |v| pick(v)))
    }
}

fn state_name(state: StudyState) -> &'static str {
    match state {
        StudyState::Assembling => "assembling",
        StudyState::Solving => "solving",
        StudyState::Recovering => "recovering",
        StudyState::Done => "done",
        StudyState::Cancelled => "cancelled",
        StudyState::Failed => "failed",
    }
}

fn section_name(section: SectionKind) -> &'static str {
    match section {
        SectionKind::Solid => "solid",
        SectionKind::Tube => "tube",
    }
}

fn restraint_name(restraint: Restraint) -> &'static str {
    match restraint {
        Restraint::FixedOneEnd => "one_end",
        Restraint::FixedBothEnds => "both_ends",
    }
}

fn load_name(load: LoadKind) -> &'static str {
    match load {
        LoadKind::Push => "push",
        LoadKind::Twist => "twist",
    }
}

fn parse_section(name: &str) -> Result<SectionKind, JsValue> {
    match name {
        "solid" => Ok(SectionKind::Solid),
        "tube" => Ok(SectionKind::Tube),
        other => Err(JsValue::from_str(&format!("unknown section {other:?}"))),
    }
}

fn parse_restraint(name: &str) -> Result<Restraint, JsValue> {
    match name {
        "one_end" => Ok(Restraint::FixedOneEnd),
        "both_ends" => Ok(Restraint::FixedBothEnds),
        other => Err(JsValue::from_str(&format!("unknown restraint {other:?}"))),
    }
}

fn parse_load(name: &str) -> Result<LoadKind, JsValue> {
    match name {
        "push" => Ok(LoadKind::Push),
        "twist" => Ok(LoadKind::Twist),
        other => Err(JsValue::from_str(&format!("unknown load {other:?}"))),
    }
}

/// A finite number as JSON; a non-finite one (the safety factor before any stress) as `null`.
fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

/// A string as JSON text.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
