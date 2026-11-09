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
//! browser does. `examples.rs` is the table of example parts and what the page says about each,
//! and `example_study.rs` carries one from its surface file to a [`engine::DemoStudy`], through
//! the benchmark case's own holds and loads ([`Demo::start_example`] starts one).

pub mod catalog;
pub mod engine;
pub mod example_study;
pub mod examples;
pub mod part;
pub mod plan;
pub mod section;
pub mod validity;
pub mod view;

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use catalog::{LoadKind, Restraint, SectionKind, StudyRequest, DISPLAY_NAME, MATERIALS};
use engine::{DemoStudy, StudyState};
use example_study::{ExampleMaterial, ExampleState, ExampleStudy};
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
        "{{\"displayName\":{},\"maxElements\":{},\"materials\":[{}],\"combinations\":[{}],\
         \"examples\":[{}]}}",
        json_string(DISPLAY_NAME),
        plan::MAX_ELEMENTS,
        materials.join(","),
        combinations.join(","),
        examples::EXAMPLES
            .iter()
            .map(example_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// One row of the example table as JSON.
fn example_json(part: &examples::ExamplePart) -> String {
    format!(
        "{{\"id\":{},\"name\":{},\"tests\":{},\"held\":{},\"loaded\":{},\"asset\":{},\
         \"load\":{},\"verdict\":{},\"status\":{},\"view\":[{},{},{}]}}",
        json_string(part.id),
        json_string(part.name),
        json_string(part.tests),
        json_string(part.held),
        json_string(part.loaded),
        json_string(part.surface_file),
        json_string(part.load),
        json_string(part.verdict.name()),
        json_string(part.status),
        json_number(part.view[0]),
        json_number(part.view[1]),
        json_number(part.view[2]),
    )
}

/// What the page is driving: a bar from the catalog or an example part.
enum Active {
    Bar(Box<DemoStudy>),
    Example(Box<ExampleStudy>),
}

impl Active {
    /// The solve behind this study, once there is one.
    fn study(&self) -> Option<&DemoStudy> {
        match self {
            Active::Bar(study) => Some(study),
            Active::Example(example) => example.study(),
        }
    }
}

/// The study the page is driving.
#[wasm_bindgen]
pub struct Demo {
    study: Option<Active>,
    last_refusal: Option<String>,
    /// The render arrays of the current estimate: one page refresh reads several of them, and
    /// building them once keeps that refresh inside its frame. Cleared by whatever can change them.
    view_cache: RefCell<Option<Rc<view::ViewData>>>,
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
            view_cache: RefCell::new(None),
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
        self.view_cache.replace(None);
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
                self.study = Some(Active::Bar(Box::new(study)));
                Ok(state.to_string())
            }
            Err(error @ PlanError::TooLarge { .. }) => {
                self.last_refusal = Some(error.to_string());
                Ok("too_large".to_string())
            }
            Err(error) => Err(JsValue::from_str(&error.to_string())),
        }
    }

    /// Starts an example part from the text of its surface file, cancelling any running study.
    /// `material` indexes the page's materials, or is negative for the material the benchmark
    /// used; `load_scale` is the factor on the benchmark's load. Starting the part already held
    /// keeps its mesh, so only the solve runs again. Returns the state name, which is
    /// `"failed"` with `failure` in [`Demo::summary_json`] when the part is refused; an unknown
    /// id or a material out of range returns an error naming the value.
    #[wasm_bindgen(js_name = startExample)]
    pub fn start_example(
        &mut self,
        id: &str,
        surface_text: &str,
        material: i32,
        load_scale: f64,
    ) -> Result<String, JsValue> {
        let part = examples::find(id)
            .ok_or_else(|| JsValue::from_str(&format!("unknown example part {id:?}")))?;
        let retained = match self.study.take() {
            Some(Active::Example(previous)) => previous.take_meshed(),
            _ => None,
        };
        self.last_refusal = None;
        self.view_cache.replace(None);
        let material = example_material(part, material)?;
        let study = ExampleStudy::start(part, surface_text, material, load_scale, retained);
        let state = example_state_name(study.state());
        self.study = Some(Active::Example(Box::new(study)));
        Ok(state.to_string())
    }

    /// Shows the example part at `factor` times the benchmark's load, without solving again.
    #[wasm_bindgen(js_name = setLoadScale)]
    pub fn set_load_scale(&mut self, factor: f64) {
        if let Some(Active::Example(example)) = self.study.as_mut() {
            example.set_load_scale(factor);
            self.view_cache.replace(None);
        }
    }

    /// Runs one slice with `budget` units of work (see `engine.rs`); returns the state name:
    /// `"meshing"`, `"refining"`, `"preparing"` (an example part only), `"assembling"`,
    /// `"solving"`, `"recovering"`, `"done"`, `"cancelled"`, `"failed"`, or `"idle"` with no
    /// study.
    pub fn step(&mut self, budget: u32) -> String {
        self.view_cache.replace(None);
        match self.study.as_mut() {
            None => "idle".to_string(),
            Some(Active::Bar(study)) => state_name(study.step(budget)).to_string(),
            Some(Active::Example(example)) => example_state_name(example.step(budget)).to_string(),
        }
    }

    /// Cancels the running study; a new `start` replaces it at any time anyway.
    pub fn cancel(&mut self) {
        match self.study.as_mut() {
            None => {}
            Some(Active::Bar(study)) => study.cancel(),
            Some(Active::Example(example)) => example.cancel(),
        }
    }

    /// The reason for the last `"too_large"` state, if any.
    pub fn refusal(&self) -> Option<String> {
        self.last_refusal.clone()
    }

    /// JSON text of the current estimate's numbers: state, iterations, residual, element and node
    /// counts, min/max stress, max displacement, max local rotation in radians (`maxRotation`,
    /// which caps the page's exaggeration), peak stress outside the clamps, safety factor
    /// (`null` before there is stress), `validity` (whether the result is inside what the linear
    /// model can describe: `withinModel`, `exceedsYield`, `excessiveDeflection`, `excessiveTwist`,
    /// and `deflectionLimit`, `twistRadians`, `twistLimitRadians`, each `null` where it does not
    /// apply; see `validity.rs`) and either `rows` (quantity, unit, place, fea, textbook,
    /// percentDifference) or `textbook: "none"` with `explanation` when the combination has no
    /// textbook answer. Failure gives `failure` with the message. While an example part is being
    /// meshed there is no solve yet: `state`, `meshDone`, `meshTotal` and `failure` only.
    #[wasm_bindgen(js_name = summaryJson)]
    pub fn summary_json(&self) -> String {
        let study = match self.study.as_ref() {
            None => return "{\"state\":\"idle\"}".to_string(),
            Some(active) => match active.study() {
                Some(study) => study,
                None => {
                    let Active::Example(example) = active else {
                        return "{\"state\":\"idle\"}".to_string();
                    };
                    let (done, total) = example.mesh_progress();
                    return format!(
                        "{{\"state\":{},\"meshDone\":{done},\"meshTotal\":{total},\"failure\":{}}}",
                        json_string(example_state_name(example.state())),
                        example.failure().map_or("null".to_string(), json_string),
                    );
                }
            },
        };
        let Some(view) = self.view() else {
            return "{\"state\":\"idle\"}".to_string();
        };
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
             \"minStress\":{},\"maxStress\":{},\"maxDisplacement\":{},\"maxRotation\":{},\
             \"peakStressOutsideClamps\":{},\
             \"safetyFactor\":{safety},\"validity\":{validity},\"failure\":{},{textbook}}}",
            json_string(state_name(study.state())),
            study.iterations(),
            json_number(study.residual()),
            json_number(view.min_stress),
            json_number(view.max_stress),
            json_number(view.max_displacement),
            json_number(view.max_rotation),
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
        let view = self.view();
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

    /// The local rotation vector in radians, three per vertex, at the current estimate.
    pub fn rotations(&self) -> Float32Array {
        self.array(|v| &v.rotation)
    }

    /// An example part's held node positions, three per node; empty for a bar or before its mesh
    /// is ready.
    #[wasm_bindgen(js_name = heldPoints)]
    pub fn held_points(&self) -> Float32Array {
        match self.study.as_ref() {
            Some(Active::Example(example)) => Float32Array::from(example.held_points()),
            _ => Float32Array::new_with_length(0),
        }
    }

    /// An example part's load arrows: tail, unit direction and length, seven numbers each.
    #[wasm_bindgen(js_name = loadArrows)]
    pub fn load_arrows(&self) -> Float32Array {
        match self.study.as_ref() {
            Some(Active::Example(example)) => Float32Array::from(example.load_arrows()),
            _ => Float32Array::new_with_length(0),
        }
    }

    /// The render arrays of the current estimate, built at most once between changes to it.
    fn view(&self) -> Option<Rc<view::ViewData>> {
        if let Some(view) = self.view_cache.borrow().as_ref() {
            return Some(Rc::clone(view));
        }
        let view = Rc::new(self.study.as_ref()?.study().map(DemoStudy::view)?);
        self.view_cache.replace(Some(Rc::clone(&view)));
        Some(view)
    }

    fn array(&self, pick: impl Fn(&view::ViewData) -> &Vec<f32>) -> Float32Array {
        let view = self.view();
        let empty = Vec::new();
        Float32Array::from(view.as_ref().map_or(empty.as_slice(), |v| pick(v)))
    }
}

fn example_material(
    part: &examples::ExamplePart,
    material: i32,
) -> Result<ExampleMaterial, JsValue> {
    if material < 0 {
        return Ok(ExampleMaterial {
            material: part.benchmark_material(),
            yield_strength: None,
        });
    }
    let chosen = MATERIALS
        .get(material as usize)
        .ok_or_else(|| JsValue::from_str(&format!("no material number {material}")))?;
    Ok(ExampleMaterial {
        material: solidfem::elastic::Material {
            youngs_modulus: chosen.youngs_modulus,
            poisson_ratio: chosen.poisson_ratio,
        },
        yield_strength: Some(chosen.yield_strength),
    })
}

fn example_state_name(state: ExampleState) -> &'static str {
    match state {
        ExampleState::Meshing => "meshing",
        ExampleState::Refining => "refining",
        ExampleState::Preparing => "preparing",
        ExampleState::Study(state) => state_name(state),
        ExampleState::Failed => "failed",
        ExampleState::Cancelled => "cancelled",
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
