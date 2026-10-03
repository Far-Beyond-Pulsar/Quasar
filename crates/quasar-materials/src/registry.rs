use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use quasar_core::bands::Band8;
use quasar_core::backend::MaterialProvider;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::RayInteractionContext;

use crate::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use crate::instance::{AcousticMaterialInstance, MaterialModelId, MaterialParameterBuffer};

/// Bits of an instance handle holding the slot index; the remaining high bits hold the
/// slot generation (so a handle of a removed instance never aliases a later one).
const INDEX_BITS: u32 = 20;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const GEN_MASK: u32 = (1 << (32 - INDEX_BITS)) - 1;
/// At most this many distinct errors are remembered for "report once" (bounds memory).
const MAX_REPORTED: usize = 256;

/// Kinds of runtime lookup failures (the "report once" key is `(handle, kind)`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ErrKind {
    BadHandle,
    NoEvaluator,
    BadParams,
}

struct Slot {
    generation: u32,
    instance: Option<AcousticMaterialInstance>,
    /// Parameters were validated by the evaluator (set when the evaluator was known at
    /// add/update time, else lazily on first evaluation).
    validated: AtomicBool,
}

struct Instances {
    slots: Vec<Slot>,
    /// Indices of free (removed) slots, reused with a bumped generation.
    free: Vec<u32>,
    live: usize,
}

/// Thread-safe registry of material evaluators and material instances.
///
/// # Handles
/// A handle is `generation << 20 | slot index`. Fresh handles are the slot index (generation 0)
/// so the first instances are `0, 1, 2, ...`. Removing an instance frees its slot and bumps
/// the slot generation; every other handle keeps its meaning, and the removed handle (and any
/// copy of it) is invalid even after the slot is reused (the generation wraps after 4096
/// reuses of one slot; at most 2^20 simultaneous instances).
///
/// # Errors, defaults
/// Parameter buffers are validated by the model's evaluator at [`Self::try_add_instance`] /
/// [`Self::update_instance`]. The [`MaterialProvider`] path (ray tracing) cannot fail: an invalid
/// handle, a missing evaluator or a buffer that cannot be evaluated uses the explicit **default
/// response** (see [`Self::set_default_response`]; default absorption 0.9, opaque), counts in
/// [`Self::error_count`] and is printed once per `(handle, kind)`.
pub struct AcousticMaterialRegistry {
    evaluators: RwLock<HashMap<MaterialModelId, Box<dyn IAcousticMaterialEvaluator>>>,
    instances: RwLock<Instances>,
    default_response: RwLock<AcousticResponse8Band>,
    errors: AtomicU64,
    reported: Mutex<HashSet<(u32, ErrKind)>>,
}

impl AcousticMaterialRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            evaluators: RwLock::new(HashMap::new()),
            instances: RwLock::new(Instances { slots: Vec::new(), free: Vec::new(), live: 0 }),
            default_response: RwLock::new(Self::builtin_default_response()),
            errors: AtomicU64::new(0),
            reported: Mutex::new(HashSet::new()),
        }
    }

    /// The default used until [`Self::set_default_response`]: absorption 0.9, no scattering,
    /// opaque (the historic fallback, now explicit and observable).
    pub fn builtin_default_response() -> AcousticResponse8Band {
        AcousticResponse8Band {
            absorption: Band8::splat(0.9),
            scattering: Band8::zeros(),
            transmission: Band8::zeros(),
        }
    }

    /// Replace the response used for invalid handles / missing evaluators / unusable buffers.
    pub fn set_default_response(&self, response: AcousticResponse8Band) {
        *self.default_response.write().expect("default lock poisoned") = response;
    }

    /// The response currently used for failed lookups.
    pub fn default_response(&self) -> AcousticResponse8Band {
        self.default_response.read().expect("default lock poisoned").clone()
    }

    /// Total number of failed lookups (bad/stale handle, missing evaluator, unusable
    /// parameters) seen by [`Self::evaluate`] and the [`MaterialProvider`] methods.
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    fn report(&self, handle: u32, kind: ErrKind, msg: &str) {
        self.errors.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut seen) = self.reported.lock() {
            if seen.len() < MAX_REPORTED && seen.insert((handle, kind)) {
                eprintln!("quasar-materials: {msg} (handle {handle}); using the default material (reported once)");
            }
        }
    }

    /// Register a material evaluator (model). Called at engine startup.
    pub fn register_evaluator(&self, evaluator: Box<dyn IAcousticMaterialEvaluator>) {
        let id = evaluator.model_id();
        self.evaluators.write().expect("evaluators lock poisoned").insert(id, evaluator);
    }

    /// Unregister a material evaluator by model ID. Returns `true` if one was removed.
    pub fn unregister_evaluator(&self, model_id: MaterialModelId) -> bool {
        self.evaluators.write().expect("evaluators lock poisoned").remove(&model_id).is_some()
    }

    /// Check if an evaluator is registered for the given model ID.
    pub fn has_evaluator(&self, model_id: MaterialModelId) -> bool {
        self.evaluators.read().expect("evaluators lock poisoned").contains_key(&model_id)
    }

    /// Number of registered evaluators.
    pub fn evaluator_count(&self) -> usize {
        self.evaluators.read().expect("evaluators lock poisoned").len()
    }

    /// `Some(result)` when the model's evaluator is registered, `None` when it is not (the
    /// buffer can then only be validated at evaluation time).
    fn validate(&self, model: MaterialModelId, params: &MaterialParameterBuffer) -> Option<Result<(), String>> {
        let evaluators = self.evaluators.read().expect("evaluators lock poisoned");
        evaluators.get(&model).map(|e| e.validate(params))
    }

    /// Add a material instance. Returns a stable handle (see the type docs).
    ///
    /// Fails if the model's evaluator is registered and rejects the parameter buffer. When the
    /// evaluator is not registered yet the buffer is validated on first use instead.
    pub fn try_add_instance(&self, instance: AcousticMaterialInstance) -> Result<u32, SpatialAudioError> {
        let checked = self.validate(instance.model_id, &instance.parameters);
        if let Some(Err(e)) = &checked {
            return Err(SpatialAudioError::Material(format!(
                "invalid parameters for model {:?}: {e}",
                instance.model_id
            )));
        }
        let validated = matches!(checked, Some(Ok(())));
        let mut inst = self.instances.write().expect("instances lock poisoned");
        let handle = if let Some(idx) = inst.free.pop() {
            let slot = &mut inst.slots[idx as usize];
            slot.instance = Some(instance);
            slot.validated = AtomicBool::new(validated);
            (slot.generation << INDEX_BITS) | idx
        } else {
            let idx = inst.slots.len() as u32;
            if idx > INDEX_MASK {
                return Err(SpatialAudioError::Material("too many material instances (max 2^20)".into()));
            }
            inst.slots.push(Slot { generation: 0, instance: Some(instance), validated: AtomicBool::new(validated) });
            idx
        };
        inst.live += 1;
        Ok(handle)
    }

    /// Add a material instance and return its handle.
    ///
    /// # Panics
    /// If the model's evaluator rejects the parameter buffer (a programming error; use
    /// [`Self::try_add_instance`] to handle it).
    pub fn add_instance(&self, instance: AcousticMaterialInstance) -> u32 {
        match self.try_add_instance(instance) {
            Ok(h) => h,
            Err(e) => panic!("AcousticMaterialRegistry::add_instance: {e}"),
        }
    }

    fn slot_index(inst: &Instances, handle: u32) -> Option<usize> {
        let idx = (handle & INDEX_MASK) as usize;
        let generation = (handle >> INDEX_BITS) & GEN_MASK;
        let slot = inst.slots.get(idx)?;
        (slot.generation == generation && slot.instance.is_some()).then_some(idx)
    }

    /// Update a material instance's parameter buffer (hot-swappable).
    ///
    /// No acceleration structure rebuild is needed. A buffer the evaluator rejects is an error
    /// and the previous parameters are kept.
    pub fn update_instance(
        &self,
        handle: u32,
        params: MaterialParameterBuffer,
    ) -> Result<(), SpatialAudioError> {
        let mut inst = self.instances.write().expect("instances lock poisoned");
        let Some(idx) = Self::slot_index(&inst, handle) else {
            return Err(SpatialAudioError::Material(format!(
                "instance handle {handle} is invalid (out of range or removed; {} live)",
                inst.live
            )));
        };
        let model = inst.slots[idx].instance.as_ref().map(|i| i.model_id).unwrap_or(MaterialModelId(0));
        let checked = self.validate(model, &params);
        if let Some(Err(e)) = &checked {
            return Err(SpatialAudioError::Material(format!("invalid parameters for model {model:?}: {e}")));
        }
        let slot = &mut inst.slots[idx];
        if let Some(i) = slot.instance.as_mut() {
            i.parameters = params;
        }
        slot.validated = AtomicBool::new(matches!(checked, Some(Ok(()))));
        Ok(())
    }

    /// Get a material instance by handle (`None` for an invalid or removed handle).
    pub fn get_instance(&self, handle: u32) -> Option<AcousticMaterialInstance> {
        let inst = self.instances.read().expect("instances lock poisoned");
        let idx = Self::slot_index(&inst, handle)?;
        inst.slots[idx].instance.clone()
    }

    /// Remove a material instance by handle.
    ///
    /// Returns `true` if the handle was live. No other handle changes meaning; the removed
    /// handle is invalid from now on (even once its slot is reused).
    pub fn remove_instance(&self, handle: u32) -> bool {
        let mut inst = self.instances.write().expect("instances lock poisoned");
        let Some(idx) = Self::slot_index(&inst, handle) else {
            return false;
        };
        let slot = &mut inst.slots[idx];
        slot.instance = None;
        slot.generation = (slot.generation + 1) & GEN_MASK;
        inst.free.push(idx as u32);
        inst.live -= 1;
        true
    }

    /// Number of live material instances.
    pub fn instance_count(&self) -> usize {
        self.instances.read().expect("instances lock poisoned").live
    }

    /// Evaluate a material instance. Called from the compute thread.
    ///
    /// Errors (invalid handle, missing evaluator, unusable parameters) are counted and reported
    /// once; use the [`MaterialProvider`] methods for the never-failing, default-material path.
    pub fn evaluate(
        &self,
        handle: u32,
        context: &RayInteractionContext,
    ) -> Result<AcousticResponse8Band, SpatialAudioError> {
        let inst = self.instances.read().expect("instances lock poisoned");
        let Some(idx) = Self::slot_index(&inst, handle) else {
            self.report(handle, ErrKind::BadHandle, "invalid material instance handle (out of range or removed)");
            return Err(SpatialAudioError::Material(format!("instance handle {handle} is invalid")));
        };
        let slot = &inst.slots[idx];
        let Some(instance) = slot.instance.as_ref() else {
            return Err(SpatialAudioError::Material(format!("instance handle {handle} is invalid")));
        };
        let evaluators = self.evaluators.read().expect("evaluators lock poisoned");
        let Some(evaluator) = evaluators.get(&instance.model_id) else {
            self.report(handle, ErrKind::NoEvaluator, "no evaluator registered for the material model");
            return Err(SpatialAudioError::Material(format!(
                "no evaluator registered for model_id {:?}",
                instance.model_id
            )));
        };
        if !slot.validated.load(Ordering::Relaxed) {
            if let Err(e) = evaluator.validate(&instance.parameters) {
                self.report(handle, ErrKind::BadParams, "material parameters are invalid");
                return Err(SpatialAudioError::Material(format!(
                    "invalid parameters for model {:?}: {e}",
                    instance.model_id
                )));
            }
            slot.validated.store(true, Ordering::Relaxed);
        }
        Ok(evaluator.evaluate(&instance.parameters, context))
    }

    /// Get the total byte size of all instance parameter buffers combined.
    ///
    /// Useful for allocating GPU storage buffers.
    pub fn total_parameter_bytes(&self) -> usize {
        let inst = self.instances.read().expect("instances lock poisoned");
        inst.slots.iter().filter_map(|s| s.instance.as_ref()).map(|i| i.parameters.len()).sum()
    }
}

impl MaterialProvider for AcousticMaterialRegistry {
    fn evaluate_material(&self, handle: u32, context: &RayInteractionContext) -> Band8 {
        match self.evaluate(handle, context) {
            Ok(r) => r.absorption,
            Err(_) => self.default_response().absorption,
        }
    }

    fn evaluate_transmission(&self, handle: u32, context: &RayInteractionContext) -> Band8 {
        match self.evaluate(handle, context) {
            Ok(r) => {
                let mut t = r.transmission;
                for v in t.0.iter_mut() {
                    *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                }
                t
            }
            // Unknown / unusable material: the configured default's transmission (opaque by default).
            Err(_) => self.default_response().transmission,
        }
    }
}

impl Default for AcousticMaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}
