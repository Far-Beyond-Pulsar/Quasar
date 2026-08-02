use std::collections::HashMap;
use std::sync::RwLock;

use quasar_core::bands::Band8;
use quasar_core::backend::MaterialProvider;
use quasar_core::error::SpatialAudioError;
use quasar_core::rays::RayInteractionContext;

use crate::evaluator::{AcousticResponse8Band, IAcousticMaterialEvaluator};
use crate::instance::{AcousticMaterialInstance, MaterialModelId, MaterialParameterBuffer};

/// Thread-safe registry of material evaluators and material instances.
pub struct AcousticMaterialRegistry {
    evaluators: RwLock<HashMap<MaterialModelId, Box<dyn IAcousticMaterialEvaluator>>>,
    instances: RwLock<Vec<AcousticMaterialInstance>>,
}

impl AcousticMaterialRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            evaluators: RwLock::new(HashMap::new()),
            instances: RwLock::new(Vec::new()),
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

    /// Add a material instance. Returns a handle (index into the instances array).
    pub fn add_instance(&self, instance: AcousticMaterialInstance) -> u32 {
        let mut instances = self.instances.write().expect("instances lock poisoned");
        let handle = instances.len() as u32;
        instances.push(instance);
        handle
    }

    /// Update a material instance's parameter buffer (hot-swappable).
    ///
    /// No acceleration structure rebuild is needed.
    pub fn update_instance(
        &self,
        handle: u32,
        params: MaterialParameterBuffer,
    ) -> Result<(), SpatialAudioError> {
        let mut instances = self.instances.write().expect("instances lock poisoned");
        let idx = handle as usize;
        if idx >= instances.len() {
            return Err(SpatialAudioError::Material(format!(
                "instance handle {} out of range (count {})",
                handle,
                instances.len()
            )));
        }
        instances[idx].parameters = params;
        Ok(())
    }

    /// Get a material instance by handle.
    pub fn get_instance(&self, handle: u32) -> Option<AcousticMaterialInstance> {
        let instances = self.instances.read().expect("instances lock poisoned");
        let idx = handle as usize;
        instances.get(idx).cloned()
    }

    /// Remove a material instance by handle (swaps with the last element).
    ///
    /// Returns `true` if a matching handle was removed. After removal the last
    /// element moves to the vacated slot; callers are responsible for tracking
    /// handle invalidations or re-fetching handles via `add_instance`.
    pub fn remove_instance(&self, handle: u32) -> bool {
        let mut instances = self.instances.write().expect("instances lock poisoned");
        let idx = handle as usize;
        if idx >= instances.len() {
            return false;
        }
        instances.swap_remove(idx);
        true
    }

    /// Number of material instances.
    pub fn instance_count(&self) -> usize {
        self.instances.read().expect("instances lock poisoned").len()
    }

    /// Evaluate a material instance. Called from the compute thread.
    pub fn evaluate(
        &self,
        handle: u32,
        context: &RayInteractionContext,
    ) -> Result<AcousticResponse8Band, SpatialAudioError> {
        let instances = self.instances.read().expect("instances lock poisoned");
        let idx = handle as usize;
        if idx >= instances.len() {
            return Err(SpatialAudioError::Material(format!(
                "instance handle {} out of range",
                handle
            )));
        }
        let instance = &instances[idx];
        let evaluators = self.evaluators.read().expect("evaluators lock poisoned");
        let evaluator = evaluators.get(&instance.model_id).ok_or_else(|| {
            SpatialAudioError::Material(format!(
                "no evaluator registered for model_id {:?}",
                instance.model_id
            ))
        })?;
        Ok(evaluator.evaluate(&instance.parameters, context))
    }

    /// Get the total byte size of all instance parameter buffers combined.
    ///
    /// Useful for allocating GPU storage buffers.
    pub fn total_parameter_bytes(&self) -> usize {
        let instances = self.instances.read().expect("instances lock poisoned");
        instances.iter().map(|inst| inst.parameters.len()).sum()
    }
}

impl MaterialProvider for AcousticMaterialRegistry {
    fn evaluate_material(&self, handle: u32, context: &RayInteractionContext) -> Band8 {
        self.evaluate(handle, context)
            .map(|r| r.absorption)
            .unwrap_or_else(|_| Band8::splat(0.9))
    }
}

impl Default for AcousticMaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}
