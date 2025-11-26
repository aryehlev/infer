use crate::error::{InferError, Result};
use crate::model::{DynModel, Model, ModelInput, ModelOutput};
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::Arc;

/// Thread-safe model registry using ArcSwap for lock-free reads
///
/// This registry allows you to:
/// - Add/remove models dynamically without blocking inference
/// - Run inference on multiple models in parallel
/// - Swap out models atomically without downtime
#[derive(Default)]
pub struct ModelRegistry {
    models: ArcSwap<HashMap<String, DynModel>>,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: ArcSwap::from_pointee(HashMap::new()),
        }
    }

    /// Register a model with the given ID
    ///
    /// If a model with the same ID already exists, it will be replaced atomically.
    pub fn register<M: Model + 'static>(&self, id: String, model: M) {
        let model = Arc::new(model) as DynModel;
        self.register_dyn(id, model);
    }

    /// Register a type-erased model
    pub fn register_dyn(&self, id: String, model: DynModel) {
        self.models.rcu(|models| {
            let mut new_models = HashMap::clone(models);
            new_models.insert(id.clone(), model.clone());
            new_models
        });
    }

    /// Remove a model from the registry
    ///
    /// Returns true if the model was found and removed, false otherwise.
    pub fn unregister(&self, id: &str) -> bool {
        let mut found = false;
        self.models.rcu(|models| {
            let mut new_models = HashMap::clone(models);
            found = new_models.remove(id).is_some();
            new_models
        });
        found
    }

    /// Get a model by ID (returns a clone of the Arc for thread-safe usage)
    pub fn get(&self, id: &str) -> Result<DynModel> {
        let models = self.models.load();
        models
            .get(id)
            .cloned()
            .ok_or_else(|| InferError::ModelNotFound { id: id.to_string() })
    }

    /// Check if a model exists
    pub fn contains(&self, id: &str) -> bool {
        let models = self.models.load();
        models.contains_key(id)
    }

    /// Get the number of registered models
    pub fn len(&self) -> usize {
        let models = self.models.load();
        models.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a list of all model IDs
    pub fn list_ids(&self) -> Vec<String> {
        let models = self.models.load();
        models.keys().cloned().collect()
    }

    /// Run inference on a specific model
    pub fn predict(&self, model_id: &str, input: &ModelInput) -> Result<ModelOutput> {
        let model = self.get(model_id)?;
        model.predict(input)
    }

    /// Run inference on multiple models in parallel
    ///
    /// Returns a Vec of (model_id, Result<ModelOutput>) tuples
    pub fn predict_many(
        &self,
        model_ids: &[&str],
        inputs: &[ModelInput],
    ) -> Vec<(String, Result<ModelOutput>)> {
        use rayon::prelude::*;

        model_ids
            .par_iter()
            .zip(inputs.par_iter())
            .map(|(id, input)| {
                let result = self.get(id).and_then(|model| model.predict(input));
                (id.to_string(), result)
            })
            .collect()
    }

    /// Run the same input on multiple models in parallel
    ///
    /// Useful for model ensembling or A/B testing
    pub fn predict_broadcast(
        &self,
        model_ids: &[&str],
        input: &ModelInput,
    ) -> Vec<(String, Result<ModelOutput>)> {
        use rayon::prelude::*;

        model_ids
            .par_iter()
            .map(|id| {
                let result = self.get(id).and_then(|model| model.predict(input));
                (id.to_string(), result)
            })
            .collect()
    }

    /// Clear all models from the registry
    pub fn clear(&self) {
        self.models.store(Arc::new(HashMap::new()));
    }
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let models = self.models.load();
        f.debug_struct("ModelRegistry")
            .field("num_models", &models.len())
            .field("model_ids", &models.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock model for testing
    struct MockModel {
        id: String,
    }

    impl Model for MockModel {
        fn id(&self) -> &str {
            &self.id
        }

        fn backend(&self) -> crate::model::ModelBackend {
            crate::model::ModelBackend::CatBoost
        }

        fn predict(&self, _input: &ModelInput) -> Result<ModelOutput> {
            Ok(ModelOutput::Single(vec![1.0, 2.0, 3.0]))
        }

        fn num_features(&self) -> Result<usize> {
            Ok(10)
        }
    }

    #[test]
    fn test_registry_basic_operations() {
        let registry = ModelRegistry::new();

        // Register a model
        registry.register(
            "model1".to_string(),
            MockModel {
                id: "model1".to_string(),
            },
        );
        assert!(registry.contains("model1"));
        assert_eq!(registry.len(), 1);

        // Get the model
        let model = registry.get("model1").unwrap();
        assert_eq!(model.id(), "model1");

        // Unregister
        assert!(registry.unregister("model1"));
        assert!(!registry.contains("model1"));
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_not_found() {
        let registry = ModelRegistry::new();
        let result = registry.get("nonexistent");
        assert!(matches!(result, Err(InferError::ModelNotFound { .. })));
    }
}
