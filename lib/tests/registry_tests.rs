use infer_lib::prelude::*;
use polars::prelude::*;
use std::sync::Arc;
use std::thread;

// Mock model for testing
struct MockModel {
    id: String,
    prediction: Vec<f64>,
}

impl MockModel {
    fn new(id: &str, prediction: Vec<f64>) -> Self {
        Self {
            id: id.to_string(),
            prediction,
        }
    }
}

impl Model for MockModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::Perpetual
    }

    fn predict(&self, _input: &ModelInput) -> Result<ModelOutput> {
        Ok(ModelOutput::Single(self.prediction.clone()))
    }

    fn num_features(&self) -> Result<usize> {
        Ok(10)
    }
}

#[test]
fn test_registry_basic_operations() {
    let registry = ModelRegistry::new();

    // Register a model
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0, 2.0]));
    assert!(registry.contains("model1"));
    assert_eq!(registry.len(), 1);

    // Get the model
    let model = registry.get("model1").unwrap();
    assert_eq!(model.id(), "model1");

    // List IDs
    let ids = registry.list_ids();
    assert_eq!(ids.len(), 1);
    assert!(ids.contains(&"model1".to_string()));

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

#[test]
fn test_registry_multiple_models() {
    let registry = ModelRegistry::new();

    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));
    registry.register("model2".to_string(), MockModel::new("model2", vec![2.0]));
    registry.register("model3".to_string(), MockModel::new("model3", vec![3.0]));

    assert_eq!(registry.len(), 3);
    assert!(registry.contains("model1"));
    assert!(registry.contains("model2"));
    assert!(registry.contains("model3"));

    let ids = registry.list_ids();
    assert_eq!(ids.len(), 3);
}

#[test]
fn test_registry_predict() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![42.0]));

    let df = df! {
        "a" => [1.0f32],
        "b" => [2.0f32],
    }
    .unwrap();

    let output = registry.predict("model1", &ModelInput(df)).unwrap();
    match output {
        ModelOutput::Single(predictions) => {
            assert_eq!(predictions, vec![42.0]);
        }
        _ => panic!("Expected Single output"),
    }
}

#[test]
fn test_registry_predict_many() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));
    registry.register("model2".to_string(), MockModel::new("model2", vec![2.0]));
    registry.register("model3".to_string(), MockModel::new("model3", vec![3.0]));

    let df1 = df! { "a" => [1.0f32] }.unwrap();
    let df2 = df! { "a" => [2.0f32] }.unwrap();
    let df3 = df! { "a" => [3.0f32] }.unwrap();

    let results = registry.predict_many(
        &["model1", "model2", "model3"],
        &[ModelInput(df1), ModelInput(df2), ModelInput(df3)],
    );

    assert_eq!(results.len(), 3);

    for (id, result) in results {
        assert!(result.is_ok());
        let output = result.unwrap();
        match (&id[..], output) {
            ("model1", ModelOutput::Single(pred)) => assert_eq!(pred, vec![1.0]),
            ("model2", ModelOutput::Single(pred)) => assert_eq!(pred, vec![2.0]),
            ("model3", ModelOutput::Single(pred)) => assert_eq!(pred, vec![3.0]),
            _ => panic!("Unexpected result"),
        }
    }
}

#[test]
fn test_registry_predict_broadcast() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));
    registry.register("model2".to_string(), MockModel::new("model2", vec![2.0]));
    registry.register("model3".to_string(), MockModel::new("model3", vec![3.0]));

    let df = df! {
        "a" => [1.0f32],
        "b" => [2.0f32],
    }
    .unwrap();

    let results = registry.predict_broadcast(&["model1", "model2", "model3"], &ModelInput(df));

    assert_eq!(results.len(), 3);

    for (id, result) in results {
        assert!(result.is_ok());
        let output = result.unwrap();
        match (&id[..], output) {
            ("model1", ModelOutput::Single(pred)) => assert_eq!(pred, vec![1.0]),
            ("model2", ModelOutput::Single(pred)) => assert_eq!(pred, vec![2.0]),
            ("model3", ModelOutput::Single(pred)) => assert_eq!(pred, vec![3.0]),
            _ => panic!("Unexpected result"),
        }
    }
}

#[test]
fn test_registry_clear() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));
    registry.register("model2".to_string(), MockModel::new("model2", vec![2.0]));

    assert_eq!(registry.len(), 2);

    registry.clear();
    assert_eq!(registry.len(), 0);
    assert!(!registry.contains("model1"));
    assert!(!registry.contains("model2"));
}

#[test]
fn test_registry_replace_model() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));

    let df = df! { "a" => [1.0f32] }.unwrap();

    // First prediction
    let output1 = registry.predict("model1", &ModelInput(df.clone())).unwrap();
    match output1 {
        ModelOutput::Single(pred) => assert_eq!(pred, vec![1.0]),
        _ => panic!("Expected Single output"),
    }

    // Replace model
    registry.register("model1".to_string(), MockModel::new("model1", vec![42.0]));

    // Second prediction should use new model
    let output2 = registry.predict("model1", &ModelInput(df)).unwrap();
    match output2 {
        ModelOutput::Single(pred) => assert_eq!(pred, vec![42.0]),
        _ => panic!("Expected Single output"),
    }
}

#[test]
fn test_registry_thread_safety() {
    let registry = Arc::new(ModelRegistry::new());
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));

    let mut handles = vec![];

    // Spawn multiple threads reading from registry
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let df = df! { "a" => [i as f32] }.unwrap();
            registry_clone.predict("model1", &ModelInput(df)).unwrap()
        });
        handles.push(handle);
    }

    // All threads should succeed
    for handle in handles {
        let result = handle.join().unwrap();
        match result {
            ModelOutput::Single(pred) => assert_eq!(pred, vec![1.0]),
            _ => panic!("Expected Single output"),
        }
    }
}

#[test]
fn test_registry_concurrent_updates() {
    let registry = Arc::new(ModelRegistry::new());

    let mut handles = vec![];

    // Spawn threads that register different models
    for i in 0..5 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let model_id = format!("model{}", i);
            registry_clone.register(
                model_id.clone(),
                MockModel::new(&model_id, vec![i as f64]),
            );
        });
        handles.push(handle);
    }

    // Wait for all registrations
    for handle in handles {
        handle.join().unwrap();
    }

    // All models should be registered
    assert_eq!(registry.len(), 5);
    for i in 0..5 {
        assert!(registry.contains(&format!("model{}", i)));
    }
}

#[test]
fn test_registry_debug() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));
    registry.register("model2".to_string(), MockModel::new("model2", vec![2.0]));

    let debug_str = format!("{:?}", registry);
    assert!(debug_str.contains("ModelRegistry"));
    assert!(debug_str.contains("num_models"));
}

#[test]
fn test_registry_predict_many_with_error() {
    let registry = ModelRegistry::new();
    registry.register("model1".to_string(), MockModel::new("model1", vec![1.0]));

    let df1 = df! { "a" => [1.0f32] }.unwrap();
    let df2 = df! { "a" => [2.0f32] }.unwrap();

    // model2 doesn't exist
    let results = registry.predict_many(&["model1", "model2"], &[ModelInput(df1), ModelInput(df2)]);

    assert_eq!(results.len(), 2);
    assert!(results[0].1.is_ok());
    assert!(results[1].1.is_err());
}
