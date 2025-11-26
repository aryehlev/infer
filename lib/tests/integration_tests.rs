use infer_lib::prelude::*;
use polars::prelude::*;
use std::sync::Arc;

// Mock transformer for testing
struct NormalizeTransformer;

impl DataFrameTransformer for NormalizeTransformer {
    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // Simple normalization: divide by 10
        let transformed = df
            .lazy()
            .with_columns(vec![col("*").cast(DataType::Float64) / lit(10.0)])
            .collect()?;
        Ok(transformed)
    }

    fn name(&self) -> &str {
        "Normalize"
    }
}

// Mock model for integration testing
struct MockPredictorModel {
    id: String,
    transformers: Vec<DynDataFrameTransformer>,
}

impl MockPredictorModel {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            transformers: Vec::new(),
        }
    }

    fn with_transformer(mut self, transformer: DynDataFrameTransformer) -> Self {
        self.transformers.push(transformer);
        self
    }
}

impl Model for MockPredictorModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::Perpetual
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        // Apply transformers
        let mut df = input.0.clone();
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Convert DataFrame to dense for calculation
        use infer_lib::polars_ext::dataframe_to_dense_f32;
        let (data, num_rows, _) = dataframe_to_dense_f32(&df)?;

        // Simple prediction: sum of all values per row
        let predictions: Vec<f64> = data
            .chunks(data.len() / num_rows)
            .map(|row| row.iter().map(|&x| x as f64).sum())
            .collect();

        Ok(ModelOutput::Single(predictions))
    }

    fn num_features(&self) -> Result<usize> {
        Ok(2)
    }
}

#[test]
fn test_end_to_end_dataframe_workflow() {
    // 1. Create DataFrame
    let df = df! {
        "feature1" => [10.0f32, 20.0, 30.0],
        "feature2" => [40.0f32, 50.0, 60.0],
    }
    .unwrap();

    // 2. Create model with transformer
    let model =
        MockPredictorModel::new("test_model").with_transformer(Arc::new(NormalizeTransformer));

    // 3. Run prediction
    let output = model.predict(&ModelInput(df)).unwrap();

    // 4. Verify: After normalization (divide by 10):
    //    Row 0: (10/10) + (40/10) = 1 + 4 = 5
    //    Row 1: (20/10) + (50/10) = 2 + 5 = 7
    //    Row 2: (30/10) + (60/10) = 3 + 6 = 9
    match output {
        ModelOutput::Single(predictions) => {
            assert_eq!(predictions.len(), 3);
            assert!((predictions[0] - 5.0).abs() < 0.001);
            assert!((predictions[1] - 7.0).abs() < 0.001);
            assert!((predictions[2] - 9.0).abs() < 0.001);
        }
        _ => panic!("Expected Single output"),
    }
}

#[test]
fn test_registry_with_transformers() {
    let registry = ModelRegistry::new();

    // Register model with transformer
    let model = MockPredictorModel::new("model1").with_transformer(Arc::new(NormalizeTransformer));
    registry.register("model1".to_string(), model);

    // Create test data
    let df = df! {
        "a" => [10.0f32, 20.0],
        "b" => [30.0f32, 40.0],
    }
    .unwrap();

    // Predict through registry
    let output = registry.predict("model1", &ModelInput(df)).unwrap();

    match output {
        ModelOutput::Single(predictions) => {
            assert_eq!(predictions.len(), 2);
            // (10+30)/10 = 4, (20+40)/10 = 6
            assert!((predictions[0] - 4.0).abs() < 0.001);
            assert!((predictions[1] - 6.0).abs() < 0.001);
        }
        _ => panic!("Expected Single output"),
    }
}

#[test]
fn test_multiple_models_parallel() {
    let registry = ModelRegistry::new();

    // Register multiple models
    registry.register("model1".to_string(), MockPredictorModel::new("model1"));
    registry.register("model2".to_string(), MockPredictorModel::new("model2"));
    registry.register("model3".to_string(), MockPredictorModel::new("model3"));

    // Create DataFrames
    let df1 = df! {
        "a" => [1.0f32],
        "b" => [2.0f32],
    }
    .unwrap();

    let df2 = df! {
        "a" => [3.0f32],
        "b" => [4.0f32],
    }
    .unwrap();

    let df3 = df! {
        "a" => [5.0f32],
        "b" => [6.0f32],
    }
    .unwrap();

    // Run parallel predictions
    let results = registry.predict_many(
        &["model1", "model2", "model3"],
        &[ModelInput(df1), ModelInput(df2), ModelInput(df3)],
    );

    assert_eq!(results.len(), 3);

    // Verify all succeeded
    for (_, result) in &results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_broadcast_same_data_multiple_models() {
    let registry = ModelRegistry::new();

    // Register models with different transformers
    let model1 = MockPredictorModel::new("model1");
    let model2 = MockPredictorModel::new("model2").with_transformer(Arc::new(NormalizeTransformer));

    registry.register("model1".to_string(), model1);
    registry.register("model2".to_string(), model2);

    let df = df! {
        "a" => [10.0f32],
        "b" => [20.0f32],
    }
    .unwrap();

    let results = registry.predict_broadcast(&["model1", "model2"], &ModelInput(df));

    assert_eq!(results.len(), 2);

    // Model1: no transformation, 10 + 20 = 30
    match &results[0].1 {
        Ok(ModelOutput::Single(pred)) => {
            assert!((pred[0] - 30.0).abs() < 0.001);
        }
        _ => panic!("Expected successful prediction from model1"),
    }

    // Model2: normalized, (10+20)/10 = 3
    match &results[1].1 {
        Ok(ModelOutput::Single(pred)) => {
            assert!((pred[0] - 3.0).abs() < 0.001);
        }
        _ => panic!("Expected successful prediction from model2"),
    }
}

#[test]
fn test_dense_input_workflow() {
    let model = MockPredictorModel::new("test");

    // Create DataFrame input
    let df = df! {
        "a" => [1.0f32, 3.0],
        "b" => [2.0f32, 4.0],
    }
    .unwrap();

    let output = model.predict(&ModelInput(df)).unwrap();

    match output {
        ModelOutput::Single(predictions) => {
            assert_eq!(predictions.len(), 2);
            assert_eq!(predictions[0], 3.0); // 1 + 2
            assert_eq!(predictions[1], 7.0); // 3 + 4
        }
        _ => panic!("Expected Single output"),
    }
}

#[test]
fn test_output_to_series_integration() {
    let model = MockPredictorModel::new("test");

    let df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();

    let output = model.predict(&ModelInput(df)).unwrap();
    let series = infer_lib::output_to_series(&output, "predictions").unwrap();

    assert_eq!(series.len(), 2);
    assert_eq!(series.name(), "predictions");

    // Can add back to DataFrame
    let mut base_df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();
    let result_df = base_df.with_column(series).unwrap();

    assert_eq!(result_df.width(), 3);
    assert!(result_df.column("predictions").is_ok());
}

#[test]
fn test_model_hot_swap() {
    let registry = ModelRegistry::new();

    // Register initial model
    registry.register("prod_model".to_string(), MockPredictorModel::new("v1"));

    // Create DataFrame input for first prediction
    let df1 = df! {
        "a" => [5.0f32],
        "b" => [5.0f32],
    }
    .unwrap();

    // First prediction
    let output1 = registry.predict("prod_model", &ModelInput(df1)).unwrap();

    // Hot-swap model (with transformer this time)
    let new_model = MockPredictorModel::new("v2").with_transformer(Arc::new(NormalizeTransformer));
    registry.register("prod_model".to_string(), new_model);

    // Create DataFrame input for second prediction
    let df2 = df! {
        "a" => [5.0f32],
        "b" => [5.0f32],
    }
    .unwrap();

    // Second prediction uses new model
    let output2 = registry.predict("prod_model", &ModelInput(df2)).unwrap();

    // Results should be different
    match (output1, output2) {
        (ModelOutput::Single(pred1), ModelOutput::Single(pred2)) => {
            assert_eq!(pred1[0], 10.0); // 5 + 5
            assert_eq!(pred2[0], 1.0); // (5 + 5) / 10
        }
        _ => panic!("Expected Single outputs"),
    }
}

#[test]
fn test_chain_multiple_transformers() {
    struct AddTenTransformer;
    impl DataFrameTransformer for AddTenTransformer {
        fn transform(&self, df: DataFrame) -> Result<DataFrame> {
            let transformed = df
                .lazy()
                .with_columns(vec![col("*").cast(DataType::Float64) + lit(10.0)])
                .collect()?;
            Ok(transformed)
        }
        fn name(&self) -> &str {
            "AddTen"
        }
    }

    let model = MockPredictorModel::new("test")
        .with_transformer(Arc::new(AddTenTransformer))
        .with_transformer(Arc::new(NormalizeTransformer));

    let df = df! {
        "a" => [0.0f32],
        "b" => [0.0f32],
    }
    .unwrap();

    let output = model.predict(&ModelInput(df)).unwrap();

    match output {
        ModelOutput::Single(predictions) => {
            // (0+10) + (0+10) = 20, then /10 = 2
            assert!((predictions[0] - 2.0).abs() < 0.001);
        }
        _ => panic!("Expected Single output"),
    }
}
