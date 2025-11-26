use infer_lib::prelude::*;
use infer_lib::ModelMetadata;
use polars::prelude::*;

#[test]
fn test_model_input_dataframe_creation() {
    let df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();

    let input = ModelInput(df);

    assert_eq!(input.0.height(), 2);
    assert_eq!(input.0.width(), 2);
}

#[test]
fn test_model_output_single() {
    let output = ModelOutput::Single(vec![1.0, 2.0, 3.0]);

    assert_eq!(output.num_rows(), 3);
    assert_eq!(output.as_slice(), Some(&[1.0, 2.0, 3.0][..]));

    let vec = output.into_vec();
    assert_eq!(vec, Some(vec![1.0, 2.0, 3.0]));
}

#[test]
fn test_model_output_multi() {
    let output = ModelOutput::Multi {
        data: vec![0.7, 0.2, 0.1, 0.3, 0.5, 0.2],
        num_classes: 3,
    };

    assert_eq!(output.num_rows(), 2);
    assert_eq!(output.as_slice().unwrap().len(), 6);

    let vec = output.into_vec();
    assert_eq!(vec.unwrap().len(), 6);
}

#[test]
fn test_model_backend_types() {
    let backends = vec![
        ModelBackend::CatBoost,
        ModelBackend::XGBoost,
        ModelBackend::LightGBM,
        ModelBackend::Perpetual,
    ];

    for backend in backends {
        // Test Display
        let display = format!("{}", backend);
        assert!(!display.is_empty());

        // Test Debug
        let debug = format!("{:?}", backend);
        assert!(!debug.is_empty());

        // Test Clone
        let cloned = backend;
        assert_eq!(backend, cloned);
    }
}

#[test]
fn test_model_metadata() {
    let metadata = ModelMetadata {
        name: "test_model".to_string(),
        version: Some("1.0.0".to_string()),
        description: Some("A test model".to_string()),
        created_at: Some("2024-01-01".to_string()),
        tags: vec!["test".to_string(), "demo".to_string()],
    };

    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.version.as_ref().unwrap(), "1.0.0");
    assert_eq!(metadata.tags.len(), 2);
}

#[test]
fn test_model_input_clone() {
    let df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();

    let input1 = ModelInput(df.clone());
    let input2 = input1.clone();

    assert_eq!(input1.0.height(), input2.0.height());
    assert_eq!(input1.0.width(), input2.0.width());
}

#[test]
fn test_model_output_clone() {
    let output1 = ModelOutput::Single(vec![1.0, 2.0, 3.0]);
    let output2 = output1.clone();

    assert_eq!(output1.as_slice(), output2.as_slice());
}

#[test]
fn test_empty_model_output() {
    let output = ModelOutput::Single(vec![]);
    assert_eq!(output.num_rows(), 0);
    assert_eq!(output.as_slice().unwrap().len(), 0);
}

#[test]
fn test_large_model_output() {
    let large_data: Vec<f64> = (0..10000).map(|x| x as f64).collect();
    let output = ModelOutput::Single(large_data.clone());

    assert_eq!(output.num_rows(), 10000);
    assert_eq!(output.as_slice().unwrap(), large_data.as_slice());
}

#[test]
fn test_multi_output_dimensions() {
    let output = ModelOutput::Multi {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        num_classes: 3,
    };

    // 9 values / 3 classes = 3 rows
    assert_eq!(output.num_rows(), 3);
}

#[test]
fn test_model_output_as_slice_vs_into_vec() {
    let data = vec![1.0, 2.0, 3.0];
    let output = ModelOutput::Single(data.clone());

    // as_slice doesn't consume
    let slice = output.as_slice();
    assert_eq!(slice, Some(data.as_slice()));

    // Can still use output
    let vec = output.into_vec();
    assert_eq!(vec, Some(data));
}

#[test]
fn test_dataframe_input_edge_cases() {
    // Single row, single feature
    let df1 = df! {
        "a" => [42.0f32],
    }
    .unwrap();
    let input1 = ModelInput(df1);
    assert_eq!(input1.0.height(), 1);
    assert_eq!(input1.0.width(), 1);

    // Many features, one row
    let df2 = df! {
        "a" => [1.0f32],
        "b" => [2.0f32],
        "c" => [3.0f32],
        "d" => [4.0f32],
        "e" => [5.0f32],
    }
    .unwrap();
    let input2 = ModelInput(df2);
    assert_eq!(input2.0.height(), 1);
    assert_eq!(input2.0.width(), 5);

    // Many rows, one feature
    let df3 = df! {
        "a" => [1.0f32, 2.0, 3.0, 4.0, 5.0],
    }
    .unwrap();
    let input3 = ModelInput(df3);
    assert_eq!(input3.0.height(), 5);
    assert_eq!(input3.0.width(), 1);
}

#[test]
fn test_multi_output_single_class() {
    // Edge case: multi-output with just 1 class
    let output = ModelOutput::Multi {
        data: vec![0.8, 0.9, 0.7],
        num_classes: 1,
    };

    assert_eq!(output.num_rows(), 3);
}

#[test]
fn test_model_backend_copy_trait() {
    let backend1 = ModelBackend::XGBoost;
    let backend2 = backend1; // Copy, not move

    // Both can still be used
    assert_eq!(backend1, ModelBackend::XGBoost);
    assert_eq!(backend2, ModelBackend::XGBoost);
}

#[test]
fn test_metadata_optional_fields() {
    let minimal_metadata = ModelMetadata {
        name: "minimal".to_string(),
        version: None,
        description: None,
        created_at: None,
        tags: vec![],
    };

    assert_eq!(minimal_metadata.name, "minimal");
    assert!(minimal_metadata.version.is_none());
    assert!(minimal_metadata.description.is_none());
    assert!(minimal_metadata.created_at.is_none());
    assert!(minimal_metadata.tags.is_empty());
}

#[test]
fn test_inference_result() {
    let output = ModelOutput::Single(vec![1.0, 2.0]);
    let metadata = InferenceMetadata {
        model_id: "test".to_string(),
        backend: ModelBackend::Perpetual,
        custom: std::collections::HashMap::new(),
    };

    let result = InferenceResult {
        output: output.clone(),
        metadata: Some(metadata.clone()),
    };

    match result.output {
        ModelOutput::Single(ref preds) => assert_eq!(preds, &vec![1.0, 2.0]),
        _ => panic!("Expected Single output"),
    }

    assert!(result.metadata.is_some());
    assert_eq!(result.metadata.unwrap().model_id, "test");
}

#[test]
fn test_inference_metadata_custom_fields() {
    let mut custom = std::collections::HashMap::new();
    custom.insert("key1".to_string(), "value1".to_string());
    custom.insert("key2".to_string(), "value2".to_string());

    let metadata = InferenceMetadata {
        model_id: "test".to_string(),
        backend: ModelBackend::XGBoost,
        custom: custom.clone(),
    };

    assert_eq!(metadata.custom.len(), 2);
    assert_eq!(metadata.custom.get("key1").unwrap(), "value1");
}
