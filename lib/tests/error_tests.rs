use infer_lib::prelude::*;
use polars::prelude::*;

#[test]
fn test_model_not_found_error() {
    let registry = ModelRegistry::new();
    let result = registry.get("nonexistent");

    assert!(result.is_err());
    match result {
        Err(InferError::ModelNotFound { id }) => {
            assert_eq!(id, "nonexistent");
        }
        _ => panic!("Expected ModelNotFound error"),
    }
}

#[test]
fn test_empty_dataframe_error() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;
    let df = DataFrame::empty();
    let result = dataframe_to_dense_f32(&df);

    assert!(result.is_err());
    match result {
        Err(InferError::ConversionError(msg)) => {
            assert!(msg.contains("zero"));
        }
        _ => panic!("Expected ConversionError"),
    }
}

#[test]
fn test_invalid_column_selection() {
    let df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();

    // Test selecting non-existent column
    let result = df.select(vec!["a", "nonexistent"]);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_dtype_error() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;
    // Create DataFrame with string type (unsupported)
    let df = df! {
        "strings" => ["a", "b", "c"],
    }
    .unwrap();

    let result = dataframe_to_dense_f32(&df);
    assert!(result.is_err());
    match result {
        Err(InferError::ConversionError(msg)) => {
            assert!(msg.contains("Unsupported"));
        }
        _ => panic!("Expected ConversionError for unsupported type"),
    }
}

#[test]
fn test_error_display() {
    let err = InferError::ModelNotFound {
        id: "test_model".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("Model not found"));
    assert!(display.contains("test_model"));

    let err2 = InferError::InvalidShape {
        expected: "10".to_string(),
        actual: "5".to_string(),
    };
    let display2 = format!("{}", err2);
    assert!(display2.contains("Invalid input shape"));
    assert!(display2.contains("10"));
    assert!(display2.contains("5"));
}

#[test]
fn test_error_from_polars() {
    let df = df! {
        "a" => [1.0f32],
    }
    .unwrap();

    // Try to select non-existent column
    let result: Result<DataFrame> = df
        .select(vec!["nonexistent".to_string()])
        .map_err(Into::into);
    assert!(result.is_err());
    match result {
        Err(InferError::PolarsError(_)) => {}
        _ => panic!("Expected PolarsError"),
    }
}

#[test]
fn test_conversion_error() {
    let err = InferError::ConversionError("test conversion failed".to_string());
    let display = format!("{}", err);
    assert!(display.contains("Conversion error"));
    assert!(display.contains("test conversion failed"));
}

#[test]
fn test_other_error() {
    let err = InferError::Other("something went wrong".to_string());
    let display = format!("{}", err);
    assert!(display.contains("Other error"));
    assert!(display.contains("something went wrong"));
}

#[test]
fn test_backend_enum_display() {
    assert_eq!(format!("{}", ModelBackend::CatBoost), "CatBoost");
    assert_eq!(format!("{}", ModelBackend::XGBoost), "XGBoost");
    assert_eq!(format!("{}", ModelBackend::LightGBM), "LightGBM");
    assert_eq!(format!("{}", ModelBackend::Perpetual), "Perpetual");
}

#[test]
fn test_backend_enum_equality() {
    assert_eq!(ModelBackend::CatBoost, ModelBackend::CatBoost);
    assert_ne!(ModelBackend::CatBoost, ModelBackend::XGBoost);
    assert_ne!(ModelBackend::LightGBM, ModelBackend::Perpetual);
}

#[test]
fn test_dataframe_with_nulls() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;
    let df = df! {
        "a" => [Some(1.0f32), None, Some(3.0)],
    }
    .unwrap();

    let result = dataframe_to_dense_f32(&df);
    // Should fail on null value
    assert!(result.is_err());
    match result {
        Err(InferError::ConversionError(msg)) => {
            assert!(msg.contains("Null"));
        }
        _ => panic!("Expected ConversionError for null value"),
    }
}
