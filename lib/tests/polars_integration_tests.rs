use infer_lib::prelude::*;
use polars::prelude::*;

#[test]
fn test_dataframe_input_basic() {
    let df = df! {
        "feature1" => [1.0f32, 2.0, 3.0],
        "feature2" => [4.0f32, 5.0, 6.0],
    }
    .unwrap();

    // ModelInput now just wraps the DataFrame
    let input = ModelInput(df);

    // Verify we can access the DataFrame
    assert_eq!(input.0.height(), 3);
    assert_eq!(input.0.width(), 2);
}

#[test]
fn test_dataframe_empty_error() {
    let df = DataFrame::empty();

    // Empty DataFrames are valid to create, but may cause errors during prediction
    let input = ModelInput(df);
    assert_eq!(input.0.height(), 0);
}

#[test]
fn test_dataframe_with_multiple_types() {
    let df = df! {
        "float32" => [1.0f32, 2.0, 3.0],
        "float64" => [4.0f64, 5.0, 6.0],
        "int32" => [7i32, 8, 9],
        "int64" => [10i64, 11, 12],
        "bool" => [true, false, true],
    }
    .unwrap();

    let input = ModelInput(df);

    assert_eq!(input.0.height(), 3);
    assert_eq!(input.0.width(), 5);
}

#[test]
fn test_output_to_series_single() {
    let output = ModelOutput::Single(vec![1.0, 2.0, 3.0]);
    let series = infer_lib::output_to_series(&output, "predictions").unwrap();

    assert_eq!(series.name(), "predictions");
    assert_eq!(series.len(), 3);
    assert_eq!(series.f64().unwrap().get(0), Some(1.0));
    assert_eq!(series.f64().unwrap().get(1), Some(2.0));
    assert_eq!(series.f64().unwrap().get(2), Some(3.0));
}

#[test]
fn test_output_to_series_multi() {
    let output = ModelOutput::Multi {
        data: vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1],
        num_classes: 3,
    };
    let series = infer_lib::output_to_series(&output, "class_probs").unwrap();

    assert_eq!(series.name(), "class_probs");
    assert_eq!(series.len(), 2); // 2 rows

    // Check it's a list type
    let list = series.list().unwrap();
    assert_eq!(list.len(), 2);
}

#[test]
fn test_model_output_methods() {
    let single = ModelOutput::Single(vec![1.0, 2.0, 3.0]);
    assert_eq!(single.num_rows(), 3);
    assert_eq!(single.as_slice(), Some(&[1.0, 2.0, 3.0][..]));

    let multi = ModelOutput::Multi {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        num_classes: 2,
    };
    assert_eq!(multi.num_rows(), 3);
    assert_eq!(multi.as_slice().unwrap().len(), 6);
}

#[test]
fn test_dataframe_to_dense_f32_conversion() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;

    let df = df! {
        "a" => [1.0f32, 2.0],
        "b" => [3.0f32, 4.0],
    }
    .unwrap();

    let (data, num_rows, num_features) = dataframe_to_dense_f32(&df).unwrap();

    assert_eq!(num_rows, 2);
    assert_eq!(num_features, 2);
    assert_eq!(data.len(), 4);
    // Data is row-major: [1.0, 3.0, 2.0, 4.0]
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 3.0);
    assert_eq!(data[2], 2.0);
    assert_eq!(data[3], 4.0);
}

#[test]
fn test_boolean_conversion() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;

    let df = df! {
        "bool_col" => [true, false, true, false],
    }
    .unwrap();

    let (data, _, _) = dataframe_to_dense_f32(&df).unwrap();

    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 1.0);
    assert_eq!(data[3], 0.0);
}

#[test]
fn test_unsigned_integer_types() {
    use infer_lib::polars_ext::dataframe_to_dense_f32;

    let df = df! {
        "u8" => [1u8, 2, 3],
        "u16" => [100u16, 200, 300],
        "u32" => [1000u32, 2000, 3000],
        "u64" => [10000u64, 20000, 30000],
    }
    .unwrap();

    let (data, num_rows, num_features) = dataframe_to_dense_f32(&df).unwrap();

    assert_eq!(num_rows, 3);
    assert_eq!(num_features, 4);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 100.0);
    assert_eq!(data[2], 1000.0);
    assert_eq!(data[3], 10000.0);
}
