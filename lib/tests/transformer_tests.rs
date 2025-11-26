use infer_lib::prelude::*;
use polars::prelude::*;
use std::sync::Arc;

// Test transformer that adds 1 to all values
struct AddOneTransformer;

impl DataFrameTransformer for AddOneTransformer {
    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        let transformed = df
            .lazy()
            .with_columns(vec![col("*").cast(DataType::Float64) + lit(1.0)])
            .collect()?;
        Ok(transformed)
    }

    fn name(&self) -> &str {
        "AddOne"
    }
}

// Test transformer that multiplies by 2
struct MultiplyByTwoTransformer;

impl DataFrameTransformer for MultiplyByTwoTransformer {
    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        let transformed = df
            .lazy()
            .with_columns(vec![col("*").cast(DataType::Float64) * lit(2.0)])
            .collect()?;
        Ok(transformed)
    }

    fn name(&self) -> &str {
        "MultiplyByTwo"
    }
}

// Transformer that fails
struct FailingTransformer;

impl DataFrameTransformer for FailingTransformer {
    fn transform(&self, _df: DataFrame) -> Result<DataFrame> {
        Err(InferError::Other("Intentional failure".to_string()))
    }

    fn name(&self) -> &str {
        "Failing"
    }
}

#[test]
fn test_single_transformer() {
    let df = df! {
        "a" => [1.0f64, 2.0, 3.0],
        "b" => [4.0f64, 5.0, 6.0],
    }
    .unwrap();

    let transformer = AddOneTransformer;
    let result = transformer.transform(df).unwrap();

    assert_eq!(result.column("a").unwrap().f64().unwrap().get(0), Some(2.0));
    assert_eq!(result.column("b").unwrap().f64().unwrap().get(0), Some(5.0));
}

#[test]
fn test_transformer_chain() {
    let df = df! {
        "a" => [1.0f64, 2.0, 3.0],
    }
    .unwrap();

    // Apply: +1 then *2 = (1+1)*2 = 4
    let transformer1 = AddOneTransformer;
    let transformer2 = MultiplyByTwoTransformer;

    let result = transformer1.transform(df).unwrap();
    let result = transformer2.transform(result).unwrap();

    assert_eq!(result.column("a").unwrap().f64().unwrap().get(0), Some(4.0));
}

#[test]
fn test_transformer_name() {
    let transformer = AddOneTransformer;
    assert_eq!(transformer.name(), "AddOne");
}

#[test]
fn test_failing_transformer() {
    let df = df! {
        "a" => [1.0f64, 2.0, 3.0],
    }
    .unwrap();

    let transformer = FailingTransformer;
    let result = transformer.transform(df);
    assert!(result.is_err());
}

#[test]
fn test_dyn_transformer() {
    let df = df! {
        "a" => [1.0f64, 2.0, 3.0],
    }
    .unwrap();

    let transformer: DynDataFrameTransformer = Arc::new(AddOneTransformer);
    let result = transformer.transform(df).unwrap();

    assert_eq!(result.column("a").unwrap().f64().unwrap().get(0), Some(2.0));
}

#[test]
fn test_vec_of_transformers() {
    let df = df! {
        "a" => [1.0f64],
    }
    .unwrap();

    let transformers: Vec<DynDataFrameTransformer> = vec![
        Arc::new(AddOneTransformer),
        Arc::new(MultiplyByTwoTransformer),
        Arc::new(AddOneTransformer),
    ];

    let mut result = df;
    for transformer in &transformers {
        result = transformer.transform(result).unwrap();
    }

    // ((1 + 1) * 2) + 1 = 5
    assert_eq!(result.column("a").unwrap().f64().unwrap().get(0), Some(5.0));
}

// Test that transformers preserve schema
#[test]
fn test_transformer_preserves_columns() {
    let df = df! {
        "a" => [1.0f64, 2.0],
        "b" => [3.0f64, 4.0],
        "c" => [5.0f64, 6.0],
    }
    .unwrap();

    let transformer = AddOneTransformer;
    let result = transformer.transform(df).unwrap();

    assert_eq!(result.width(), 3);
    assert_eq!(result.height(), 2);
    assert!(result.column("a").is_ok());
    assert!(result.column("b").is_ok());
    assert!(result.column("c").is_ok());
}

// Test transformer thread safety
#[test]
fn test_transformer_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AddOneTransformer>();
    assert_send_sync::<DynDataFrameTransformer>();
}
