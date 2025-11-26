/// Data transformation pipeline steps
use std::sync::Arc;
use polars::prelude::*;
use crate::{DataFrameTransformer, Result};
use crate::pipeline::{PipelineStep, StepResult, ExecutionContext, ContextData};

/// Step that applies a DataFrame transformation
pub struct TransformStep {
    name: String,
    transformer: Arc<dyn DataFrameTransformer>,
    input_key: String,
    output_key: String,
}

impl TransformStep {
    /// Create a new transform step
    pub fn new(
        name: impl Into<String>,
        transformer: Arc<dyn DataFrameTransformer>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            transformer,
            input_key: input_key.into(),
            output_key: output_key.into(),
        }
    }
}

impl PipelineStep for TransformStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get DataFrame from context
        let data = ctx.get(&self.input_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' not found in context", self.input_key)
            ))?;

        let df = data.as_dataframe()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' is not a DataFrame", self.input_key)
            ))?;

        // Apply transformation
        let transformed = self.transformer.transform(df.clone())?;

        // Store result
        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Applies a DataFrame transformation")
    }
}

/// Step that applies a chain of transformations
pub struct TransformChainStep {
    name: String,
    transformers: Vec<Arc<dyn DataFrameTransformer>>,
    input_key: String,
    output_key: String,
}

impl TransformChainStep {
    /// Create a new transform chain step
    pub fn new(
        name: impl Into<String>,
        transformers: Vec<Arc<dyn DataFrameTransformer>>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            transformers,
            input_key: input_key.into(),
            output_key: output_key.into(),
        }
    }
}

impl PipelineStep for TransformChainStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get DataFrame from context
        let data = ctx.get(&self.input_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' not found in context", self.input_key)
            ))?;

        let mut df = data.as_dataframe()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' is not a DataFrame", self.input_key)
            ))?
            .clone();

        // Apply transformations sequentially
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Store result
        ctx.insert(&self.output_key, ContextData::DataFrame(df));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Applies a chain of DataFrame transformations")
    }
}

/// Step that filters a DataFrame based on a condition
pub struct FilterStep {
    name: String,
    input_key: String,
    output_key: String,
    condition: Arc<dyn Fn(&DataFrame) -> Result<Series> + Send + Sync>,
}

impl FilterStep {
    /// Create a new filter step
    pub fn new<F>(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        condition: F,
    ) -> Self
    where
        F: Fn(&DataFrame) -> Result<Series> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            condition: Arc::new(condition),
        }
    }
}

impl PipelineStep for FilterStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get DataFrame from context
        let data = ctx.get(&self.input_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' not found in context", self.input_key)
            ))?;

        let df = data.as_dataframe()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' is not a DataFrame", self.input_key)
            ))?;

        // Apply filter condition
        let mask = (self.condition)(df)?;
        let filtered = df.filter(mask.bool()?)?;

        // Store result
        ctx.insert(&self.output_key, ContextData::DataFrame(filtered));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Filters a DataFrame based on a condition")
    }
}

/// Step that selects columns from a DataFrame
pub struct SelectColumnsStep {
    name: String,
    input_key: String,
    output_key: String,
    columns: Vec<String>,
}

impl SelectColumnsStep {
    /// Create a new select columns step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        columns: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            columns,
        }
    }
}

impl PipelineStep for SelectColumnsStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get DataFrame from context
        let data = ctx.get(&self.input_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' not found in context", self.input_key)
            ))?;

        let df = data.as_dataframe()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Input key '{}' is not a DataFrame", self.input_key)
            ))?;

        // Select columns
        let selected = df.select(&self.columns)?;

        // Store result
        ctx.insert(&self.output_key, ContextData::DataFrame(selected));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Selects columns from a DataFrame")
    }
}

/// Step that adds a prediction column to a DataFrame
pub struct AddPredictionColumnStep {
    name: String,
    df_key: String,
    prediction_key: String,
    output_key: String,
    column_name: String,
}

impl AddPredictionColumnStep {
    /// Create a new add prediction column step
    pub fn new(
        name: impl Into<String>,
        df_key: impl Into<String>,
        prediction_key: impl Into<String>,
        output_key: impl Into<String>,
        column_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            df_key: df_key.into(),
            prediction_key: prediction_key.into(),
            output_key: output_key.into(),
            column_name: column_name.into(),
        }
    }
}

impl PipelineStep for AddPredictionColumnStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        use crate::ModelOutput;

        // Get DataFrame
        let df_data = ctx.get(&self.df_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("DataFrame key '{}' not found in context", self.df_key)
            ))?;

        let mut df = df_data.as_dataframe()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Key '{}' is not a DataFrame", self.df_key)
            ))?
            .clone();

        // Get predictions
        let pred_data = ctx.get(&self.prediction_key)
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Prediction key '{}' not found in context", self.prediction_key)
            ))?;

        let predictions = pred_data.as_model_output()
            .ok_or_else(|| infer_lib::InferError::InvalidInput(
                format!("Key '{}' is not a ModelOutput", self.prediction_key)
            ))?;

        // Convert predictions to Series
        let series = match predictions {
            ModelOutput::Single(preds) => {
                Series::new(self.column_name.as_str().into(), preds)
            }
            ModelOutput::Multi { .. } => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Multi-output predictions not supported for adding to DataFrame".to_string()
                ))
            }
            _ => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Only Single numeric predictions are supported for adding to DataFrame".to_string()
                ))
            }
        };

        // Add column to DataFrame
        let df = df.with_column(series)?.clone();

        // Store result
        ctx.insert(&self.output_key, ContextData::DataFrame(df));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Adds a prediction column to a DataFrame")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ExecutionContext;
    use crate::ModelOutput;

    struct TestTransformer;

    impl DataFrameTransformer for TestTransformer {
        fn transform(&self, df: DataFrame) -> Result<DataFrame> {
            // Add 1 to all values
            let transformed = df.lazy()
                .with_columns(vec![col("*").cast(DataType::Float64) + lit(1.0)])
                .collect()?;
            Ok(transformed)
        }

        fn name(&self) -> &str {
            "TestTransformer"
        }
    }

    #[test]
    fn test_transform_step() {
        let df = df! {
            "a" => [1.0f64, 2.0],
            "b" => [3.0f64, 4.0],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = TransformStep::new(
            "transform",
            Arc::new(TestTransformer),
            "input",
            "output",
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.shape(), (2, 2));
    }

    #[test]
    fn test_select_columns_step() {
        let df = df! {
            "a" => [1, 2, 3],
            "b" => [4, 5, 6],
            "c" => [7, 8, 9],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = SelectColumnsStep::new(
            "select",
            "input",
            "output",
            vec!["a".to_string(), "c".to_string()],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 2);
        assert!(output.column("a").is_ok());
        assert!(output.column("c").is_ok());
        assert!(output.column("b").is_err());
    }

    #[test]
    fn test_add_prediction_column_step() {
        let df = df! {
            "a" => [1, 2, 3],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("data", ContextData::DataFrame(df));
        ctx.insert("predictions", ContextData::ModelOutput(
            ModelOutput::Single(vec![10.0, 20.0, 30.0])
        ));

        let step = AddPredictionColumnStep::new(
            "add_pred",
            "data",
            "predictions",
            "output",
            "pred",
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 2);
        assert!(output.column("pred").is_ok());
    }
}
