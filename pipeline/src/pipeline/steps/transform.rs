use crate::pipeline::{ContextData, ExecutionContext, PipelineStep, StepResult};
use crate::{DataFrameTransformer, Result};
use polars::prelude::*;
/// Data transformation pipeline steps
use std::sync::Arc;

/// Type alias for filter condition functions to reduce type complexity.
/// Takes a DataFrame reference and returns a boolean Series for filtering.
pub type FilterConditionFn = dyn Fn(&DataFrame) -> Result<Series> + Send + Sync;

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
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

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
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let mut df = data
            .as_dataframe()
            .ok_or_else(|| {
                infer_lib::InferError::InvalidInput(format!(
                    "Input key '{}' is not a DataFrame",
                    self.input_key
                ))
            })?
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
    condition: Arc<FilterConditionFn>,
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
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

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
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

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
        let df_data = ctx.get(&self.df_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "DataFrame key '{}' not found in context",
                self.df_key
            ))
        })?;

        let mut df = df_data
            .as_dataframe()
            .ok_or_else(|| {
                infer_lib::InferError::InvalidInput(format!(
                    "Key '{}' is not a DataFrame",
                    self.df_key
                ))
            })?
            .clone();

        // Get predictions
        let pred_data = ctx.get(&self.prediction_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Prediction key '{}' not found in context",
                self.prediction_key
            ))
        })?;

        let predictions = pred_data.as_model_output().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Key '{}' is not a ModelOutput",
                self.prediction_key
            ))
        })?;

        // Convert predictions to Series
        let series = match predictions {
            ModelOutput::Single(preds) => Series::new(self.column_name.as_str().into(), preds),
            ModelOutput::Multi { .. } => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Multi-output predictions not supported for adding to DataFrame".to_string(),
                ))
            }
            _ => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Only Single numeric predictions are supported for adding to DataFrame"
                        .to_string(),
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

// =============================================================================
// Polars LazyFrame Transform Steps
// =============================================================================

/// Type alias for LazyFrame transformation functions.
/// Takes a LazyFrame and returns a transformed LazyFrame.
pub type LazyTransformFn = dyn Fn(LazyFrame) -> LazyFrame + Send + Sync;

/// Step that applies a custom LazyFrame transformation.
///
/// This is the most flexible transformation step, allowing you to apply
/// any Polars lazy operations using a closure.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = LazyTransformStep::new(
///     "normalize",
///     "input",
///     "output",
///     |lf| {
///         lf.with_columns([
///             (col("value") - col("value").mean()) / col("value").std(1)
///         ])
///     },
/// );
/// ```
pub struct LazyTransformStep {
    name: String,
    input_key: String,
    output_key: String,
    transform: Arc<LazyTransformFn>,
}

impl LazyTransformStep {
    /// Create a new lazy transform step
    pub fn new<F>(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        transform: F,
    ) -> Self
    where
        F: Fn(LazyFrame) -> LazyFrame + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            transform: Arc::new(transform),
        }
    }
}

impl PipelineStep for LazyTransformStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        // Apply lazy transformation and collect
        let transformed = (self.transform)(df.clone().lazy()).collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Applies a custom LazyFrame transformation")
    }
}

/// Step that applies Polars expressions using `with_columns`.
///
/// This step allows you to add or modify columns using Polars expressions.
/// It's ideal for feature engineering and data preprocessing.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = WithColumnsStep::new(
///     "add_features",
///     "input",
///     "output",
///     vec![
///         (col("a") + col("b")).alias("sum"),
///         (col("a") * col("b")).alias("product"),
///         col("value").log(10.0).alias("log_value"),
///     ],
/// );
/// ```
pub struct WithColumnsStep {
    name: String,
    input_key: String,
    output_key: String,
    expressions: Vec<Expr>,
}

impl WithColumnsStep {
    /// Create a new with_columns step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        expressions: Vec<Expr>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            expressions,
        }
    }

    /// Create from a single expression
    pub fn single(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        expression: Expr,
    ) -> Self {
        Self::new(name, input_key, output_key, vec![expression])
    }
}

impl PipelineStep for WithColumnsStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let transformed = df
            .clone()
            .lazy()
            .with_columns(self.expressions.clone())
            .collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Adds or modifies columns using Polars expressions")
    }
}

/// Step that applies a `select` operation with Polars expressions.
///
/// Unlike `SelectColumnsStep` which takes column names, this step takes
/// full Polars expressions, allowing for computed columns in the output.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = SelectExprStep::new(
///     "select_computed",
///     "input",
///     "output",
///     vec![
///         col("id"),
///         (col("price") * col("quantity")).alias("total"),
///         col("category"),
///     ],
/// );
/// ```
pub struct SelectExprStep {
    name: String,
    input_key: String,
    output_key: String,
    expressions: Vec<Expr>,
}

impl SelectExprStep {
    /// Create a new select expression step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        expressions: Vec<Expr>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            expressions,
        }
    }
}

impl PipelineStep for SelectExprStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let transformed = df
            .clone()
            .lazy()
            .select(self.expressions.clone())
            .collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Selects columns using Polars expressions")
    }
}

/// Step that performs a filter operation using a Polars expression.
///
/// This is more flexible than `FilterStep` as it uses Polars' expression
/// system directly.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = FilterExprStep::new(
///     "filter_active",
///     "input",
///     "output",
///     col("status").eq(lit("active")).and(col("value").gt(lit(100))),
/// );
/// ```
pub struct FilterExprStep {
    name: String,
    input_key: String,
    output_key: String,
    predicate: Expr,
}

impl FilterExprStep {
    /// Create a new filter expression step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        predicate: Expr,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            predicate,
        }
    }
}

impl PipelineStep for FilterExprStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let transformed = df
            .clone()
            .lazy()
            .filter(self.predicate.clone())
            .collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Filters rows using a Polars expression")
    }
}

/// Step that performs groupby and aggregation operations.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = GroupByAggStep::new(
///     "aggregate_by_category",
///     "input",
///     "output",
///     vec![col("category")],
///     vec![
///         col("value").sum().alias("total_value"),
///         col("value").mean().alias("avg_value"),
///         col("id").count().alias("count"),
///     ],
/// );
/// ```
pub struct GroupByAggStep {
    name: String,
    input_key: String,
    output_key: String,
    group_by: Vec<Expr>,
    aggregations: Vec<Expr>,
}

impl GroupByAggStep {
    /// Create a new groupby aggregation step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        group_by: Vec<Expr>,
        aggregations: Vec<Expr>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            group_by,
            aggregations,
        }
    }

    /// Create with column names for grouping (convenience method)
    pub fn by_columns(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        group_by_columns: &[&str],
        aggregations: Vec<Expr>,
    ) -> Self {
        let group_by: Vec<Expr> = group_by_columns.iter().map(|c| col(*c)).collect();
        Self::new(name, input_key, output_key, group_by, aggregations)
    }
}

impl PipelineStep for GroupByAggStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let transformed = df
            .clone()
            .lazy()
            .group_by(self.group_by.clone())
            .agg(self.aggregations.clone())
            .collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Performs groupby aggregation")
    }
}

/// Step that sorts a DataFrame.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = SortStep::new(
///     "sort_by_value",
///     "input",
///     "output",
///     vec!["value", "timestamp"],
///     vec![true, false], // descending flags
/// );
/// ```
pub struct SortStep {
    name: String,
    input_key: String,
    output_key: String,
    by_columns: Vec<String>,
    descending: Vec<bool>,
}

impl SortStep {
    /// Create a new sort step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        by_columns: Vec<impl Into<String>>,
        descending: Vec<bool>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            by_columns: by_columns.into_iter().map(Into::into).collect(),
            descending,
        }
    }

    /// Create a simple ascending sort by one or more columns
    pub fn ascending(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        by_columns: Vec<impl Into<String>>,
    ) -> Self {
        let len = by_columns.len();
        Self::new(name, input_key, output_key, by_columns, vec![false; len])
    }

    /// Create a simple descending sort by one or more columns
    pub fn descending(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        by_columns: Vec<impl Into<String>>,
    ) -> Self {
        let len = by_columns.len();
        Self::new(name, input_key, output_key, by_columns, vec![true; len])
    }
}

impl PipelineStep for SortStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let sort_exprs: Vec<Expr> = self
            .by_columns
            .iter()
            .zip(self.descending.iter())
            .map(|(c, desc)| {
                if *desc {
                    col(c.as_str()).sort(SortOptions::default().with_order_descending(true))
                } else {
                    col(c.as_str()).sort(SortOptions::default())
                }
            })
            .collect();

        let transformed = df.clone().lazy().with_columns(sort_exprs).collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Sorts DataFrame by columns")
    }
}

/// Step that joins two DataFrames from the context.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = JoinStep::new(
///     "join_tables",
///     "orders",      // left DataFrame key
///     "customers",   // right DataFrame key
///     "joined",      // output key
///     vec!["customer_id"],
///     vec!["id"],
///     JoinType::Left,
/// );
/// ```
pub struct JoinStep {
    name: String,
    left_key: String,
    right_key: String,
    output_key: String,
    left_on: Vec<String>,
    right_on: Vec<String>,
    join_type: JoinType,
}

impl JoinStep {
    /// Create a new join step
    pub fn new(
        name: impl Into<String>,
        left_key: impl Into<String>,
        right_key: impl Into<String>,
        output_key: impl Into<String>,
        left_on: Vec<impl Into<String>>,
        right_on: Vec<impl Into<String>>,
        join_type: JoinType,
    ) -> Self {
        Self {
            name: name.into(),
            left_key: left_key.into(),
            right_key: right_key.into(),
            output_key: output_key.into(),
            left_on: left_on.into_iter().map(Into::into).collect(),
            right_on: right_on.into_iter().map(Into::into).collect(),
            join_type,
        }
    }

    /// Create a join on the same column names
    pub fn on_columns(
        name: impl Into<String>,
        left_key: impl Into<String>,
        right_key: impl Into<String>,
        output_key: impl Into<String>,
        on: Vec<impl Into<String>>,
        join_type: JoinType,
    ) -> Self {
        let on_vec: Vec<String> = on.into_iter().map(Into::into).collect();
        Self::new(
            name,
            left_key,
            right_key,
            output_key,
            on_vec.clone(),
            on_vec,
            join_type,
        )
    }
}

impl PipelineStep for JoinStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let left_data = ctx.get(&self.left_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Left key '{}' not found in context",
                self.left_key
            ))
        })?;

        let left_df = left_data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Left key '{}' is not a DataFrame",
                self.left_key
            ))
        })?;

        let right_data = ctx.get(&self.right_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Right key '{}' not found in context",
                self.right_key
            ))
        })?;

        let right_df = right_data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Right key '{}' is not a DataFrame",
                self.right_key
            ))
        })?;

        let left_on: Vec<&str> = self.left_on.iter().map(|s| s.as_str()).collect();
        let right_on: Vec<&str> = self.right_on.iter().map(|s| s.as_str()).collect();

        let joined = left_df.clone().lazy().join(
            right_df.clone().lazy(),
            left_on.iter().map(|s| col(*s)).collect::<Vec<_>>(),
            right_on.iter().map(|s| col(*s)).collect::<Vec<_>>(),
            JoinArgs::new(self.join_type.clone()),
        );

        let result = joined.collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(result));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Joins two DataFrames")
    }
}

/// Step that renames columns in a DataFrame.
///
/// # Example
///
/// ```ignore
/// let step = RenameColumnsStep::new(
///     "rename",
///     "input",
///     "output",
///     vec![("old_name", "new_name"), ("col_a", "feature_a")],
/// );
/// ```
pub struct RenameColumnsStep {
    name: String,
    input_key: String,
    output_key: String,
    renames: Vec<(String, String)>,
}

impl RenameColumnsStep {
    /// Create a new rename columns step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        renames: Vec<(impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            renames: renames
                .into_iter()
                .map(|(old, new)| (old.into(), new.into()))
                .collect(),
        }
    }
}

impl PipelineStep for RenameColumnsStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let mut result = df.clone();
        for (old_name, new_name) in &self.renames {
            result = result.lazy().rename([old_name.as_str()], [new_name.as_str()], true).collect()?;
        }

        ctx.insert(&self.output_key, ContextData::DataFrame(result));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Renames columns in a DataFrame")
    }
}

/// Step that casts column types.
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
///
/// let step = CastColumnsStep::new(
///     "cast_types",
///     "input",
///     "output",
///     vec![
///         ("price", DataType::Float64),
///         ("quantity", DataType::Int64),
///     ],
/// );
/// ```
pub struct CastColumnsStep {
    name: String,
    input_key: String,
    output_key: String,
    casts: Vec<(String, DataType)>,
}

impl CastColumnsStep {
    /// Create a new cast columns step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        casts: Vec<(impl Into<String>, DataType)>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            casts: casts.into_iter().map(|(c, dt)| (c.into(), dt)).collect(),
        }
    }
}

impl PipelineStep for CastColumnsStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let cast_exprs: Vec<Expr> = self
            .casts
            .iter()
            .map(|(c, dt)| col(c.as_str()).cast(dt.clone()))
            .collect();

        let transformed = df.clone().lazy().with_columns(cast_exprs).collect()?;

        ctx.insert(&self.output_key, ContextData::DataFrame(transformed));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Casts column types")
    }
}

/// Step that drops null values from specified columns.
///
/// # Example
///
/// ```ignore
/// let step = DropNullsStep::new(
///     "drop_nulls",
///     "input",
///     "output",
///     Some(vec!["important_col1", "important_col2"]),
/// );
/// ```
pub struct DropNullsStep {
    name: String,
    input_key: String,
    output_key: String,
    subset: Option<Vec<String>>,
}

impl DropNullsStep {
    /// Create a new drop nulls step
    pub fn new(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
        subset: Option<Vec<impl Into<String>>>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            subset: subset.map(|v| v.into_iter().map(Into::into).collect()),
        }
    }

    /// Drop nulls from all columns
    pub fn all(
        name: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
            subset: None,
        }
    }
}

impl PipelineStep for DropNullsStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        let data = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        let df = data.as_dataframe().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' is not a DataFrame",
                self.input_key
            ))
        })?;

        let result = match &self.subset {
            Some(cols) => {
                let col_refs: Vec<String> = cols.iter().map(|s| s.clone()).collect();
                df.clone().drop_nulls(Some(&col_refs))?
            }
            None => df.clone().drop_nulls::<String>(None)?,
        };

        ctx.insert(&self.output_key, ContextData::DataFrame(result));
        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Drops rows with null values")
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
            let transformed = df
                .lazy()
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

        let step = TransformStep::new("transform", Arc::new(TestTransformer), "input", "output");

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
        ctx.insert(
            "predictions",
            ContextData::ModelOutput(ModelOutput::Single(vec![10.0, 20.0, 30.0])),
        );

        let step =
            AddPredictionColumnStep::new("add_pred", "data", "predictions", "output", "pred");

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 2);
        assert!(output.column("pred").is_ok());
    }

    // ==========================================================================
    // Tests for Polars LazyFrame Transform Steps
    // ==========================================================================

    #[test]
    fn test_lazy_transform_step() {
        let df = df! {
            "a" => [1.0f64, 2.0, 3.0],
            "b" => [4.0f64, 5.0, 6.0],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        // Use lazy transform to add a computed column
        let step = LazyTransformStep::new("compute", "input", "output", |lf| {
            lf.with_columns([(col("a") + col("b")).alias("sum")])
        });

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 3);
        assert!(output.column("sum").is_ok());
    }

    #[test]
    fn test_with_columns_step() {
        let df = df! {
            "price" => [10.0f64, 20.0, 30.0],
            "quantity" => [2i64, 3, 4],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = WithColumnsStep::new(
            "add_total",
            "input",
            "output",
            vec![(col("price") * col("quantity").cast(DataType::Float64)).alias("total")],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 3);
        assert!(output.column("total").is_ok());
    }

    #[test]
    fn test_select_expr_step() {
        let df = df! {
            "a" => [1i64, 2, 3],
            "b" => [4i64, 5, 6],
            "c" => [7i64, 8, 9],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = SelectExprStep::new(
            "select_computed",
            "input",
            "output",
            vec![col("a"), (col("b") + col("c")).alias("bc_sum")],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.width(), 2);
        assert!(output.column("a").is_ok());
        assert!(output.column("bc_sum").is_ok());
        assert!(output.column("b").is_err());
    }

    #[test]
    fn test_filter_expr_step() {
        let df = df! {
            "value" => [10i64, 25, 5, 30],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step =
            FilterExprStep::new("filter_large", "input", "output", col("value").gt(lit(15)));

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.height(), 2); // Only 25 and 30
    }

    #[test]
    fn test_group_by_agg_step() {
        let df = df! {
            "category" => ["A", "B", "A", "B"],
            "value" => [10i64, 20, 30, 40],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = GroupByAggStep::by_columns(
            "aggregate",
            "input",
            "output",
            &["category"],
            vec![col("value").sum().alias("total")],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.height(), 2); // Two categories
        assert!(output.column("total").is_ok());
    }

    #[test]
    fn test_join_step() {
        let left = df! {
            "id" => [1i64, 2, 3],
            "value" => ["a", "b", "c"],
        }
        .unwrap();

        let right = df! {
            "id" => [1i64, 2, 4],
            "score" => [100i64, 200, 400],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("left", ContextData::DataFrame(left));
        ctx.insert("right", ContextData::DataFrame(right));

        let step = JoinStep::on_columns(
            "join",
            "left",
            "right",
            "output",
            vec!["id"],
            JoinType::Inner,
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.height(), 2); // Only ids 1 and 2 match
        assert!(output.column("value").is_ok());
        assert!(output.column("score").is_ok());
    }

    #[test]
    fn test_rename_columns_step() {
        let df = df! {
            "old_name" => [1, 2, 3],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = RenameColumnsStep::new(
            "rename",
            "input",
            "output",
            vec![("old_name", "new_name")],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert!(output.column("new_name").is_ok());
        assert!(output.column("old_name").is_err());
    }

    #[test]
    fn test_cast_columns_step() {
        let df = df! {
            "int_col" => [1i32, 2, 3],
        }
        .unwrap();

        let mut ctx = ExecutionContext::new();
        ctx.insert("input", ContextData::DataFrame(df));

        let step = CastColumnsStep::new(
            "cast",
            "input",
            "output",
            vec![("int_col", DataType::Float64)],
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_dataframe().unwrap();
        assert_eq!(output.column("int_col").unwrap().dtype(), &DataType::Float64);
    }
}
