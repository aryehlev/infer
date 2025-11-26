use crate::error::Result;
use polars::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Trait for transforming Polars DataFrames before inference
pub trait DataFrameTransformer: Send + Sync {
    /// Apply transformation to the DataFrame
    fn transform(&self, df: DataFrame) -> Result<DataFrame>;

    /// Get a description of this transformation
    fn name(&self) -> &str;
}

/// Type-erased DataFrame transformer
pub type DynDataFrameTransformer = Arc<dyn DataFrameTransformer>;

/// Input data for model inference
#[derive(Debug, Clone)]
pub enum ModelInput {
    /// Dense float features as a flat vector (row-major format)
    /// Shape: (num_rows, num_features)
    Dense {
        data: Vec<f32>,
        num_rows: usize,
        num_features: usize,
    },

    /// Polars DataFrame input (will be converted to dense format)
    DataFrame(DataFrame),
}

/// Output from model inference
#[derive(Debug, Clone)]
pub enum ModelOutput {
    /// Single predictions per row
    Single(Vec<f64>),

    /// Multi-output predictions (e.g., multiclass probabilities)
    /// Shape: (num_rows, num_classes)
    Multi { data: Vec<f64>, num_classes: usize },
}

/// Result from model inference including optional metadata
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The prediction output
    pub output: ModelOutput,

    /// Optional metadata about the inference
    pub metadata: Option<InferenceMetadata>,
}

/// Metadata returned with inference results
#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    /// Model ID that produced the result
    pub model_id: String,

    /// Model backend used
    pub backend: ModelBackend,

    /// Custom key-value metadata
    pub custom: HashMap<String, String>,
}

impl ModelOutput {
    /// Get the raw prediction data
    pub fn as_slice(&self) -> &[f64] {
        match self {
            ModelOutput::Single(data) => data,
            ModelOutput::Multi { data, .. } => data,
        }
    }

    /// Convert to Vec
    pub fn into_vec(self) -> Vec<f64> {
        match self {
            ModelOutput::Single(data) => data,
            ModelOutput::Multi { data, .. } => data,
        }
    }

    /// Get number of rows
    pub fn num_rows(&self) -> usize {
        match self {
            ModelOutput::Single(data) => data.len(),
            ModelOutput::Multi { data, num_classes } => data.len() / num_classes,
        }
    }
}

/// Trait for preprocessing pipelines that transform input data before inference
pub trait PreprocessingPipeline: Send + Sync {
    /// Apply preprocessing to the input data
    fn preprocess(&self, input: ModelInput) -> Result<ModelInput>;

    /// Get a description of the preprocessing steps
    fn description(&self) -> Option<String> {
        None
    }
}

/// Type-erased preprocessing pipeline
pub type DynPreprocessingPipeline = Arc<dyn PreprocessingPipeline>;

/// Core trait that all ML models must implement
pub trait Model: Send + Sync {
    /// Get the model's unique identifier
    fn id(&self) -> &str;

    /// Get the model backend type
    fn backend(&self) -> ModelBackend;

    /// Run inference on the input data
    fn predict(&self, input: &ModelInput) -> Result<ModelOutput>;

    /// Get the expected number of input features
    fn num_features(&self) -> Result<usize>;

    /// Get model metadata (optional)
    fn metadata(&self) -> Option<&ModelMetadata> {
        None
    }

    /// Get the preprocessing pipeline (optional)
    fn preprocessing_pipeline(&self) -> Option<&DynPreprocessingPipeline> {
        None
    }
}

/// Model backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBackend {
    CatBoost,
    XGBoost,
    LightGBM,
}

impl std::fmt::Display for ModelBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelBackend::CatBoost => write!(f, "CatBoost"),
            ModelBackend::XGBoost => write!(f, "XGBoost"),
            ModelBackend::LightGBM => write!(f, "LightGBM"),
        }
    }
}

/// Optional metadata for models
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub created_at: Option<String>,
    pub tags: Vec<String>,
}

/// Type-erased model that can hold any backend implementation
pub type DynModel = Arc<dyn Model>;
