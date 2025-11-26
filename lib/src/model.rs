use crate::error::Result;
use polars::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

/// Trait for transforming Polars DataFrames before inference
///
/// DataFrameTransformers can be used in two ways:
///
/// 1. **Model-level transformers**: Attached to a model for required preprocessing
///    that should always be applied (e.g., normalization the model was trained with).
///    ```rust,ignore
///    let model = XGBoostModel::load("model", "path")?
///        .with_transformer(Arc::new(MyTransformer));
///    ```
///
/// 2. **Pipeline-level transformers**: Used in `TransformStep` for flexible,
///    composable workflows where different preprocessing may be needed.
///    ```rust,ignore
///    let pipeline = Pipeline::builder("workflow")
///        .add_step(Arc::new(TransformStep::new(
///            "transform",
///            Arc::new(MyTransformer),
///            "input",
///            "output",
///        )))
///        .build();
///    ```
///
/// **Best Practice**: Use model-level for required preprocessing, pipeline-level for flexible composition.
pub trait DataFrameTransformer: Send + Sync {
    /// Apply transformation to the DataFrame
    fn transform(&self, df: DataFrame) -> Result<DataFrame>;

    /// Get a description of this transformation
    fn name(&self) -> &str;
}

/// Type-erased DataFrame transformer
pub type DynDataFrameTransformer = Arc<dyn DataFrameTransformer>;

/// Input data for model inference
///
/// All models accept Polars DataFrames as input. Each backend handles
/// DataFrame conversion internally (native Polars support or conversion to dense).
#[derive(Debug, Clone)]
pub struct ModelInput(pub DataFrame);

/// Output from model inference
#[derive(Debug, Clone)]
pub enum ModelOutput {
    /// Single numeric predictions per row (regression, binary classification)
    Single(Vec<f64>),

    /// Multi-output numeric predictions (multiclass probabilities)
    /// Shape: (num_rows, num_classes)
    Multi {
        data: Vec<f64>,
        num_classes: usize
    },

    /// Text output (LLM generation, translation)
    Text(Vec<String>),

    /// Single text output
    TextSingle(String),

    /// Embeddings/vectors (sentence embeddings, feature vectors)
    Embeddings {
        data: Vec<f32>,
        embedding_dim: usize,
    },

    /// Token IDs (for tokenized output)
    Tokens(Vec<Vec<i64>>),

    /// Classification with labels
    Classifications {
        labels: Vec<String>,
        scores: Vec<f64>,
    },

    /// Custom/mixed output
    Custom(std::collections::HashMap<String, Vec<u8>>),
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
    /// Get the raw prediction data (only for numeric outputs)
    pub fn as_slice(&self) -> Option<&[f64]> {
        match self {
            ModelOutput::Single(data) => Some(data),
            ModelOutput::Multi { data, .. } => Some(data),
            _ => None,
        }
    }

    /// Convert to Vec (only for numeric outputs)
    pub fn into_vec(self) -> Option<Vec<f64>> {
        match self {
            ModelOutput::Single(data) => Some(data),
            ModelOutput::Multi { data, .. } => Some(data),
            _ => None,
        }
    }

    /// Get number of rows
    pub fn num_rows(&self) -> usize {
        match self {
            ModelOutput::Single(data) => data.len(),
            ModelOutput::Multi { data, num_classes } => data.len() / num_classes,
            ModelOutput::Text(texts) => texts.len(),
            ModelOutput::TextSingle(_) => 1,
            ModelOutput::Embeddings { data, embedding_dim } => data.len() / embedding_dim,
            ModelOutput::Tokens(tokens) => tokens.len(),
            ModelOutput::Classifications { labels, .. } => labels.len(),
            ModelOutput::Custom(_) => 0,
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
    Perpetual,
    ONNX,
    Candle,
}

impl std::fmt::Display for ModelBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelBackend::CatBoost => write!(f, "CatBoost"),
            ModelBackend::XGBoost => write!(f, "XGBoost"),
            ModelBackend::LightGBM => write!(f, "LightGBM"),
            ModelBackend::Perpetual => write!(f, "Perpetual"),
            ModelBackend::ONNX => write!(f, "ONNX"),
            ModelBackend::Candle => write!(f, "Candle"),
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
