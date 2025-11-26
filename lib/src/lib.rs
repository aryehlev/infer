//! # Infer-Lib
//!
//! A high-performance ML inference library for Rust with an integrated execution engine
//! for building ML pipelines and RAG (Retrieval-Augmented Generation) workflows.
//!
//! ## Features
//!
//! - **Multiple ML Backends**: Support for CatBoost, XGBoost, LightGBM, and Perpetual
//! - **Execution Engine**: Build complex ML pipelines with conditional branching and parallel execution
//! - **RAG Support**: Built-in pipeline steps for retrieval-augmented generation workflows
//! - **Lock-Free Model Registry**: Use `ArcSwap` for zero-downtime model updates
//! - **Parallel Inference**: Run inference on multiple models simultaneously
//! - **Polars Integration**: Native support for Polars DataFrames with zero-copy Arrow transfer
//! - **Type-Safe**: Unified interface across all backends
//!
//! ## Example
//!
//! ```rust,no_run
//! use infer::{ModelRegistry, backends::XGBoostModel};
//!
//! // Create a registry
//! let registry = ModelRegistry::new();
//!
//! // Load and register a model
//! let model = XGBoostModel::load("model1".to_string(), "model.json").unwrap();
//! registry.register("model1".to_string(), model);
//!
//! // Run inference
//! # use infer::ModelInput;
//! let input = ModelInput::DenseF32 {
//!     data: vec![1.0, 2.0, 3.0, 4.0],
//!     num_rows: 2,
//!     num_features: 2,
//! };
//! let output = registry.predict("model1", &input).unwrap();
//! ```

pub mod backends;
pub mod error;
pub mod model;
pub mod polars_ext;
pub mod registry;

// Re-export commonly used types
pub use error::{InferError, Result};
pub use model::{
    DataFrameTransformer, DynDataFrameTransformer, DynModel, DynPreprocessingPipeline,
    InferenceMetadata, InferenceResult, Model, ModelBackend, ModelInput, ModelMetadata,
    ModelOutput, PreprocessingPipeline,
};
pub use polars_ext::{dataframe_to_dense_f32, output_to_series};
pub use registry::ModelRegistry;

// Re-export backend types when features are enabled
#[cfg(feature = "catboost")]
pub use backends::CatBoostModel;

#[cfg(feature = "xgboost")]
pub use backends::XGBoostModel;

#[cfg(feature = "lightgbm")]
pub use backends::LightGBMModel;

#[cfg(feature = "perpetual")]
pub use backends::PerpetualModel;

#[cfg(feature = "onnx")]
pub use backends::ONNXModel;

#[cfg(feature = "candle")]
pub use backends::CandleModel;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::error::{InferError, Result};
    pub use crate::model::{
        DataFrameTransformer, DynDataFrameTransformer, DynModel, DynPreprocessingPipeline,
        InferenceMetadata, InferenceResult, Model, ModelBackend, ModelInput, ModelOutput,
        PreprocessingPipeline,
    };
    pub use crate::polars_ext::{dataframe_to_dense_f32, output_to_series};
    pub use crate::registry::ModelRegistry;

    #[cfg(feature = "catboost")]
    pub use crate::backends::CatBoostModel;

    #[cfg(feature = "xgboost")]
    pub use crate::backends::XGBoostModel;

    #[cfg(feature = "lightgbm")]
    pub use crate::backends::LightGBMModel;

    #[cfg(feature = "perpetual")]
    pub use crate::backends::PerpetualModel;

    #[cfg(feature = "onnx")]
    pub use crate::backends::ONNXModel;

    #[cfg(feature = "candle")]
    pub use crate::backends::CandleModel;
}
