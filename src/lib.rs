//! # Infer
//!
//! A high-performance ML inference library supporting CatBoost, XGBoost, and LightGBM
//! with parallel execution and Polars DataFrame integration.
//!
//! ## Features
//!
//! - **Multiple ML Backends**: Support for CatBoost, XGBoost, and LightGBM
//! - **Lock-Free Model Registry**: Use `ArcSwap` for zero-downtime model updates
//! - **Parallel Inference**: Run inference on multiple models simultaneously
//! - **Polars Integration**: Native support for Polars DataFrames
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
//! let input = ModelInput::Dense {
//!     data: vec![1.0, 2.0, 3.0, 4.0],
//!     num_rows: 2,
//!     num_features: 2,
//! };
//! let output = registry.predict("model1", &input).unwrap();
//! ```

pub mod error;
pub mod model;
pub mod registry;
pub mod backends;
pub mod polars_ext;

// Re-export commonly used types
pub use error::{InferError, Result};
pub use model::{
    Model, ModelBackend, ModelInput, ModelOutput, ModelMetadata, DynModel,
    PreprocessingPipeline, DynPreprocessingPipeline, InferenceResult, InferenceMetadata,
    DataFrameTransformer, DynDataFrameTransformer,
};
pub use registry::ModelRegistry;
pub use polars_ext::{DataFrameExt, output_to_series};

// Re-export backend types when features are enabled
#[cfg(feature = "catboost")]
pub use backends::CatBoostModel;

#[cfg(feature = "xgboost")]
pub use backends::XGBoostModel;

#[cfg(feature = "lightgbm")]
pub use backends::LightGBMModel;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::error::{InferError, Result};
    pub use crate::model::{
        Model, ModelBackend, ModelInput, ModelOutput, DynModel,
        PreprocessingPipeline, DynPreprocessingPipeline, InferenceResult, InferenceMetadata,
        DataFrameTransformer, DynDataFrameTransformer,
    };
    pub use crate::registry::ModelRegistry;
    pub use crate::polars_ext::DataFrameExt;

    #[cfg(feature = "catboost")]
    pub use crate::backends::CatBoostModel;

    #[cfg(feature = "xgboost")]
    pub use crate::backends::XGBoostModel;

    #[cfg(feature = "lightgbm")]
    pub use crate::backends::LightGBMModel;
}
