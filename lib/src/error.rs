use thiserror::Error;

/// Result type for infer operations
pub type Result<T> = std::result::Result<T, InferError>;

/// Errors that can occur during inference operations
#[derive(Error, Debug)]
pub enum InferError {
    #[error("Model not found: {id}")]
    ModelNotFound { id: String },

    #[error("Invalid input shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Backend error (CatBoost): {0}")]
    CatBoostError(String),

    #[error("Backend error (XGBoost): {0}")]
    XGBoostError(String),

    #[error("Backend error (LightGBM): {0}")]
    LightGBMError(String),

    #[error("Backend error (Perpetual): {0}")]
    PerpetualError(String),

    #[error("Polars error: {0}")]
    PolarsError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

// Implement conversions from backend errors
#[cfg(feature = "catboost")]
impl From<catboost_rust::CatBoostError> for InferError {
    fn from(err: catboost_rust::CatBoostError) -> Self {
        InferError::CatBoostError(err.to_string())
    }
}

#[cfg(feature = "xgboost")]
impl From<xgboost_rust::XGBoostError> for InferError {
    fn from(err: xgboost_rust::XGBoostError) -> Self {
        InferError::XGBoostError(err.to_string())
    }
}

#[cfg(feature = "lightgbm")]
impl From<lightgbm_rust::LightGBMError> for InferError {
    fn from(err: lightgbm_rust::LightGBMError) -> Self {
        InferError::LightGBMError(err.to_string())
    }
}

#[cfg(feature = "perpetual")]
impl From<perpetual::errors::PerpetualError> for InferError {
    fn from(err: perpetual::errors::PerpetualError) -> Self {
        InferError::PerpetualError(err.to_string())
    }
}

impl From<polars::error::PolarsError> for InferError {
    fn from(err: polars::error::PolarsError) -> Self {
        InferError::PolarsError(err.to_string())
    }
}
