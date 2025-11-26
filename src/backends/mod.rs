/// Backend implementations for different ML frameworks

#[cfg(feature = "catboost")]
pub mod catboost;

#[cfg(feature = "xgboost")]
pub mod xgboost;

#[cfg(feature = "lightgbm")]
pub mod lightgbm;

// Re-export backend types for convenience
#[cfg(feature = "catboost")]
pub use catboost::CatBoostModel;

#[cfg(feature = "xgboost")]
pub use xgboost::XGBoostModel;

#[cfg(feature = "lightgbm")]
pub use lightgbm::LightGBMModel;
