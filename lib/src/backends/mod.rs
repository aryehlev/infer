/// Backend implementations for different ML frameworks

#[cfg(feature = "catboost")]
pub mod catboost;

#[cfg(feature = "xgboost")]
pub mod xgboost;

#[cfg(feature = "lightgbm")]
pub mod lightgbm;

#[cfg(feature = "perpetual")]
pub mod perpetual;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "candle")]
pub mod candle;

// Re-export backend types for convenience
#[cfg(feature = "catboost")]
pub use catboost::CatBoostModel;

#[cfg(feature = "xgboost")]
pub use xgboost::XGBoostModel;

#[cfg(feature = "lightgbm")]
pub use lightgbm::LightGBMModel;

#[cfg(feature = "perpetual")]
pub use perpetual::PerpetualModel;

#[cfg(feature = "onnx")]
pub use onnx::ONNXModel;

#[cfg(feature = "candle")]
pub use candle::CandleModel;
