/// ONNX Runtime backend
use crate::{
    error::Result,
    model::{DynDataFrameTransformer, Model, ModelBackend, ModelInput, ModelOutput},
};
use ort::session::{builder::{GraphOptimizationLevel, SessionBuilder}, Session};
use polars::prelude::*;
use std::path::Path;

/// ONNX model implementation using ONNX Runtime
pub struct ONNXModel {
    id: String,
    session: Session,
    transformers: Vec<DynDataFrameTransformer>,
    input_name: String,
    output_name: String,
}

impl ONNXModel {
    /// Load an ONNX model from a file
    pub fn load(id: impl Into<String>, path: impl AsRef<Path>) -> Result<Self> {
        let session = SessionBuilder::new()
            .map_err(|e| crate::InferError::Other(format!("ONNX init error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| crate::InferError::Other(format!("ONNX config error: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| crate::InferError::Other(format!("ONNX thread config error: {}", e)))?
            .commit_from_file(path)
            .map_err(|e| crate::InferError::Other(format!("ONNX model load error: {}", e)))?;

        // Get input and output names
        let input_name = session
            .inputs
            .first()
            .ok_or_else(|| crate::InferError::Other("No input found in ONNX model".to_string()))?
            .name
            .clone();

        let output_name = session
            .outputs
            .first()
            .ok_or_else(|| crate::InferError::Other("No output found in ONNX model".to_string()))?
            .name
            .clone();

        Ok(Self {
            id: id.into(),
            session,
            transformers: Vec::new(),
            input_name,
            output_name,
        })
    }

    /// Add a DataFrame transformer to the preprocessing pipeline
    pub fn with_transformer(mut self, transformer: DynDataFrameTransformer) -> Self {
        self.transformers.push(transformer);
        self
    }

    /// Add multiple DataFrame transformers to the preprocessing pipeline
    pub fn with_transformers(mut self, transformers: Vec<DynDataFrameTransformer>) -> Self {
        self.transformers.extend(transformers);
        self
    }
}

impl Model for ONNXModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::ONNX
    }

    fn predict(&self, _input: &ModelInput) -> Result<ModelOutput> {
        // TODO: Fix ONNX Runtime 2.0 API compatibility
        // The ort 2.0 API has changed significantly from 1.x
        // This is a placeholder implementation until the API is updated
        Err(crate::InferError::Other(
            "ONNX backend temporarily disabled - API migration in progress".to_string(),
        ))
    }

    fn num_features(&self) -> Result<usize> {
        // TODO: Fix ONNX Runtime 2.0 API compatibility
        Err(crate::InferError::Other(
            "ONNX backend temporarily disabled - API migration in progress".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require actual ONNX model files
    // They are commented out but show the expected usage

    /*
    #[test]
    fn test_onnx_model_load() {
        let model = ONNXModel::load("test_model", "path/to/model.onnx").unwrap();
        assert_eq!(model.id(), "test_model");
        assert_eq!(model.backend(), ModelBackend::ONNX);
    }

    #[test]
    fn test_onnx_prediction() {
        let model = ONNXModel::load("test_model", "path/to/model.onnx").unwrap();

        let input = ModelInput::DenseF32 {
            data: vec![1.0, 2.0, 3.0, 4.0],
            num_rows: 2,
            num_features: 2,
        };

        let output = model.predict(&input).unwrap();
        match output {
            ModelOutput::Single(preds) => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("Expected Single output"),
        }
    }
    */
}
