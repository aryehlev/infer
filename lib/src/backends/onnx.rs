/// ONNX Runtime backend
use crate::{
    error::Result,
    model::{DataFrameTransformer, DynDataFrameTransformer, Model, ModelBackend, ModelInput, ModelOutput},
    polars_ext::DataFrameExt,
};
use ort::{GraphOptimizationLevel, Session, SessionBuilder, Value};
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
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

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

        fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        // Apply DataFrame transformations
        let mut df = input.0.clone();
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Convert DataFrame to dense f32
        use crate::polars_ext::dataframe_to_dense_f32;
        let (data, num_rows, num_features) = dataframe_to_dense_f32(&df)?;

        // Create ONNX tensor
        let shape = vec![num_rows, num_features];
        let tensor = Value::from_array(self.session.allocator(), &data, &shape)?;

        // Run inference
        let outputs = self.session.run(vec![(&self.input_name, tensor)])?;

        // Extract output - handle different tensor types
        let output_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| crate::InferError::Other(format!("Output '{}' not found", self.output_name)))?;

        // Try to extract as different types and convert to f64
        let (predictions, shape) = if let Ok(data) = output_tensor.try_extract_tensor::<f32>() {
            let shape = data.shape().to_vec();
            let preds: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            (preds, shape)
        } else if let Ok(data) = output_tensor.try_extract_tensor::<f64>() {
            let shape = data.shape().to_vec();
            let preds: Vec<f64> = data.iter().copied().collect();
            (preds, shape)
        } else if let Ok(data) = output_tensor.try_extract_tensor::<i64>() {
            let shape = data.shape().to_vec();
            let preds: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            (preds, shape)
        } else if let Ok(data) = output_tensor.try_extract_tensor::<i32>() {
            let shape = data.shape().to_vec();
            let preds: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            (preds, shape)
        } else {
            return Err(crate::InferError::Other(
                "Unsupported ONNX output tensor type (supported: f32, f64, i32, i64)".to_string()
            ));
        };

        // Determine output format based on shape
        if shape.len() == 1 || (shape.len() == 2 && shape[1] == 1) {
            Ok(ModelOutput::Single(predictions))
        } else if shape.len() == 2 {
            let num_classes = shape[1];
            Ok(ModelOutput::Multi {
                data: predictions,
                num_classes,
            })
        } else {
            Err(crate::InferError::Other(format!(
                "Unsupported output shape: {:?}",
                shape
            )))
        }
    }

    fn num_features(&self) -> Result<usize> {
        let input = self
            .session
            .inputs
            .first()
            .ok_or_else(|| crate::InferError::Other("No input found in ONNX model".to_string()))?;

        // Get the feature dimension (usually the last dimension)
        if let Some(shape) = &input.input_type.tensor_dimensions() {
            if let Some(&num_features) = shape.last() {
                Ok(num_features as usize)
            } else {
                Err(crate::InferError::Other("Could not determine number of features".to_string()))
            }
        } else {
            Err(crate::InferError::Other("Input is not a tensor".to_string()))
        }
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
