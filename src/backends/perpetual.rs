use crate::error::{InferError, Result};
use crate::model::{Model, ModelBackend, ModelInput, ModelMetadata, ModelOutput, DynPreprocessingPipeline, DynDataFrameTransformer};
use crate::polars_ext::DataFrameExt;
use perpetual::{PerpetualBooster, Matrix};
use std::path::Path;

/// Perpetual model wrapper
pub struct PerpetualModel {
    id: String,
    booster: PerpetualBooster,
    metadata: Option<ModelMetadata>,
    preprocessing_pipeline: Option<DynPreprocessingPipeline>,
    transformers: Vec<DynDataFrameTransformer>,
}

impl PerpetualModel {
    /// Load a Perpetual model from a file
    pub fn load<P: AsRef<Path>>(id: String, path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or_else(|| {
            InferError::Other("Invalid path".to_string())
        })?;
        let booster = PerpetualBooster::load_booster(path_str)?;
        Ok(Self {
            id,
            booster,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Load a Perpetual model from a JSON string
    pub fn from_json(id: String, json: &str) -> Result<Self> {
        let booster = PerpetualBooster::from_json(json)?;
        Ok(Self {
            id,
            booster,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Set model metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set preprocessing pipeline
    pub fn with_preprocessing_pipeline(mut self, pipeline: DynPreprocessingPipeline) -> Self {
        self.preprocessing_pipeline = Some(pipeline);
        self
    }

    /// Add a DataFrame transformer
    pub fn with_transformer(mut self, transformer: DynDataFrameTransformer) -> Self {
        self.transformers.push(transformer);
        self
    }

    /// Set all DataFrame transformers at once
    pub fn with_transformers(mut self, transformers: Vec<DynDataFrameTransformer>) -> Self {
        self.transformers = transformers;
        self
    }
}

impl Model for PerpetualModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::Perpetual
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Dense {
                data,
                num_rows,
                num_features,
            } => {
                // Convert f32 to f64 and create Matrix
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let matrix = Matrix::new(&data_f64, *num_rows, *num_features);

                // Run prediction (parallel=true for multi-threading)
                let predictions = self.booster.predict(&matrix, true);

                // Perpetual returns Vec<f64> directly
                Ok(ModelOutput::Single(predictions))
            }
            ModelInput::DataFrame(df) => {
                // Apply DataFrame transformations
                let mut transformed_df = df.clone();
                for transformer in &self.transformers {
                    transformed_df = transformer.transform(transformed_df)?;
                }

                // Convert DataFrame to dense format (Perpetual doesn't have native Polars support)
                let dense_input = transformed_df.to_model_input()?;

                // Extract the dense data and predict
                match dense_input {
                    ModelInput::Dense {
                        data,
                        num_rows,
                        num_features,
                    } => {
                        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                        let matrix = Matrix::new(&data_f64, num_rows, num_features);
                        let predictions = self.booster.predict(&matrix, true);
                        Ok(ModelOutput::Single(predictions))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn num_features(&self) -> Result<usize> {
        // Perpetual doesn't expose a direct num_features method,
        // so we'll need to infer it from the model structure
        // For now, return an error indicating this needs to be set via metadata
        Err(InferError::Other(
            "Perpetual models require feature count to be specified in metadata".to_string()
        ))
    }

    fn metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }

    fn preprocessing_pipeline(&self) -> Option<&DynPreprocessingPipeline> {
        self.preprocessing_pipeline.as_ref()
    }
}

// Perpetual booster is Send + Sync
unsafe impl Send for PerpetualModel {}
unsafe impl Sync for PerpetualModel {}
