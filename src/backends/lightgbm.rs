use crate::error::{InferError, Result};
use crate::model::{Model, ModelBackend, ModelInput, ModelMetadata, ModelOutput, DynPreprocessingPipeline, DynDataFrameTransformer};
use lightgbm_rust::{Booster, BoosterPolarsExt};
use std::path::Path;
use std::sync::Mutex;

/// LightGBM model wrapper
///
/// Note: LightGBM's C API doesn't guarantee thread-safety for concurrent predictions,
/// so we wrap the booster in a Mutex to ensure safe concurrent access.
pub struct LightGBMModel {
    id: String,
    booster: Mutex<Booster>,
    num_features_cache: usize,
    metadata: Option<ModelMetadata>,
    preprocessing_pipeline: Option<DynPreprocessingPipeline>,
    transformers: Vec<DynDataFrameTransformer>,
}

impl LightGBMModel {
    /// Load a LightGBM model from a file
    pub fn load<P: AsRef<Path>>(id: String, path: P) -> Result<Self> {
        let booster = Booster::load(path)?;
        let num_features_cache = booster.num_features()? as usize;
        Ok(Self {
            id,
            booster: Mutex::new(booster),
            num_features_cache,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Load a LightGBM model from a buffer
    pub fn load_from_buffer(id: String, buffer: &[u8]) -> Result<Self> {
        let booster = Booster::load_from_buffer(buffer)?;
        let num_features_cache = booster.num_features()? as usize;
        Ok(Self {
            id,
            booster: Mutex::new(booster),
            num_features_cache,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Load a LightGBM model from a string
    pub fn load_from_string(id: String, model_str: &str) -> Result<Self> {
        let booster = Booster::load_from_string(model_str)?;
        let num_features_cache = booster.num_features()? as usize;
        Ok(Self {
            id,
            booster: Mutex::new(booster),
            num_features_cache,
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

impl Model for LightGBMModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::LightGBM
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Dense {
                data,
                num_rows,
                num_features,
            } => {
                // Validate feature count
                if *num_features != self.num_features_cache {
                    return Err(InferError::InvalidShape {
                        expected: format!("{}", self.num_features_cache),
                        actual: format!("{}", num_features),
                    });
                }

                // Lock the booster for prediction
                let booster = self.booster.lock().map_err(|e| {
                    InferError::Other(format!("Failed to acquire lock on LightGBM booster: {}", e))
                })?;

                // Convert f32 to f64 for LightGBM
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

                // Run prediction on dense data
                // predict_type: 0 = normal prediction
                let predictions = booster.predict(&data_f64, *num_rows as i32, *num_features as i32, 0)?;

                // LightGBM returns one value per row for binary/regression
                // For multiclass, it returns num_rows * num_classes values
                if predictions.len() == *num_rows {
                    Ok(ModelOutput::Single(predictions))
                } else {
                    let num_classes = predictions.len() / num_rows;
                    Ok(ModelOutput::Multi {
                        data: predictions,
                        num_classes,
                    })
                }
            }
            ModelInput::DataFrame(df) => {
                // Apply DataFrame transformations
                let mut transformed_df = df.clone();
                for transformer in &self.transformers {
                    transformed_df = transformer.transform(transformed_df)?;
                }

                // Use native Polars API for zero-copy Arrow-based prediction
                let num_rows = transformed_df.height();

                // Validate feature count
                if transformed_df.width() != self.num_features_cache {
                    return Err(InferError::InvalidShape {
                        expected: format!("{}", self.num_features_cache),
                        actual: format!("{}", transformed_df.width()),
                    });
                }

                // Lock the booster for prediction
                let booster = self.booster.lock().map_err(|e| {
                    InferError::Other(format!("Failed to acquire lock on LightGBM booster: {}", e))
                })?;

                // Use the native Polars extension trait for optimized prediction
                // predict_type: 0 = normal prediction
                let predictions = booster.predict_dataframe(&transformed_df, 0)?;

                // LightGBM returns one value per row for binary/regression
                // For multiclass, it returns num_rows * num_classes values
                if predictions.len() == num_rows {
                    Ok(ModelOutput::Single(predictions))
                } else {
                    let num_classes = predictions.len() / num_rows;
                    Ok(ModelOutput::Multi {
                        data: predictions,
                        num_classes,
                    })
                }
            }
        }
    }

    fn num_features(&self) -> Result<usize> {
        Ok(self.num_features_cache)
    }

    fn metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }

    fn preprocessing_pipeline(&self) -> Option<&DynPreprocessingPipeline> {
        self.preprocessing_pipeline.as_ref()
    }
}

// LightGBMModel is Send + Sync thanks to the Mutex wrapper
unsafe impl Send for LightGBMModel {}
unsafe impl Sync for LightGBMModel {}
