use crate::error::{InferError, Result};
use crate::model::{Model, ModelBackend, ModelInput, ModelMetadata, ModelOutput, DynPreprocessingPipeline, DynDataFrameTransformer};
use std::path::Path;
use xgboost_rust::{Booster, BoosterPolarsExt};

/// XGBoost model wrapper
pub struct XGBoostModel {
    id: String,
    booster: Booster,
    metadata: Option<ModelMetadata>,
    preprocessing_pipeline: Option<DynPreprocessingPipeline>,
    transformers: Vec<DynDataFrameTransformer>,
}

impl XGBoostModel {
    /// Load an XGBoost model from a file
    pub fn load<P: AsRef<Path>>(id: String, path: P) -> Result<Self> {
        let booster = Booster::load(path)?;
        Ok(Self {
            id,
            booster,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Load an XGBoost model from a buffer
    pub fn load_from_buffer(id: String, buffer: &[u8]) -> Result<Self> {
        let booster = Booster::load_from_buffer(buffer)?;
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

impl Model for XGBoostModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::XGBoost
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Dense {
                data,
                num_rows,
                num_features,
            } => {
                // Validate feature count
                let expected_features = self.booster.num_features()?;
                if *num_features != expected_features {
                    return Err(InferError::InvalidShape {
                        expected: format!("{}", expected_features),
                        actual: format!("{}", num_features),
                    });
                }

                // Run prediction on dense data
                // option_mask: 0 = normal prediction
                // training: false = inference mode
                let predictions = self.booster.predict(data, *num_rows, *num_features, 0, false)?;

                // Convert f32 to f64
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();

                // XGBoost returns one value per row for binary/regression
                // For multiclass, it returns num_rows * num_classes values
                if predictions_f64.len() == *num_rows {
                    Ok(ModelOutput::Single(predictions_f64))
                } else {
                    let num_classes = predictions_f64.len() / num_rows;
                    Ok(ModelOutput::Multi {
                        data: predictions_f64,
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
                let expected_features = self.booster.num_features()?;
                if transformed_df.width() != expected_features {
                    return Err(InferError::InvalidShape {
                        expected: format!("{}", expected_features),
                        actual: format!("{}", transformed_df.width()),
                    });
                }

                // Use the native Polars extension trait for optimized prediction
                // option_mask: 0 = normal prediction, training: false = inference mode
                let predictions = self.booster.predict_dataframe(&transformed_df, 0, false)?;

                // Convert f32 to f64
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();

                // XGBoost returns one value per row for binary/regression
                // For multiclass, it returns num_rows * num_classes values
                if predictions_f64.len() == num_rows {
                    Ok(ModelOutput::Single(predictions_f64))
                } else {
                    let num_classes = predictions_f64.len() / num_rows;
                    Ok(ModelOutput::Multi {
                        data: predictions_f64,
                        num_classes,
                    })
                }
            }
        }
    }

    fn num_features(&self) -> Result<usize> {
        Ok(self.booster.num_features()?)
    }

    fn metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }

    fn preprocessing_pipeline(&self) -> Option<&DynPreprocessingPipeline> {
        self.preprocessing_pipeline.as_ref()
    }
}

// XGBoost Booster is Send + Sync (for version >= 1.4)
// The safety is guaranteed by xgboost-rust's conditional implementation
