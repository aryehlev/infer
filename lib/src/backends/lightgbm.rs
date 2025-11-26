use crate::error::{InferError, Result};
use crate::model::{
    DynDataFrameTransformer, DynPreprocessingPipeline, Model, ModelBackend, ModelInput,
    ModelMetadata, ModelOutput,
};
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
        // Apply DataFrame transformations
        let mut df = input.0.clone();
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Use native Polars API for zero-copy Arrow-based prediction
        let num_rows = df.height();

        // Validate feature count
        if df.width() != self.num_features_cache {
            return Err(InferError::InvalidShape {
                expected: format!("{}", self.num_features_cache),
                actual: format!("{}", df.width()),
            });
        }

        // Lock the booster for prediction
        let booster = self.booster.lock().map_err(|e| {
            InferError::Other(format!("Failed to acquire lock on LightGBM booster: {}", e))
        })?;

        // Use the native Polars extension trait for optimized prediction
        let predictions = booster.predict_dataframe(&df, 0)?;

        // LightGBM returns one value per row for binary/regression
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
