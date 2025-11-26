use crate::error::{InferError, Result};
use crate::model::{
    DynDataFrameTransformer, DynPreprocessingPipeline, Model, ModelBackend, ModelInput,
    ModelMetadata, ModelOutput,
};
use catboost_rust::{Model as CatBoostNativeModel, ModelPolarsExt};
use std::path::Path;

/// CatBoost model wrapper
pub struct CatBoostModel {
    id: String,
    model: CatBoostNativeModel,
    metadata: Option<ModelMetadata>,
    preprocessing_pipeline: Option<DynPreprocessingPipeline>,
    transformers: Vec<DynDataFrameTransformer>,
}

impl CatBoostModel {
    /// Load a CatBoost model from a file
    pub fn load<P: AsRef<Path>>(id: String, path: P) -> Result<Self> {
        let model = CatBoostNativeModel::load(path)?;
        Ok(Self {
            id,
            model,
            metadata: None,
            preprocessing_pipeline: None,
            transformers: Vec::new(),
        })
    }

    /// Load a CatBoost model from a buffer
    pub fn load_from_buffer(id: String, buffer: &[u8]) -> Result<Self> {
        let model = CatBoostNativeModel::load_buffer(&buffer.to_vec())?;
        Ok(Self {
            id,
            model,
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

    /// Get the expected number of float features
    pub fn num_float_features(&self) -> usize {
        self.model.get_float_features_count()
    }

    /// Get the expected number of categorical features
    pub fn num_cat_features(&self) -> usize {
        self.model.get_cat_features_count()
    }
}

impl Model for CatBoostModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::CatBoost
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        // Apply DataFrame transformations
        let mut df = input.0.clone();
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Validate feature count
        let expected_features = self.num_float_features();
        if df.width() != expected_features {
            return Err(InferError::InvalidShape {
                expected: format!("{}", expected_features),
                actual: format!("{}", df.width()),
            });
        }

        // Use native Polars extension trait for optimized prediction
        let predictions = self.model.predict_dataframe(&df)?;

        // Determine if this is single or multi-output
        let dim_count = self.model.get_dimensions_count();
        if dim_count == 1 {
            Ok(ModelOutput::Single(predictions))
        } else {
            Ok(ModelOutput::Multi {
                data: predictions,
                num_classes: dim_count,
            })
        }
    }

    fn num_features(&self) -> Result<usize> {
        Ok(self.num_float_features())
    }

    fn metadata(&self) -> Option<&ModelMetadata> {
        self.metadata.as_ref()
    }

    fn preprocessing_pipeline(&self) -> Option<&DynPreprocessingPipeline> {
        self.preprocessing_pipeline.as_ref()
    }
}

// CatBoost Model is Send + Sync
unsafe impl Send for CatBoostModel {}
unsafe impl Sync for CatBoostModel {}
