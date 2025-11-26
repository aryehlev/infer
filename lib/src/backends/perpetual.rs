use crate::error::{InferError, Result};
use crate::model::{Model, ModelBackend, ModelInput, ModelMetadata, ModelOutput, DynPreprocessingPipeline, DynDataFrameTransformer};
use perpetual::{PerpetualBooster, Matrix};
use polars::prelude::*;
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
        // Apply DataFrame transformations
        let mut df = input.0.clone();
        for transformer in &self.transformers {
            df = transformer.transform(df)?;
        }

        // Extract f64 data directly from DataFrame (column-major for Matrix)
        let num_rows = df.height();
        let num_features = df.width();

        if num_rows == 0 || num_features == 0 {
            return Err(InferError::ConversionError(
                "DataFrame has zero rows or columns".to_string(),
            ));
        }

        let mut data_f64 = Vec::with_capacity(num_rows * num_features);
        for col in df.get_columns() {
            let series = col.as_materialized_series();
            for row_idx in 0..num_rows {
                let value = extract_f64_value(series, row_idx)?;
                data_f64.push(value);
            }
        }

        // Create Matrix and predict
        let matrix = Matrix::new(&data_f64, num_rows, num_features);
        let predictions = self.booster.predict(&matrix, true);
        Ok(ModelOutput::Single(predictions))
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

/// Extract an f64 value from a Series at the given index
fn extract_f64_value(series: &Series, idx: usize) -> Result<f64> {
    use DataType::*;

    match series.dtype() {
        Float32 => {
            let ca = series.f32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to f32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        Float64 => {
            let ca = series.f64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to f64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })?)
        }
        Int8 => {
            let ca = series.i8().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i8: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        Int16 => {
            let ca = series.i16().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i16: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        Int32 => {
            let ca = series.i32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        Int64 => {
            let ca = series.i64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        UInt8 => {
            let ca = series.u8().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u8: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        UInt16 => {
            let ca = series.u16().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u16: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        UInt32 => {
            let ca = series.u32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        UInt64 => {
            let ca = series.u64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f64)
        }
        Boolean => {
            let ca = series.bool().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to bool: {}", e))
            })?;
            Ok(if ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? {
                1.0
            } else {
                0.0
            })
        }
        dt => Err(InferError::ConversionError(format!(
            "Unsupported data type for conversion to f64: {}",
            dt
        ))),
    }
}
