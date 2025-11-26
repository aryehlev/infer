use crate::error::{InferError, Result};
use crate::model::ModelInput;
use polars::prelude::*;

/// Extension trait for converting Polars DataFrames to ModelInput
pub trait DataFrameExt {
    /// Convert DataFrame to dense model input
    ///
    /// All columns will be converted to f32 and concatenated in column order.
    /// Non-numeric columns will cause an error.
    fn to_model_input(&self) -> Result<ModelInput>;

    /// Convert specific columns to model input
    fn to_model_input_with_columns(&self, columns: &[&str]) -> Result<ModelInput>;
}

impl DataFrameExt for DataFrame {
    fn to_model_input(&self) -> Result<ModelInput> {
        let num_rows = self.height();
        let num_features = self.width();

        if num_rows == 0 || num_features == 0 {
            return Err(InferError::ConversionError(
                "DataFrame has zero rows or columns".to_string(),
            ));
        }

        // Pre-allocate the data vector
        let mut data = Vec::with_capacity(num_rows * num_features);

        // Convert each row
        for row_idx in 0..num_rows {
            for col in self.get_columns() {
                let series = col.as_materialized_series();
                let value = extract_f32_value(series, row_idx)?;
                data.push(value);
            }
        }

        Ok(ModelInput::Dense {
            data,
            num_rows,
            num_features,
        })
    }

    fn to_model_input_with_columns(&self, columns: &[&str]) -> Result<ModelInput> {
        let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let selected = self.select(column_names)?;
        selected.to_model_input()
    }
}

/// Convert ModelOutput back to a Polars Series
pub fn output_to_series(output: &crate::model::ModelOutput, name: &str) -> Result<Series> {
    match output {
        crate::model::ModelOutput::Single(data) => {
            Ok(Series::new(name.into(), data))
        }
        crate::model::ModelOutput::Multi { data, num_classes } => {
            // For multi-output, create a list column where each row is a list of predictions
            let num_rows = data.len() / num_classes;
            let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                name.into(),
                num_rows,
                *num_classes,
                DataType::Float64,
            );

            for row_idx in 0..num_rows {
                let start = row_idx * num_classes;
                let end = start + num_classes;
                let row_data = &data[start..end];
                builder.append_slice(row_data);
            }

            Ok(builder.finish().into_series())
        }
    }
}

/// Extract an f32 value from a Series at the given index
fn extract_f32_value(series: &Series, idx: usize) -> Result<f32> {
    use DataType::*;

    match series.dtype() {
        Float32 => {
            let ca = series.f32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to f32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })?)
        }
        Float64 => {
            let ca = series.f64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to f64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        Int8 => {
            let ca = series.i8().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i8: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        Int16 => {
            let ca = series.i16().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i16: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        Int32 => {
            let ca = series.i32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        Int64 => {
            let ca = series.i64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to i64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        UInt8 => {
            let ca = series.u8().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u8: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        UInt16 => {
            let ca = series.u16().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u16: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        UInt32 => {
            let ca = series.u32().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u32: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
        }
        UInt64 => {
            let ca = series.u64().map_err(|e| {
                InferError::ConversionError(format!("Failed to cast to u64: {}", e))
            })?;
            Ok(ca.get(idx).ok_or_else(|| {
                InferError::ConversionError(format!("Null value at index {}", idx))
            })? as f32)
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
            "Unsupported data type for conversion to f32: {}",
            dt
        ))),
    }
}
