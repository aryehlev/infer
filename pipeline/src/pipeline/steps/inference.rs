use crate::pipeline::{ContextData, ExecutionContext, PipelineStep, StepResult};
use crate::{ModelInput, ModelRegistry, Result};
/// Model inference pipeline steps
use std::sync::Arc;

/// Step that runs model inference
pub struct InferenceStep {
    name: String,
    registry: Arc<ModelRegistry>,
    model_id: String,
    input_key: String,
    output_key: String,
}

impl InferenceStep {
    /// Create a new inference step
    pub fn new(
        name: impl Into<String>,
        registry: Arc<ModelRegistry>,
        model_id: impl Into<String>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            registry,
            model_id: model_id.into(),
            input_key: input_key.into(),
            output_key: output_key.into(),
        }
    }
}

impl PipelineStep for InferenceStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get input from context
        let input = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        // Convert to ModelInput if it's a DataFrame
        let model_input = match input {
            ContextData::ModelInput(input) => input.clone(),
            ContextData::DataFrame(df) => ModelInput(df.clone()),
            _ => {
                return Err(infer_lib::InferError::InvalidInput(format!(
                    "Input key '{}' is not a valid model input",
                    self.input_key
                )))
            }
        };

        // Run inference
        let output = self.registry.predict(&self.model_id, &model_input)?;

        // Store output in context
        ctx.insert(&self.output_key, ContextData::ModelOutput(output));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Runs model inference")
    }
}

/// Step that runs inference on multiple models in parallel
pub struct ParallelInferenceStep {
    name: String,
    registry: Arc<ModelRegistry>,
    model_ids: Vec<String>,
    input_key: String,
    output_prefix: String,
}

impl ParallelInferenceStep {
    /// Create a new parallel inference step
    pub fn new(
        name: impl Into<String>,
        registry: Arc<ModelRegistry>,
        model_ids: Vec<String>,
        input_key: impl Into<String>,
        output_prefix: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            registry,
            model_ids,
            input_key: input_key.into(),
            output_prefix: output_prefix.into(),
        }
    }
}

impl PipelineStep for ParallelInferenceStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get input from context
        let input = ctx.get(&self.input_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Input key '{}' not found in context",
                self.input_key
            ))
        })?;

        // Convert to ModelInput if it's a DataFrame
        let model_input = match input {
            ContextData::ModelInput(input) => input.clone(),
            ContextData::DataFrame(df) => ModelInput(df.clone()),
            _ => {
                return Err(infer_lib::InferError::InvalidInput(format!(
                    "Input key '{}' is not a valid model input",
                    self.input_key
                )))
            }
        };

        // Run inference on all models in parallel
        let model_id_refs: Vec<&str> = self.model_ids.iter().map(|s| s.as_str()).collect();
        let results = self
            .registry
            .predict_broadcast(&model_id_refs, &model_input);

        // Store outputs in context
        for (model_id, result) in results {
            let output = result?;
            let output_key = format!("{}_{}", self.output_prefix, model_id);
            ctx.insert(output_key, ContextData::ModelOutput(output));
        }

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Runs inference on multiple models in parallel")
    }
}

/// Step that ensembles predictions from multiple models
pub struct EnsembleStep {
    name: String,
    input_keys: Vec<String>,
    output_key: String,
    method: EnsembleMethod,
}

/// Ensemble methods
#[derive(Debug, Clone, Copy)]
pub enum EnsembleMethod {
    /// Average predictions
    Average,
    /// Weighted average (weights must match number of models)
    WeightedAverage,
    /// Take maximum prediction
    Max,
    /// Take minimum prediction
    Min,
}

impl EnsembleStep {
    /// Create a new ensemble step
    pub fn new(
        name: impl Into<String>,
        input_keys: Vec<String>,
        output_key: impl Into<String>,
        method: EnsembleMethod,
    ) -> Self {
        Self {
            name: name.into(),
            input_keys,
            output_key: output_key.into(),
            method,
        }
    }
}

impl PipelineStep for EnsembleStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        use crate::ModelOutput;

        // Collect all predictions
        let mut all_predictions = Vec::new();

        for key in &self.input_keys {
            let data = ctx.get(key).ok_or_else(|| {
                infer_lib::InferError::InvalidInput(format!(
                    "Input key '{}' not found in context",
                    key
                ))
            })?;

            let output = data.as_model_output().ok_or_else(|| {
                infer_lib::InferError::InvalidInput(format!(
                    "Input key '{}' is not a ModelOutput",
                    key
                ))
            })?;

            let predictions = match output {
                ModelOutput::Single(preds) => preds.clone(),
                ModelOutput::Multi { .. } => {
                    return Err(infer_lib::InferError::InvalidInput(
                        "Ensemble does not support multi-output models yet".to_string(),
                    ))
                }
                _ => {
                    return Err(infer_lib::InferError::InvalidInput(
                        "Ensemble only supports Single and Multi numeric outputs".to_string(),
                    ))
                }
            };

            all_predictions.push(predictions);
        }

        // Ensure all predictions have the same length
        let num_rows = all_predictions[0].len();
        for preds in &all_predictions {
            if preds.len() != num_rows {
                return Err(infer_lib::InferError::InvalidInput(
                    "All predictions must have the same number of rows".to_string(),
                ));
            }
        }

        // Compute ensemble
        let ensembled = match self.method {
            EnsembleMethod::Average => {
                let mut result = vec![0.0; num_rows];
                for preds in &all_predictions {
                    for (i, &pred) in preds.iter().enumerate() {
                        result[i] += pred;
                    }
                }
                for val in &mut result {
                    *val /= all_predictions.len() as f64;
                }
                result
            }
            EnsembleMethod::Max => {
                let mut result = all_predictions[0].clone();
                for preds in &all_predictions[1..] {
                    for (i, &pred) in preds.iter().enumerate() {
                        result[i] = result[i].max(pred);
                    }
                }
                result
            }
            EnsembleMethod::Min => {
                let mut result = all_predictions[0].clone();
                for preds in &all_predictions[1..] {
                    for (i, &pred) in preds.iter().enumerate() {
                        result[i] = result[i].min(pred);
                    }
                }
                result
            }
            EnsembleMethod::WeightedAverage => {
                // For now, just use average. Weights can be added later
                let mut result = vec![0.0; num_rows];
                for preds in &all_predictions {
                    for (i, &pred) in preds.iter().enumerate() {
                        result[i] += pred;
                    }
                }
                for val in &mut result {
                    *val /= all_predictions.len() as f64;
                }
                result
            }
        };

        // Store result
        ctx.insert(
            &self.output_key,
            ContextData::ModelOutput(ModelOutput::Single(ensembled)),
        );

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Ensembles predictions from multiple models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ExecutionContext;
    use crate::{Model, ModelBackend, ModelInput, ModelOutput, ModelRegistry};

    struct MockModel {
        id: String,
        prediction: Vec<f64>,
    }

    impl Model for MockModel {
        fn id(&self) -> &str {
            &self.id
        }

        fn backend(&self) -> ModelBackend {
            ModelBackend::Perpetual
        }

        fn predict(&self, _input: &ModelInput) -> Result<ModelOutput> {
            Ok(ModelOutput::Single(self.prediction.clone()))
        }

        fn num_features(&self) -> Result<usize> {
            Ok(2)
        }
    }

    #[test]
    fn test_inference_step() {
        let registry = Arc::new(ModelRegistry::new());
        registry.register(
            "model1".to_string(),
            MockModel {
                id: "model1".to_string(),
                prediction: vec![1.0, 2.0],
            },
        );

        let step = InferenceStep::new(
            "inference",
            Arc::clone(&registry),
            "model1",
            "input",
            "output",
        );

        let mut ctx = ExecutionContext::new();
        ctx.insert(
            "input",
            ContextData::ModelInput(ModelInput::DenseF32 {
                data: vec![1.0, 2.0],
                num_rows: 1,
                num_features: 2,
            }),
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx.get("output").unwrap().as_model_output().unwrap();
        match output {
            ModelOutput::Single(preds) => assert_eq!(preds, &vec![1.0, 2.0]),
            _ => panic!("Expected Single output"),
        }
    }

    #[test]
    fn test_ensemble_step() {
        let mut ctx = ExecutionContext::new();
        ctx.insert(
            "pred1",
            ContextData::ModelOutput(ModelOutput::Single(vec![1.0, 2.0])),
        );
        ctx.insert(
            "pred2",
            ContextData::ModelOutput(ModelOutput::Single(vec![3.0, 4.0])),
        );

        let step = EnsembleStep::new(
            "ensemble",
            vec!["pred1".to_string(), "pred2".to_string()],
            "ensemble_output",
            EnsembleMethod::Average,
        );

        step.execute(&mut ctx).unwrap();

        let output = ctx
            .get("ensemble_output")
            .unwrap()
            .as_model_output()
            .unwrap();
        match output {
            ModelOutput::Single(preds) => {
                assert_eq!(preds.len(), 2);
                assert!((preds[0] - 2.0).abs() < 0.001); // (1.0 + 3.0) / 2
                assert!((preds[1] - 3.0).abs() < 0.001); // (2.0 + 4.0) / 2
            }
            _ => panic!("Expected Single output"),
        }
    }
}
