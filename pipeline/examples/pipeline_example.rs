/// Example showing how to use the execution engine for ML pipelines
use infer_pipeline::prelude::*;
use infer_pipeline::steps::*;
use polars::prelude::*;
use std::sync::Arc;

// Mock model for demonstration
struct DemoModel {
    id: String,
    multiplier: f64,
}

impl Model for DemoModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn backend(&self) -> ModelBackend {
        ModelBackend::Perpetual
    }

    fn predict(&self, input: &ModelInput) -> Result<ModelOutput> {
        let df = &input.0;
        let num_rows = df.height();

        // Sum all columns for each row and multiply by multiplier
        let predictions: Vec<f64> = (0..num_rows)
            .map(|row_idx| {
                let mut sum = 0.0f64;
                for col in df.get_columns() {
                    if let Ok(val) = col.get(row_idx) {
                        if let Ok(f) = val.try_extract::<f64>() {
                            sum += f;
                        } else if let Ok(f) = val.try_extract::<f32>() {
                            sum += f as f64;
                        }
                    }
                }
                sum * self.multiplier
            })
            .collect();

        Ok(ModelOutput::Single(predictions))
    }

    fn num_features(&self) -> Result<usize> {
        Ok(2)
    }
}

fn main() -> Result<()> {
    println!("=== Infer Library: Pipeline Execution Example ===\n");

    // 1. Set up model registry
    println!("Setting up model registry...");
    let registry = Arc::new(ModelRegistry::new());

    registry.register(
        "model_v1".to_string(),
        DemoModel {
            id: "model_v1".to_string(),
            multiplier: 1.0,
        },
    );

    registry.register(
        "model_v2".to_string(),
        DemoModel {
            id: "model_v2".to_string(),
            multiplier: 2.0,
        },
    );

    // 2. Create a simple linear pipeline
    println!("\n--- Example 1: Simple Linear Pipeline ---");
    simple_pipeline_example(Arc::clone(&registry))?;

    // 3. Create a conditional pipeline
    println!("\n--- Example 2: Conditional Pipeline ---");
    conditional_pipeline_example(Arc::clone(&registry))?;

    // 4. Create a parallel inference pipeline
    println!("\n--- Example 3: Parallel Inference Pipeline ---");
    parallel_inference_example(Arc::clone(&registry))?;

    // 5. Create a complete ML workflow pipeline
    println!("\n--- Example 4: Complete ML Workflow ---");
    complete_workflow_example(Arc::clone(&registry))?;

    println!("\n=== All pipeline examples completed successfully! ===");
    Ok(())
}

fn simple_pipeline_example(registry: Arc<ModelRegistry>) -> Result<()> {
    // Create a simple pipeline that loads data and runs inference
    let pipeline = Pipeline::builder("simple_pipeline")
        .add_fn("load_data", |ctx: &mut ExecutionContext| {
            println!("  Step 1: Loading data...");
            let df = df! {
                "feature1" => [1.0f32, 2.0, 3.0],
                "feature2" => [4.0f32, 5.0, 6.0],
            }?;
            ctx.insert("data", ContextData::DataFrame(df));
            Ok(StepResult::Continue)
        })
        .add_step(Arc::new(InferenceStep::new(
            "inference",
            registry,
            "model_v1",
            "data",
            "predictions",
        )))
        .add_fn("display_results", |ctx: &mut ExecutionContext| {
            println!("  Step 3: Displaying results...");
            let output = ctx.get("predictions").unwrap().as_model_output().unwrap();
            if let ModelOutput::Single(preds) = output {
                println!("  Predictions: {:?}", preds);
            }
            Ok(StepResult::Continue)
        })
        .build();

    let mut ctx = ExecutionContext::new();
    pipeline.execute(&mut ctx)?;

    Ok(())
}

fn conditional_pipeline_example(registry: Arc<ModelRegistry>) -> Result<()> {
    // Create a pipeline with conditional branching
    let pipeline = Pipeline::builder("conditional_pipeline")
        .add_fn("load_data", |ctx: &mut ExecutionContext| {
            println!("  Step 1: Loading data...");
            let df = df! {
                "feature1" => [10.0f32, 20.0, 30.0],
                "feature2" => [15.0f32, 25.0, 35.0],
            }?;
            ctx.insert("data", ContextData::DataFrame(df));
            ctx.insert("use_v2", ContextData::Bool(true));
            Ok(StepResult::Continue)
        })
        .add_fn("choose_model", |ctx: &mut ExecutionContext| {
            println!("  Step 2: Choosing model based on condition...");
            let use_v2 = ctx.get("use_v2").and_then(|d| d.as_bool()).unwrap_or(false);
            if use_v2 {
                println!("  -> Using model v2");
                Ok(StepResult::Skip("run_v2".to_string()))
            } else {
                println!("  -> Using model v1");
                Ok(StepResult::Continue)
            }
        })
        .add_step(Arc::new(InferenceStep::new(
            "run_v1",
            Arc::clone(&registry),
            "model_v1",
            "data",
            "predictions",
        )))
        .add_fn("skip_v2", |_ctx: &mut ExecutionContext| {
            Ok(StepResult::Skip("display".to_string()))
        })
        .add_step(Arc::new(InferenceStep::new(
            "run_v2",
            registry,
            "model_v2",
            "data",
            "predictions",
        )))
        .add_fn("display", |ctx: &mut ExecutionContext| {
            println!("  Final Step: Displaying results...");
            let output = ctx.get("predictions").unwrap().as_model_output().unwrap();
            if let ModelOutput::Single(preds) = output {
                println!("  Predictions: {:?}", preds);
            }
            Ok(StepResult::Continue)
        })
        .build();

    let mut ctx = ExecutionContext::new();
    pipeline.execute(&mut ctx)?;

    Ok(())
}

fn parallel_inference_example(registry: Arc<ModelRegistry>) -> Result<()> {
    // Run multiple models in parallel and ensemble the results
    let pipeline = Pipeline::builder("parallel_pipeline")
        .add_fn("load_data", |ctx: &mut ExecutionContext| {
            println!("  Step 1: Loading data...");
            let df = df! {
                "feature1" => [5.0f32, 10.0],
                "feature2" => [7.0f32, 13.0],
            }?;
            ctx.insert("data", ContextData::DataFrame(df));
            Ok(StepResult::Continue)
        })
        .add_step(Arc::new(ParallelInferenceStep::new(
            "parallel_inference",
            registry,
            vec!["model_v1".to_string(), "model_v2".to_string()],
            "data",
            "pred",
        )))
        .add_step(Arc::new(EnsembleStep::new(
            "ensemble",
            vec!["pred_model_v1".to_string(), "pred_model_v2".to_string()],
            "final_predictions",
            EnsembleMethod::Average,
        )))
        .add_fn("display_results", |ctx: &mut ExecutionContext| {
            println!("  Step 4: Displaying results...");
            println!(
                "  Model v1 predictions: {:?}",
                ctx.get("pred_model_v1").unwrap().as_model_output().unwrap()
            );
            println!(
                "  Model v2 predictions: {:?}",
                ctx.get("pred_model_v2").unwrap().as_model_output().unwrap()
            );
            println!(
                "  Ensemble predictions: {:?}",
                ctx.get("final_predictions")
                    .unwrap()
                    .as_model_output()
                    .unwrap()
            );
            Ok(StepResult::Continue)
        })
        .build();

    let mut ctx = ExecutionContext::new();
    pipeline.execute(&mut ctx)?;

    Ok(())
}

fn complete_workflow_example(registry: Arc<ModelRegistry>) -> Result<()> {
    // A complete workflow: load -> transform -> predict -> postprocess
    struct ScaleTransformer;
    impl DataFrameTransformer for ScaleTransformer {
        fn transform(&self, df: DataFrame) -> Result<DataFrame> {
            let result = df
                .lazy()
                .with_columns(vec![col("*").cast(DataType::Float64) / lit(10.0)])
                .collect()?;
            Ok(result)
        }
        fn name(&self) -> &str {
            "ScaleTransformer"
        }
    }

    let pipeline = Pipeline::builder("complete_workflow")
        .add_fn("load_data", |ctx: &mut ExecutionContext| {
            println!("  Step 1: Loading raw data...");
            let df = df! {
                "feature1" => [50.0f32, 100.0, 150.0],
                "feature2" => [70.0f32, 130.0, 190.0],
            }?;
            ctx.insert("raw_data", ContextData::DataFrame(df));
            Ok(StepResult::Continue)
        })
        .add_step(Arc::new(TransformStep::new(
            "scale",
            Arc::new(ScaleTransformer),
            "raw_data",
            "scaled_data",
        )))
        .add_step(Arc::new(InferenceStep::new(
            "predict",
            registry,
            "model_v1",
            "scaled_data",
            "predictions",
        )))
        .add_step(Arc::new(AddPredictionColumnStep::new(
            "add_predictions",
            "raw_data",
            "predictions",
            "final_data",
            "pred",
        )))
        .add_fn("display_final", |ctx: &mut ExecutionContext| {
            println!("  Step 5: Final results...");
            let df = ctx.get("final_data").unwrap().as_dataframe().unwrap();
            println!("{}", df);
            Ok(StepResult::Continue)
        })
        .build();

    let mut ctx = ExecutionContext::new();
    pipeline.execute(&mut ctx)?;

    Ok(())
}
