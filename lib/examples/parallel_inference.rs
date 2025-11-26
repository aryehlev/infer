/// Example showing parallel inference on multiple models

use infer_lib::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Infer Library: Parallel Inference Example ===\n");

    let registry = ModelRegistry::new();

    // In a real scenario, you would load actual models here
    // For this example, we'll show the API structure

    println!("--- Parallel Inference API ---\n");

    // Example 1: Different inputs for different models
    {
        use polars::prelude::*;

        println!("1. predict_many: Different inputs for different models");
        println!("   Useful for batch processing where each model gets its own input\n");

        // Create sample DataFrames
        // let df1 = df! { "a" => [1.0f32], "b" => [2.0f32] }.unwrap();
        // let df2 = df! { "a" => [3.0f32], "b" => [4.0f32] }.unwrap();
        // let df3 = df! { "a" => [5.0f32], "b" => [6.0f32] }.unwrap();
        //
        // let model_ids = vec!["model1", "model2", "model3"];
        // let inputs = vec![
        //     ModelInput(df1),
        //     ModelInput(df2),
        //     ModelInput(df3),
        // ];

        // This would run inference in parallel using rayon
        // let results = registry.predict_many(&model_ids, &inputs);
        //
        // for (model_id, result) in results {
        //     match result {
        //         Ok(output) => println!("  {} -> {:?}", model_id, output.as_slice()),
        //         Err(e) => println!("  {} -> Error: {}", model_id, e),
        //     }
        // }

        println!("  API signature:");
        println!("  fn predict_many(&self, model_ids: &[&str], inputs: &[ModelInput])");
        println!("    -> Vec<(String, Result<ModelOutput>)>");
    }

    println!();

    // Example 2: Same input broadcast to multiple models
    {
        use polars::prelude::*;

        println!("2. predict_broadcast: Same input to multiple models");
        println!("   Useful for A/B testing or model ensembling\n");

        // Create sample DataFrame
        // let df = df! {
        //     "feature1" => [1.0f32, 2.0],
        //     "feature2" => [3.0f32, 4.0],
        // }.unwrap();
        //
        // let model_ids = vec!["modelA", "modelB", "modelC"];
        // let input = ModelInput(df);

        // This would broadcast the same input to all models in parallel
        // let results = registry.predict_broadcast(&model_ids, &input);
        //
        // for (model_id, result) in results {
        //     match result {
        //         Ok(output) => println!("  {} -> {:?}", model_id, output.as_slice()),
        //         Err(e) => println!("  {} -> Error: {}", model_id, e),
        //     }
        // }

        println!("  API signature:");
        println!("  fn predict_broadcast(&self, model_ids: &[&str], input: &ModelInput)");
        println!("    -> Vec<(String, Result<ModelOutput>)>");
    }

    println!();

    // Example 3: Performance comparison
    {
        println!("3. Performance Benefits");
        println!("   Parallel execution uses rayon for work-stealing parallelism\n");

        println!("  Sequential execution:");
        println!("    Time = T1 + T2 + T3 + ... + Tn");
        println!();
        println!("  Parallel execution (with N cores):");
        println!("    Time ≈ max(T1, T2, ..., Tn) + overhead");
        println!();
        println!("  With ArcSwap registry:");
        println!("    - Zero-cost cloning of model Arc references");
        println!("    - No locks during inference (only during registration)");
        println!("    - Models can be updated without stopping inference");
    }

    println!("\n--- Thread Safety ---");
    println!("✓ All model backends implement Send + Sync");
    println!("✓ CatBoost: Native thread-safe");
    println!("✓ XGBoost: Thread-safe for v1.4+");
    println!("✓ LightGBM: Uses Mutex wrapper for safety");

    println!("\n✓ Parallel inference example completed");

    Ok(())
}
