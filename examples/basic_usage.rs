/// Basic usage example showing how to load models and run inference

use infer::prelude::*;

fn main() -> Result<()> {
    println!("=== Infer Library: Basic Usage Example ===\n");

    // Create a model registry
    let registry = ModelRegistry::new();
    println!("✓ Created model registry");

    // Example 1: Load and register an XGBoost model
    #[cfg(feature = "xgboost")]
    {
        println!("\n--- XGBoost Example ---");

        // Load model from file (replace with your actual model path)
        // let model = XGBoostModel::load("xgb_model".to_string(), "path/to/model.json")?;
        // registry.register("xgb_model".to_string(), model);
        // println!("✓ Registered XGBoost model");

        // Create sample input data
        let input = ModelInput::Dense {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            num_rows: 2,
            num_features: 3,
        };

        // Run inference
        // let output = registry.predict("xgb_model", &input)?;
        // println!("Predictions: {:?}", output.as_slice());

        println!("(Skipped - no model file provided)");
    }

    // Example 2: Load and register a CatBoost model
    #[cfg(feature = "catboost")]
    {
        println!("\n--- CatBoost Example ---");

        // Load model from file (replace with your actual model path)
        // let model = CatBoostModel::load("cb_model".to_string(), "path/to/model.cbm")?;
        // registry.register("cb_model".to_string(), model);
        // println!("✓ Registered CatBoost model");

        println!("(Skipped - no model file provided)");
    }

    // Example 3: Load and register a LightGBM model
    #[cfg(feature = "lightgbm")]
    {
        println!("\n--- LightGBM Example ---");

        // Load model from file (replace with your actual model path)
        // let model = LightGBMModel::load("lgbm_model".to_string(), "path/to/model.txt")?;
        // registry.register("lgbm_model".to_string(), model);
        // println!("✓ Registered LightGBM model");

        println!("(Skipped - no model file provided)");
    }

    // Show registry status
    println!("\n--- Registry Status ---");
    println!("Number of models registered: {}", registry.len());
    println!("Model IDs: {:?}", registry.list_ids());

    println!("\n✓ Basic usage example completed");
    println!("\nTo run with actual models:");
    println!("1. Train a model using your preferred framework");
    println!("2. Update the paths in this example");
    println!("3. Run: cargo run --example basic_usage");

    Ok(())
}
