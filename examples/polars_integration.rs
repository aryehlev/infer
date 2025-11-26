/// Example showing Polars DataFrame integration

use infer::prelude::*;
use polars::prelude::*;

fn main() -> Result<()> {
    println!("=== Infer Library: Polars Integration Example ===\n");

    // Create a sample Polars DataFrame
    let df = create_sample_dataframe()?;
    println!("Created sample DataFrame:");
    println!("{}\n", df);

    // Example 1: Convert DataFrame to ModelInput
    {
        println!("--- Example 1: DataFrame to ModelInput ---");

        let input = df.to_model_input()?;

        if let ModelInput::Dense {
            num_rows,
            num_features,
            ..
        } = &input
        {
            println!("✓ Converted DataFrame to dense input");
            println!("  Shape: {} rows × {} features", num_rows, num_features);
        }
    }

    println!();

    // Example 2: Select specific columns
    {
        println!("--- Example 2: Select Specific Columns ---");

        let columns = ["feature1", "feature2"];
        let input = df.to_model_input_with_columns(&columns)?;

        if let ModelInput::Dense {
            num_rows,
            num_features,
            ..
        } = &input
        {
            println!("✓ Selected columns: {:?}", columns);
            println!("  Shape: {} rows × {} features", num_rows, num_features);
        }
    }

    println!();

    // Example 3: Full workflow with model inference
    {
        println!("--- Example 3: Full Workflow ---");
        println!("1. Load data into Polars DataFrame");
        println!("2. Preprocess/feature engineering in Polars");
        println!("3. Convert to ModelInput");
        println!("4. Run inference");
        println!("5. Convert predictions back to Polars Series");
        println!();

        // Simulate a prediction output
        let mock_output = ModelOutput::Single(vec![0.1, 0.8, 0.3, 0.9, 0.2]);

        // Convert back to Polars Series
        let predictions_series = output_to_series(&mock_output, "predictions")?;
        println!("✓ Predictions as Polars Series:");
        println!("{:?}", predictions_series);

        // Add predictions to the original DataFrame
        let mut df_with_preds = df.clone();
        df_with_preds.with_column(predictions_series)?;
        println!("\n✓ DataFrame with predictions:");
        println!("{}", df_with_preds);
    }

    println!();

    // Example 4: Working with multiclass predictions
    {
        println!("--- Example 4: Multiclass Predictions ---");

        // Simulate multiclass output (5 rows, 3 classes)
        let multiclass_output = ModelOutput::Multi {
            data: vec![
                0.7, 0.2, 0.1, // row 0
                0.1, 0.8, 0.1, // row 1
                0.2, 0.3, 0.5, // row 2
                0.6, 0.3, 0.1, // row 3
                0.1, 0.2, 0.7, // row 4
            ],
            num_classes: 3,
        };

        // Convert to Polars List column
        let predictions_series = output_to_series(&multiclass_output, "class_probabilities")?;
        println!("✓ Multiclass predictions as Polars List column:");
        println!("{:?}", predictions_series);
    }

    println!();

    // Example 5: Data type support
    {
        println!("--- Example 5: Supported Data Types ---");
        println!("The library automatically converts Polars dtypes to f32:");
        println!("  ✓ Float32, Float64");
        println!("  ✓ Int8, Int16, Int32, Int64");
        println!("  ✓ UInt8, UInt16, UInt32, UInt64");
        println!("  ✓ Boolean (true=1.0, false=0.0)");
        println!();
        println!("Unsupported types will return an error:");
        println!("  ✗ String, Categorical, Date, Datetime, List, Struct");
    }

    println!("\n✓ Polars integration example completed");

    Ok(())
}

/// Create a sample DataFrame for demonstration
fn create_sample_dataframe() -> Result<DataFrame> {
    let df = df! {
        "feature1" => [1.0f32, 2.0, 3.0, 4.0, 5.0],
        "feature2" => [0.5f32, 1.5, 2.5, 3.5, 4.5],
        "feature3" => [10, 20, 30, 40, 50],
        "feature4" => [true, false, true, false, true],
    }?;

    Ok(df)
}
