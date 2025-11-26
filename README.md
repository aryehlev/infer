# Infer

A high-performance ML inference library for Rust that supports multiple gradient boosting frameworks (CatBoost, XGBoost, LightGBM) with first-class Polars integration and parallel execution capabilities.

## Features

- 🚀 **Multiple ML Backends**: Unified interface for CatBoost, XGBoost, and LightGBM
- 🔒 **Lock-Free Model Registry**: Uses `ArcSwap` for atomic model updates without blocking inference
- ⚡ **Parallel Inference**: Run inference on multiple models simultaneously using Rayon
- 🐻‍❄️ **Polars Integration**: Native support for Polars DataFrames as input
- 🛡️ **Type-Safe**: Strongly typed API with comprehensive error handling
- 🔄 **Zero-Downtime Updates**: Swap models atomically while serving predictions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
infer = { path = "../infer" }
```

Or with specific features:

```toml
[dependencies]
infer = { path = "../infer", features = ["catboost", "xgboost"] }
```

Available features:
- `catboost` - Enable CatBoost support
- `xgboost` - Enable XGBoost support
- `lightgbm` - Enable LightGBM support
- `default` - Enables all backends

## Quick Start

### Basic Usage

```rust
use infer::prelude::*;

fn main() -> Result<()> {
    // Create a model registry
    let registry = ModelRegistry::new();

    // Load and register models
    let xgb_model = XGBoostModel::load("xgb".to_string(), "model.json")?;
    registry.register("xgb".to_string(), xgb_model);

    // Prepare input data
    let input = ModelInput::Dense {
        data: vec![1.0, 2.0, 3.0, 4.0],
        num_rows: 2,
        num_features: 2,
    };

    // Run inference
    let output = registry.predict("xgb", &input)?;
    println!("Predictions: {:?}", output.as_slice());

    Ok(())
}
```

### Polars Integration

```rust
use infer::prelude::*;
use polars::prelude::*;

fn main() -> Result<()> {
    // Create a DataFrame
    let df = df! {
        "feature1" => [1.0f32, 2.0, 3.0],
        "feature2" => [4.0f32, 5.0, 6.0],
    }?;

    // Convert to model input
    let input = df.to_model_input()?;

    // Run inference (assuming a model is registered)
    let registry = ModelRegistry::new();
    // ... register models ...
    let output = registry.predict("my_model", &input)?;

    // Convert predictions back to Polars Series
    let predictions = output_to_series(&output, "predictions")?;

    Ok(())
}
```

### Parallel Inference

```rust
use infer::prelude::*;

fn main() -> Result<()> {
    let registry = ModelRegistry::new();
    // ... register models ...

    // Run multiple models in parallel with different inputs
    let model_ids = vec!["model1", "model2", "model3"];
    let inputs = vec![input1, input2, input3];

    let results = registry.predict_many(&model_ids, &inputs);
    for (model_id, result) in results {
        println!("{}: {:?}", model_id, result);
    }

    // Broadcast same input to multiple models (useful for A/B testing)
    let results = registry.predict_broadcast(&model_ids, &input);

    Ok(())
}
```

## Architecture

### Model Registry (ArcSwap-based)

The `ModelRegistry` uses `ArcSwap` to provide:
- **Lock-free reads**: Inference operations don't block each other
- **Atomic updates**: Models can be swapped without downtime
- **Thread-safe**: Safe to share across threads

```rust
// Register/update models without blocking inference
registry.register("model_v2".to_string(), new_model);

// Get model (returns Arc clone, very cheap)
let model = registry.get("model_v2")?;

// Parallel execution across models
registry.predict_many(&model_ids, &inputs);
```

### Supported Backends

| Backend | Thread Safety | Multi-output | Notes |
|---------|--------------|--------------|-------|
| CatBoost | ✅ Native | ✅ | Best thread safety |
| XGBoost | ✅ v1.4+ | ✅ | Thread-safe predictions for tree models |
| LightGBM | ✅ Mutex | ✅ | Mutex wrapper for safety |

### ModelInput Types

```rust
pub enum ModelInput {
    // Dense float matrix (row-major)
    Dense {
        data: Vec<f32>,
        num_rows: usize,
        num_features: usize,
    },

    // Polars DataFrame (auto-converted to Dense)
    DataFrame(DataFrame),
}
```

### ModelOutput Types

```rust
pub enum ModelOutput {
    // Single prediction per row (regression/binary)
    Single(Vec<f64>),

    // Multiple predictions per row (multiclass)
    Multi {
        data: Vec<f64>,
        num_classes: usize,
    },
}
```

## Examples

Run the examples:

```bash
# Basic usage
cargo run --example basic_usage

# Parallel inference patterns
cargo run --example parallel_inference

# Polars integration
cargo run --example polars_integration
```

## Performance Considerations

1. **Model Loading**: Load models once and reuse them via the registry
2. **Parallel Execution**: Uses Rayon for work-stealing parallelism
3. **Memory**: Models are reference-counted (Arc), so cloning is cheap
4. **Lock-Free Reads**: Registry uses ArcSwap for lock-free model access during inference

## Future Plans

- [ ] Add support for `perpetual` library integration
- [ ] Support for ONNX models
- [ ] Batch prediction optimization
- [ ] Model versioning and rollback
- [ ] Metrics and monitoring integration
- [ ] Async inference API

## Contributing

This library is designed to be extensible. To add a new backend:

1. Implement the `Model` trait for your backend
2. Add backend-specific module in `src/backends/`
3. Add feature flag in `Cargo.toml`
4. Update documentation

## License

Apache-2.0

## Credits

Built on top of:
- [catboost-rust](https://github.com/aryehlev/catboost-rust)
- [xgboost-rust](https://github.com/aryehlev/xgboost-rust)
- [lightgbm-rust](https://github.com/aryehlev/lightgbm-rust)
- [polars](https://github.com/pola-rs/polars)
- [arc-swap](https://github.com/vorner/arc-swap)
- [rayon](https://github.com/rayon-rs/rayon)
