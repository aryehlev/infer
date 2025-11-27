# Infer-Lib

A high-performance ML inference library for Rust with native Polars DataFrame support. Provides a unified interface for multiple gradient boosting frameworks (CatBoost, XGBoost, LightGBM, Perpetual) with parallel execution and zero-downtime model updates.

[![Build Status](https://github.com/aryehlev/infer-lib/workflows/CI/badge.svg)](https://github.com/aryehlev/infer-lib/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Features

- 🚀 **Multiple ML Backends**: Unified interface for CatBoost, XGBoost, LightGBM, Perpetual, ONNX, and Candle
- 🐻‍❄️ **Polars-Native**: First-class Polars DataFrame support with zero-copy Arrow transfer
- 🔒 **Lock-Free Registry**: Uses `ArcSwap` for atomic model updates without blocking inference
- ⚡ **Parallel Inference**: Run inference on multiple models simultaneously using Rayon
- 🛡️ **Type-Safe**: Strongly typed API with comprehensive error handling
- 🔄 **Zero-Downtime Updates**: Swap models atomically while serving predictions
- 🔧 **DataFrame Transformers**: Apply preprocessing transformations before inference
- 📦 **Execution Engine**: Build complex ML pipelines with the integrated pipeline framework

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
infer-lib = { git = "https://github.com/aryehlev/infer-lib" }
```

Or with specific features:

```toml
[dependencies]
infer-lib = { git = "https://github.com/aryehlev/infer-lib", features = ["catboost", "xgboost"] }
```

**Available features:**
- `catboost` - CatBoost support with native Polars integration
- `xgboost` - XGBoost support with native Polars integration
- `lightgbm` - LightGBM support with native Polars integration
- `perpetual` - Perpetual gradient boosting (pure Rust)
- `onnx` - ONNX Runtime support for deep learning models
- `candle` - Candle support for transformers and LLMs

## Quick Start

### Basic Usage

```rust
use infer_lib::prelude::*;
use polars::prelude::*;

fn main() -> Result<()> {
    // Create a model registry
    let registry = ModelRegistry::new();

    // Load and register a model
    let model = XGBoostModel::load("xgb", "model.json")?;
    registry.register("xgb", model);

    // Create input DataFrame
    let df = df! {
        "feature1" => [1.0f32, 2.0, 3.0],
        "feature2" => [4.0f32, 5.0, 6.0],
    }?;

    // Run inference (Polars-native!)
    let input = ModelInput(df);
    let output = registry.predict("xgb", &input)?;

    println!("Predictions: {:?}", output);
    Ok(())
}
```

### Polars Integration

Infer-Lib uses Polars DataFrames natively. All backends that support it use **zero-copy Arrow transfer** for maximum performance.

```rust
use infer_lib::prelude::*;
use infer_lib::polars_ext::output_to_series;
use polars::prelude::*;

fn main() -> Result<()> {
    // Create a DataFrame
    let df = df! {
        "sepal_length" => [5.1f32, 4.9, 4.7],
        "sepal_width" => [3.5f32, 3.0, 3.2],
        "petal_length" => [1.4f32, 1.4, 1.3],
        "petal_width" => [0.2f32, 0.2, 0.2],
    }?;

    // Wrap in ModelInput
    let input = ModelInput(df);

    // Run inference
    let registry = ModelRegistry::new();
    // ... register your model ...
    let output = registry.predict("iris_model", &input)?;

    // Convert predictions back to Polars Series
    let predictions = output_to_series(&output, "predictions")?;

    // Add predictions to your DataFrame
    let mut result_df = input.0.clone();
    result_df.with_column(predictions)?;

    println!("{}", result_df);
    Ok(())
}
```

### DataFrame Transformers

Apply preprocessing transformations at the model level:

```rust
use infer_lib::prelude::*;
use std::sync::Arc;

// Define a custom transformer
struct StandardScaler;

impl DataFrameTransformer for StandardScaler {
    fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // Apply standardization to numeric columns
        // ... transformation logic ...
        Ok(df)
    }

    fn name(&self) -> &str {
        "StandardScaler"
    }
}

// Attach transformers to models
let model = XGBoostModel::load("model", "model.json")?
    .with_transformer(Arc::new(StandardScaler));

registry.register("model", model);

// Transformations are automatically applied on every prediction
let output = registry.predict("model", &input)?;
```

**When to use model-level transformers:**
- Required preprocessing that the model was trained with
- Normalization/standardization that's always needed
- Feature engineering specific to this model

### Parallel Inference

Run multiple models in parallel for A/B testing or ensembles:

```rust
use infer_lib::prelude::*;

fn main() -> Result<()> {
    let registry = ModelRegistry::new();

    // Register multiple model versions
    registry.register("model_v1", model_v1);
    registry.register("model_v2", model_v2);
    registry.register("model_v3", model_v3);

    // Broadcast same input to multiple models (A/B testing)
    let model_ids = vec!["model_v1", "model_v2", "model_v3"];
    let results = registry.predict_broadcast(&model_ids, &input);

    for (model_id, result) in results {
        match result {
            Ok(output) => println!("{}: {:?}", model_id, output),
            Err(e) => eprintln!("{}: Error - {}", model_id, e),
        }
    }

    Ok(())
}
```

## Architecture

### Polars-Native Design

All backends use Polars DataFrames as the primary input format:

```rust
pub struct ModelInput(pub DataFrame);

pub enum ModelOutput {
    Single(Vec<f64>),           // Regression/binary classification
    Multi {                      // Multiclass probabilities
        data: Vec<f64>,
        num_classes: usize
    },
    Text(Vec<String>),          // Text generation
    Embeddings {                // Vector embeddings
        data: Vec<f32>,
        embedding_dim: usize
    },
    // ... and more output types
}
```

### Backend Support

| Backend | Thread Safety | Polars Support | Notes |
|---------|--------------|----------------|-------|
| **CatBoost** | ✅ Native | ✅ Zero-copy Arrow | Native Polars integration |
| **XGBoost** | ✅ v1.4+ | ✅ Zero-copy Arrow | Native Polars integration |
| **LightGBM** | ✅ Mutex | ✅ Zero-copy Arrow | Native Polars integration |
| **Perpetual** | ✅ Native | ✅ Conversion | Pure Rust, DataFrame→ndarray |
| **ONNX** | ✅ Native | 🚧 Conversion | Via dense f32 conversion |
| **Candle** | ✅ Native | 🚧 Conversion | Via dense f32 conversion |

### Model Registry (Lock-Free)

The `ModelRegistry` uses `ArcSwap` for lock-free, thread-safe operations:

```rust
// ✅ Lock-free reads - inference doesn't block
let model = registry.get("model_v2")?;

// ✅ Atomic updates - swap models without downtime
registry.register("model_v2", new_model);

// ✅ Thread-safe - safe to share across threads
let registry = Arc::new(ModelRegistry::new());
```

**Benefits:**
- **Lock-free reads**: Predictions never block each other
- **Atomic updates**: Model swaps are instant and safe
- **Zero downtime**: Update models while serving traffic
- **Thread-safe**: Share registry across threads

## Examples

Check out the `lib/examples/` directory:

```bash
# Basic usage with XGBoost
cargo run --example basic_usage --features xgboost

# Parallel inference patterns
cargo run --example parallel_inference --features "xgboost catboost"

# Polars DataFrame integration
cargo run --example polars_integration --features xgboost
```

## Performance

### Optimization Tips

1. **Use Polars-native backends** (CatBoost, XGBoost, LightGBM) for zero-copy Arrow transfer
2. **Load models once** and reuse via the registry
3. **Leverage parallel execution** for multiple models using `predict_broadcast`
4. **Enable LTO** in release builds for maximum performance

### Memory Efficiency

- Models are reference-counted (`Arc`) - cloning is cheap
- Lock-free reads mean no mutex contention
- Zero-copy Arrow transfer for Polars-native backends
- Efficient parallel execution with Rayon's work-stealing

### Polars Arrow Optimization

See [POLARS_ARROW_OPTIMIZATION.md](POLARS_ARROW_OPTIMIZATION.md) for detailed information about zero-copy Arrow transfers and performance benchmarks.

## Pipeline Framework

Build complex ML workflows with the execution engine (in the `pipeline` crate):

```rust
use infer_pipeline::prelude::*;
use std::sync::Arc;

let pipeline = Pipeline::builder("ml_workflow")
    .add_step(Arc::new(InferenceStep::new(
        "predict",
        Arc::clone(&registry),
        "model_v1",
        "input_data",
        "predictions",
    )))
    .build();

let mut ctx = ExecutionContext::new();
ctx.insert("input_data", ContextData::DataFrame(df));
pipeline.execute(&mut ctx)?;
```

**Pipeline features:**
- Conditional branching
- Parallel execution
- Built-in steps for inference, transforms, ensembles
- RAG workflow support

## Testing

Run the test suite:

```bash
# Run all tests
cargo test --workspace --all-features

# Run only lib tests
cargo test --package infer-lib --all-features

# Run with output
cargo test --workspace --all-features -- --nocapture
```

## Roadmap

- [x] Multiple gradient boosting backends
- [x] Polars DataFrame integration with zero-copy Arrow
- [x] Lock-free model registry with ArcSwap
- [x] Parallel inference with Rayon
- [x] DataFrame transformers
- [x] Pipeline execution engine
- [ ] ONNX Runtime full implementation
- [ ] Candle transformers integration
- [ ] Async inference API
- [ ] Model versioning and rollback
- [ ] Metrics and monitoring
- [ ] Built-in vector database for RAG
- [ ] Advanced ensemble methods

## Contributing

Contributions are welcome! To add a new backend:

1. Implement the `Model` trait in `lib/src/backends/`
2. Add feature flag in `lib/Cargo.toml`
3. Add tests in `lib/tests/`
4. Update documentation

## License

Licensed under Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Credits

Built with these excellent libraries:

- **ML Backends:**
  - [catboost-rust](https://github.com/aryehlev/catboost-rust) - CatBoost Rust bindings
  - [xgboost-rust](https://github.com/aryehlev/xgboost-rust) - XGBoost Rust bindings
  - [lightgbm-rust](https://github.com/aryehlev/lightgbm-rust) - LightGBM Rust bindings
  - [perpetual](https://github.com/perpetual-ml/perpetual) - Pure Rust gradient boosting
  - [ort](https://github.com/pykeio/ort) - ONNX Runtime bindings
  - [candle](https://github.com/huggingface/candle) - Minimalist ML framework

- **Core Infrastructure:**
  - [polars](https://github.com/pola-rs/polars) - Fast DataFrame library
  - [arc-swap](https://github.com/vorner/arc-swap) - Lock-free atomic Arc swapping
  - [rayon](https://github.com/rayon-rs/rayon) - Data parallelism

---

**Documentation:** [docs.rs/infer-lib](https://docs.rs/infer-lib) (coming soon)
**Repository:** [github.com/aryehlev/infer-lib](https://github.com/aryehlev/infer-lib)
