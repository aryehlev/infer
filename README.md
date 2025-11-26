# Infer-Lib

A high-performance ML inference library for Rust with an integrated execution engine for building ML pipelines and RAG (Retrieval-Augmented Generation) workflows. Supports multiple gradient boosting frameworks (CatBoost, XGBoost, LightGBM, Perpetual) with first-class Polars integration and parallel execution capabilities.

## Features

- 🚀 **Multiple ML Backends**: Unified interface for CatBoost, XGBoost, LightGBM, and Perpetual
- 🔀 **Execution Engine**: Build complex ML pipelines with conditional branching and parallel execution
- 📚 **RAG Support**: Built-in pipeline steps for retrieval-augmented generation workflows
- 🔒 **Lock-Free Model Registry**: Uses `ArcSwap` for atomic model updates without blocking inference
- ⚡ **Parallel Inference**: Run inference on multiple models simultaneously using Rayon
- 🐻‍❄️ **Polars Integration**: Native support for Polars DataFrames with zero-copy Arrow transfer
- 🛡️ **Type-Safe**: Strongly typed API with comprehensive error handling
- 🔄 **Zero-Downtime Updates**: Swap models atomically while serving predictions
- 🔧 **DataFrame Transformers**: Apply transformations to DataFrames before inference

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
infer-lib = { path = "../infer-lib" }
```

Or with specific features:

```toml
[dependencies]
infer-lib = { path = "../infer-lib", features = ["catboost", "xgboost"] }
```

Available features:
- `catboost` - Enable CatBoost support
- `xgboost` - Enable XGBoost support
- `lightgbm` - Enable LightGBM support
- `perpetual` - Enable Perpetual support
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
    let input = ModelInput::DenseF32 {
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
use infer_lib::prelude::*;
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

### DataFrame Transformers

Infer-Lib supports two approaches for data transformations:

**1. Model-Level Transformers** (for required preprocessing):
```rust
// Transformations that are ALWAYS needed for this model
let model = XGBoostModel::load("model", "model.json")?
    .with_transformer(Arc::new(StandardScaler))  // Required normalization
    .with_transformer(Arc::new(FeatureEncoder)); // Required encoding

registry.register("model", model);

// Transformations are automatically applied on every prediction
let output = registry.predict("model", &input)?;
```

**When to use:** Required preprocessing that the model was trained with (e.g., specific normalization, encoding). These transformations are always applied.

**2. Pipeline-Level Transformers** (for flexible workflows):
```rust
// Flexible transformation pipeline
let pipeline = Pipeline::builder("workflow")
    .add_step(Arc::new(TransformStep::new(
        "scale",
        Arc::new(StandardScaler),
        "raw_data",
        "scaled_data",
    )))
    .add_step(Arc::new(InferenceStep::new(
        "predict",
        registry,
        "model",
        "scaled_data",
        "predictions",
    )))
    .build();

let mut ctx = ExecutionContext::new();
ctx.insert("raw_data", ContextData::DataFrame(df));
pipeline.execute(&mut ctx)?;
```

**When to use:** Experimental preprocessing, A/B testing different transformations, composable workflows, or when different use cases need different preprocessing for the same model.

**Best Practice:** Use model-level transformers for required preprocessing, and pipeline-level transformers for flexible composition.

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

### Execution Engine & Pipelines

Build complex ML workflows using the integrated execution engine:

```rust
use infer_lib::prelude::*;
use infer_lib::steps::*;
use std::sync::Arc;

fn main() -> Result<()> {
    // Set up model registry
    let registry = Arc::new(ModelRegistry::new());
    // ... register models ...

    // Create a pipeline with multiple steps
    let pipeline = Pipeline::builder("ml_workflow")
        .add_fn("load_data", |ctx| {
            let df = df! {
                "feature1" => [1.0f32, 2.0, 3.0],
                "feature2" => [4.0f32, 5.0, 6.0],
            }?;
            ctx.insert("data", ContextData::DataFrame(df));
            Ok(StepResult::Continue)
        })
        .add_step(Arc::new(InferenceStep::new(
            "predict",
            Arc::clone(&registry),
            "model1",
            "data",
            "predictions",
        )))
        .add_step(Arc::new(EnsembleStep::new(
            "ensemble",
            vec!["pred1".to_string(), "pred2".to_string()],
            "final_predictions",
            EnsembleMethod::Average,
        )))
        .build();

    // Execute the pipeline
    let mut ctx = ExecutionContext::new();
    pipeline.execute(&mut ctx)?;

    Ok(())
}
```

**Built-in Pipeline Steps:**
- **InferenceStep**: Run model inference
- **ParallelInferenceStep**: Run multiple models in parallel
- **EnsembleStep**: Combine predictions from multiple models
- **TransformStep**: Apply DataFrame transformations
- **FilterStep**: Filter DataFrames based on conditions
- **SelectColumnsStep**: Select specific columns
- **AddPredictionColumnStep**: Add predictions to DataFrame

**Control Flow:**
- **Conditional branching**: Use `StepResult::Skip(target)` to jump to specific steps
- **Early stopping**: Use `StepResult::Stop` to halt execution
- **Parallel execution**: Use `ParallelStep` for concurrent operations

### RAG (Retrieval-Augmented Generation) Support

Built-in pipeline steps for RAG workflows:

```rust
use infer_lib::steps::*;

// Create a RAG pipeline
let rag_pipeline = Pipeline::builder("rag_workflow")
    .add_step(Arc::new(RetrievalStep::new(
        "retrieve",
        retriever,
        "query",
        "documents",
        top_k: 5,
    )))
    .add_step(Arc::new(RankingStep::new(
        "rerank",
        ranker,
        "query",
        "documents",
        "ranked_docs",
    )))
    .add_step(Arc::new(ScoreFilterStep::new(
        "filter",
        "ranked_docs",
        "filtered_docs",
        threshold: 0.7,
    )))
    .add_step(Arc::new(PromptConstructionStep::new(
        "construct_prompt",
        "query",
        "filtered_docs",
        "prompt",
        "Query: {query}\n\nContext:\n{documents}",
    )))
    .build();
```

**RAG Pipeline Steps:**
- **RetrievalStep**: Retrieve documents based on query
- **RankingStep**: Rank/rerank documents
- **ScoreFilterStep**: Filter documents by relevance score
- **PromptConstructionStep**: Build prompts from query and documents

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

| Backend | Thread Safety | Multi-output | Polars Support | Notes |
|---------|--------------|--------------|----------------|-------|
| CatBoost | ✅ Native | ✅ | ✅ Zero-copy | Native Arrow transfer |
| XGBoost | ✅ v1.4+ | ✅ | ✅ Zero-copy | Native Arrow transfer |
| LightGBM | ✅ Mutex | ✅ | ✅ Zero-copy | Native Arrow transfer |
| Perpetual | ✅ Native | ❌ | ✅ Via conversion | Pure Rust, DataFrame→Matrix |

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

# Pipeline and execution engine
cargo run --example pipeline_example
```

## Performance Considerations

1. **Model Loading**: Load models once and reuse them via the registry
2. **Parallel Execution**: Uses Rayon for work-stealing parallelism
3. **Memory**: Models are reference-counted (Arc), so cloning is cheap
4. **Lock-Free Reads**: Registry uses ArcSwap for lock-free model access during inference

## Future Plans

- [x] Add support for `perpetual` library integration
- [x] DataFrame transformers for preprocessing
- [x] Execution engine for building ML pipelines
- [x] RAG (Retrieval-Augmented Generation) workflow support
- [ ] Support for ONNX models
- [ ] Batch prediction optimization
- [ ] Model versioning and rollback
- [ ] Metrics and monitoring integration
- [ ] Async inference API
- [ ] Built-in vector database integration for RAG
- [ ] Advanced ensemble methods (stacking, boosting)
- [ ] Pipeline serialization/deserialization

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
- [perpetual](https://github.com/perpetual-ml/perpetual)
- [polars](https://github.com/pola-rs/polars)
- [arc-swap](https://github.com/vorner/arc-swap)
- [rayon](https://github.com/rayon-rs/rayon)
