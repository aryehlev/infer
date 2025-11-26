# Infer: High-Performance ML Inference Engine in Rust

## Project Overview
A high-performance ML inference library supporting multiple backends with Polars DataFrame integration:
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost, Perpetual
- **Deep Learning**: ONNX Runtime, Candle (transformers/LLMs)
- **DataFrame Integration**: Native Polars support with Arrow optimization
- **Parallel Inference**: Built on rayon for concurrent processing

## Project Structure
```
infer/
├── lib/                    # Core inference library
│   ├── src/
│   │   ├── backends/      # ML backend implementations
│   │   ├── error.rs       # Error types
│   │   ├── lib.rs         # Library entry point
│   │   ├── model.rs       # Core model abstractions
│   │   ├── polars_ext.rs  # Polars integration
│   │   └── registry.rs    # Model registry
│   ├── examples/          # Usage examples
│   └── tests/             # Integration tests
└── pipeline/              # Execution engine for ML pipelines
    ├── src/
    │   └── pipeline/      # Pipeline steps and execution
    └── examples/          # Pipeline examples
```

## Code Conventions
- **Error Handling**: Use `Result<T>` - no panics in library code
- **Documentation**: All public APIs must have doc comments
- **Performance**: Use `#[inline]` for hot path functions
- **Testing**: Integration tests in `tests/`, unit tests in modules
- **Formatting**: Run `cargo fmt` before committing
- **Linting**: Ensure `cargo clippy` passes with no warnings

## Build Commands
```bash
cargo build --release          # Production build (LTO enabled)
cargo test --all-features      # Full test suite
cargo bench                    # Run benchmarks
cargo doc --open               # Generate documentation
cargo clippy --all-targets     # Lint checks
```

## Key Files
- `lib/src/lib.rs` - Library entry point and feature flags
- `lib/src/model.rs` - Core ML model abstractions
- `lib/src/backends/` - Backend implementations (XGBoost, LightGBM, etc.)
- `lib/src/polars_ext.rs` - Polars DataFrame integration
- `lib/src/registry.rs` - Global model registry for loading/storing models
- `pipeline/src/pipeline/` - Pipeline execution engine

## Performance Considerations
- **Always profile with release builds** - Debug builds are 10-100x slower
- **Use `--profile=bench`** for accurate performance measurements
- **Check `POLARS_ARROW_OPTIMIZATION.md`** for memory optimization details
- **Parallel inference** uses rayon's global thread pool (configurable)
- **Arrow memory layout** ensures zero-copy operations with Polars

## Dependencies to Watch
- `polars` (0.45) - Major version changes can affect performance and API
- `arc-swap` (1.7) - For lock-free model swapping
- `rayon` (1.10) - Parallel processing backend
- Backend-specific: `catboost-rust`, `xgboost-rust`, `lightgbm-rust`
- Deep learning: `ort` (ONNX), `candle-*` (transformers)

## Development Workflow
1. **Feature Development**: Create feature branch from `main`
2. **Testing**: Run full test suite with `cargo test --all-features`
3. **Linting**: Ensure `cargo clippy` passes
4. **Documentation**: Update docs and run `cargo doc`
5. **Benchmarking**: Compare performance before/after changes
6. **Review**: Submit PR with comprehensive description

## Common Tasks
- **Add new backend**: Implement trait in `lib/src/backends/`
- **Add pipeline step**: Create module in `pipeline/src/pipeline/steps/`
- **Optimize performance**: Profile with `cargo flamegraph` or `perf`
- **Update dependencies**: Check compatibility with `cargo tree`

## Security Considerations
- Use `cargo deny check advisories` for security audits
- Review `unsafe` blocks carefully (search with grep)
- Validate all external inputs at API boundaries
- Keep ML backends up-to-date for security patches
