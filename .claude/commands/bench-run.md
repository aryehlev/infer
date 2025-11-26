---
description: Run benchmarks and performance tests
allowed-tools: Bash(cargo bench:*), Bash(cargo build:*)
argument-hint: [benchmark-name] [--save-baseline]
---

# Run Benchmarks

Execute performance benchmarks for the inference library.

## Build Benchmarks
First, build benchmarks in release mode:
! cargo bench --no-run --workspace

## Run Benchmarks
Execute all benchmarks:
! cargo bench --workspace $ARGUMENTS

## Performance Tips
- Ensure machine is idle during benchmarking
- Run multiple times and compare results
- Use `--save-baseline <name>` to save results for comparison
- Compare with `cargo bench --baseline <name>`

## View Results
Benchmark results are saved in `target/criterion/`
