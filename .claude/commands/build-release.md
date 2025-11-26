---
description: Build optimized release binary with LTO and profiling
allowed-tools: Bash(cargo build:*), Bash(cargo check:*)
argument-hint: [--workspace] [--features]
---

# Build Release Binary

Build an optimized release binary for the ML inference project with full optimizations.

## Build Process
Build with all optimizations enabled (LTO, opt-level 3):
! cargo build --release $ARGUMENTS

## Verify Build
Check compilation across all features:
! cargo check --all-features --workspace

## Build Artifacts
The binaries and libraries are available at:
- `target/release/` - Main artifacts
- `target/release/deps/` - Dependencies

## Size Information
! ls -lh target/release/libinfer*.rlib target/release/libinfer_pipeline*.rlib 2>/dev/null || echo "Build artifacts ready"
