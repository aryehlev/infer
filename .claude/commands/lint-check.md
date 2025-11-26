---
description: Run clippy linter with strict checks on all targets
allowed-tools: Bash(cargo clippy:*), Bash(cargo fmt:*)
argument-hint: [--fix]
---

# Lint and Format Check

Run comprehensive linting checks across the entire workspace.

## Clippy Linting
Run clippy with all features and treat warnings as errors:
! cargo clippy --all-targets --all-features --workspace -- -D warnings $ARGUMENTS

## Format Check
Check code formatting:
! cargo fmt --all -- --check

Run `cargo fmt --all` to auto-format if needed.
