---
description: Run comprehensive test suite with all features and backends
allowed-tools: Bash(cargo test:*), Bash(cargo doc:*)
argument-hint: [--nocapture] [--test-threads]
---

# Comprehensive Testing

Run all tests across the workspace with all features enabled.

## Unit and Integration Tests
Run all tests with all features:
! cargo test --all-features --workspace $ARGUMENTS

## Documentation Tests
Verify code examples in documentation:
! cargo test --doc --all-features --workspace

## Test Summary
Show test count and coverage areas:
! find lib/tests pipeline/tests -name "*.rs" -type f 2>/dev/null | wc -l | xargs echo "Test files found:"
