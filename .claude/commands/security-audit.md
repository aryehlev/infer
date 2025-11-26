---
description: Perform security audit of dependencies and code
allowed-tools: Bash(cargo:*), Bash(grep:*), Bash(find:*)
---

# Security Audit

Analyze code and dependencies for security vulnerabilities.

## Dependency Tree
Show direct and transitive dependencies:
! cargo tree --depth 2

## Check for Unsafe Code
Audit all unsafe blocks in the codebase:
! grep -r "unsafe" lib/src pipeline/src --include="*.rs" -n -B 1 -A 3 2>/dev/null || echo "No unsafe code found"

## Recommended Actions
If cargo-deny is installed, run:
`cargo install cargo-deny`
`cargo deny check advisories`

For vulnerability scanning:
`cargo install cargo-audit`
`cargo audit`
