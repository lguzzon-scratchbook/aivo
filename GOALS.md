# Goals

Fitness goals for aivo (Rust CLI)

## North Stars

- All checks pass on every commit
- Clean clippy with no warnings

## Anti Stars

- Untested changes reaching main
- Unsafe code without justification

## Directives

### 1. Establish baseline

Get all gates passing and maintain a green baseline.

**Steer:** increase

### 2. Test coverage

Maintain and increase test coverage.

**Steer:** increase

## Gates

| ID | Check | Weight | Description |
|----|-------|--------|-------------|
| cargo-test | `cargo test` | 5 | Cargo tests pass |
| make-build | `make build` | 5 | Make build succeeds |
