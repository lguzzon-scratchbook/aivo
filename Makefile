# Makefile for aivo CLI
# Quick commands for development

.PHONY: build test check clippy clean install fmt release

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "aivo CLI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build release binary
	cargo build --release

test: ## Run all tests
	cargo test --features test-fast-crypto

test-release: ## Run tests on release build
	cargo test --release --features test-fast-crypto

check: ## Quick type check
	cargo check

clippy: ## Run clippy linter
	cargo clippy

fmt: ## Format code
	cargo fmt

clean: ## Clean build artifacts
	cargo clean

install: build ## Install binary locally (requires cargo install)
	cargo install --path . --force

dev: check test clippy ## Run all checks (check, test, clippy)

release: test clippy build ## Full release workflow (test, lint, build)
	@echo "Release binary ready at: target/release/aivo"
	@ls -lh target/release/aivo | awk '{print "Size:", $$5}'
