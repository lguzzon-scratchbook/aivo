# Makefile for aivo CLI
# Quick commands for development

.PHONY: build build-debug build-release test check clippy clean install fmt release

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "aivo CLI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build the real release binary
	cargo build --release

build-debug: ## Build debug binary for local development
	cargo build

build-release: build ## Build optimized release binary

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

install: build ## Install binary to /usr/local/bin (re-signs for macOS arm64)
	cp target/release/aivo /usr/local/bin/aivo
	codesign --force -s - /usr/local/bin/aivo 2>/dev/null || true

dev: check test clippy ## Run all checks (check, test, clippy)

release: test clippy build ## Full release workflow (test, lint, build)
	@echo "Release binary ready at: target/release/aivo"
	@ls -lh target/release/aivo | awk '{print "Size:", $$5}'
