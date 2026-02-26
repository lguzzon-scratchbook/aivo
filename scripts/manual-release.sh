#!/bin/bash
set -euo pipefail

VERSION="v$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')"
DIST="dist"
PROJECT_DIR="$(pwd)"

echo "Building aivo $VERSION"
echo "========================"

rm -rf "$DIST"
mkdir -p "$DIST"

# macOS (native)
echo "→ Building darwin-arm64..."
cargo build --release --target aarch64-apple-darwin
cp target/aarch64-apple-darwin/release/aivo "$DIST/aivo-darwin-arm64"

echo "→ Building darwin-x64..."
cargo build --release --target x86_64-apple-darwin
cp target/x86_64-apple-darwin/release/aivo "$DIST/aivo-darwin-x64"

# Linux x64 (Docker with amd64 platform — native build, no cross-compile)
echo "→ Building linux-x64..."
docker run --rm --platform linux/amd64 \
  -v "$PROJECT_DIR":/app \
  -w /app \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  rust:1.85 \
  bash -c "cargo build --release && cp /tmp/cargo-target/release/aivo /app/dist/aivo-linux-x64"

# Linux arm64 (Docker with arm64 platform — native build)
echo "→ Building linux-arm64..."
docker run --rm --platform linux/arm64 \
  -v "$PROJECT_DIR":/app \
  -w /app \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  rust:1.85 \
  bash -c "cargo build --release && cp /tmp/cargo-target/release/aivo /app/dist/aivo-linux-arm64"

# Windows x64 (cross-compile inside linux/amd64 container with mingw)
echo "→ Building windows-x64..."
docker run --rm --platform linux/amd64 \
  -v "$PROJECT_DIR":/app \
  -w /app \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  rust:1.85 \
  bash -c "apt-get update -qq && apt-get install -y -qq gcc-mingw-w64-x86-64 >/dev/null 2>&1 && \
    rustup target add x86_64-pc-windows-gnu && \
    cargo build --release --target x86_64-pc-windows-gnu && \
    cp /tmp/cargo-target/x86_64-pc-windows-gnu/release/aivo.exe /app/dist/aivo-windows-x64.exe"

# Generate SHA-256 checksums
echo "→ Generating checksums..."
cd "$DIST"
for f in aivo-*; do shasum -a 256 "$f" > "$f.sha256"; done
cd -

echo ""
echo "Built artifacts:"
ls -lh "$DIST/"

# Sync to aivo-releases repo
echo ""
echo "→ Syncing install.sh & LICENSE to aivo-releases..."
TMPDIR=$(mktemp -d)
git clone git@github.com:yuanchuan/aivo.git "$TMPDIR/aivo-releases"
cp scripts/install.sh LICENSE "$TMPDIR/aivo-releases/"
cd "$TMPDIR/aivo-releases"
git add -A
git diff --cached --quiet || git commit -m "sync from aivo $VERSION"
git push
cd -

# Create GitHub release
echo "→ Creating release $VERSION on aivo-releases..."
gh release create "$VERSION" \
  --repo yuanchuan/aivo \
  --title "$VERSION" \
  --notes "Release $VERSION" \
  "$DIST"/*

rm -rf "$TMPDIR"

echo ""
echo "Done! Release $VERSION published to yuanchuan/aivo"
