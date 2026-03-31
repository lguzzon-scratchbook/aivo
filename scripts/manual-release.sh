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

# Upload to R2
echo ""
echo "→ Uploading to Cloudflare R2..."
for f in "$DIST"/aivo-*; do
  KEY="${VERSION}/$(basename "$f")"
  echo "  ${KEY}"
  wrangler r2 object put "aivo-releases/${KEY}" --file "$f" --remote
done
V="${VERSION#v}"
echo -n "$V" | wrangler r2 object put "aivo-releases/latest" --pipe --remote
echo "R2 upload complete."

# Update Homebrew tap
echo ""
echo "→ Updating Homebrew formula..."
HOMEBREW_TAP="$PROJECT_DIR/../homebrew-tap"
if [ -d "$HOMEBREW_TAP" ]; then
  FORMULA="$HOMEBREW_TAP/Formula/aivo.rb"
  V="${VERSION#v}"
  SHA_DARWIN_ARM64=$(awk '{print $1}' "$DIST/aivo-darwin-arm64.sha256")
  SHA_DARWIN_X64=$(awk '{print $1}' "$DIST/aivo-darwin-x64.sha256")
  SHA_LINUX_ARM64=$(awk '{print $1}' "$DIST/aivo-linux-arm64.sha256")
  SHA_LINUX_X64=$(awk '{print $1}' "$DIST/aivo-linux-x64.sha256")

  sed -i '' \
    -e "s/version \".*\"/version \"$V\"/" \
    "$FORMULA"

  # Update SHA256 values in order: darwin-arm64, darwin-x64, linux-arm64, linux-x64
  awk -v s1="$SHA_DARWIN_ARM64" -v s2="$SHA_DARWIN_X64" -v s3="$SHA_LINUX_ARM64" -v s4="$SHA_LINUX_X64" '
    BEGIN { n=0 }
    /sha256 "/ { n++; if(n==1) sub(/"[a-f0-9]{64}"/, "\"" s1 "\""); if(n==2) sub(/"[a-f0-9]{64}"/, "\"" s2 "\""); if(n==3) sub(/"[a-f0-9]{64}"/, "\"" s3 "\""); if(n==4) sub(/"[a-f0-9]{64}"/, "\"" s4 "\"") }
    { print }
  ' "$FORMULA" > "$FORMULA.tmp" && mv "$FORMULA.tmp" "$FORMULA"

  cd "$HOMEBREW_TAP"
  git add Formula/aivo.rb
  git diff --cached --quiet || git commit -m "Update aivo to $VERSION"
  git push
  cd "$PROJECT_DIR"
  echo "Homebrew formula updated to $VERSION"
else
  echo "Warning: homebrew-tap not found at $HOMEBREW_TAP, skipping"
fi

echo ""
echo "Done! Release $VERSION published to yuanchuan/aivo"
