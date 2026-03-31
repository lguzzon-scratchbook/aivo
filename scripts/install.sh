#!/bin/sh
set -e

REPO="yuanchuan/aivo"
BINARY="aivo"
INSTALL_DIR="${AIVO_INSTALL_DIR:-/usr/local/bin}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

# Handle Windows (Git Bash, WSL, or Cygwin)
case "$OS" in
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM="windows"
    BINARY="${BINARY}.exe"
    ARCH="x64"
    ;;
  Linux)  PLATFORM="linux" ;;
  Darwin) PLATFORM="darwin" ;;
  *)      echo "Error: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64)  ARCH="x64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *)             echo "Error: Unsupported architecture: $ARCH"; exit 1 ;;
esac

ARTIFACT="${BINARY}-${PLATFORM}-${ARCH}"

GITHUB_BASE="https://github.com/${REPO}/releases/latest/download"
MIRROR_BASE="https://getaivo.dev/dl/latest"

download_file() {
  url="$1"
  output="$2"
  fallback="$3"

  if command -v curl >/dev/null 2>&1; then
    if curl -fSL --connect-timeout 8 --max-time 120 --progress-bar "$url" -o "$output" 2>/dev/null; then
      return 0
    fi
    if [ -n "$fallback" ]; then
      echo "  Falling back to mirror..."
      curl -fSL --progress-bar "$fallback" -o "$output"
      return $?
    fi
    return 1
  elif command -v wget >/dev/null 2>&1; then
    if wget --connect-timeout=8 --timeout=120 -q --show-progress "$url" -O "$output" 2>/dev/null; then
      return 0
    fi
    if [ -n "$fallback" ]; then
      echo "  Falling back to mirror..."
      wget -q --show-progress "$fallback" -O "$output"
      return $?
    fi
    return 1
  else
    echo "Error: curl or wget is required"
    exit 1
  fi
}

# Create temp directory
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Download
echo "Downloading ${ARTIFACT}..."
download_file "${GITHUB_BASE}/${ARTIFACT}" "${TMP_DIR}/${BINARY}" "${MIRROR_BASE}/${ARTIFACT}"
download_file "${GITHUB_BASE}/${ARTIFACT}.sha256" "${TMP_DIR}/${ARTIFACT}.sha256" "${MIRROR_BASE}/${ARTIFACT}.sha256"

EXPECTED_SHA="$(awk '{print $1}' "${TMP_DIR}/${ARTIFACT}.sha256" | tr -d '\r\n')"
if ! printf '%s' "$EXPECTED_SHA" | grep -Eq '^[A-Fa-f0-9]{64}$'; then
  echo "Error: Invalid checksum format"
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA="$(sha256sum "${TMP_DIR}/${BINARY}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL_SHA="$(shasum -a 256 "${TMP_DIR}/${BINARY}" | awk '{print $1}')"
elif command -v openssl >/dev/null 2>&1; then
  ACTUAL_SHA="$(openssl dgst -sha256 "${TMP_DIR}/${BINARY}" | awk '{print $NF}')"
else
  echo "Error: sha256sum, shasum, or openssl is required for checksum verification"
  exit 1
fi

EXPECTED_SHA_LOWER="$(printf '%s' "$EXPECTED_SHA" | tr 'A-F' 'a-f')"
ACTUAL_SHA_LOWER="$(printf '%s' "$ACTUAL_SHA" | tr 'A-F' 'a-f')"

if [ "${ACTUAL_SHA_LOWER}" != "${EXPECTED_SHA_LOWER}" ]; then
  echo "Error: Checksum verification failed for ${ARTIFACT}"
  exit 1
fi
echo "Checksum verified."

# On Windows, the .exe file is already executable
# On Unix-like systems, make it executable
case "$OS" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *) chmod +x "${TMP_DIR}/${BINARY}" ;;
esac

# Install
if [ -w "$INSTALL_DIR" ]; then
  mv "${TMP_DIR}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
else
  echo "Installing to ${INSTALL_DIR} (requires sudo)..."
  sudo mv "${TMP_DIR}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
fi

echo ""
echo "aivo installed to ${INSTALL_DIR}/${BINARY}"
echo ""
echo "Next steps:"
echo "  aivo keys add       # Add an API key"
echo "  aivo run claude     # or codex, gemini"
echo "  aivo --help"
