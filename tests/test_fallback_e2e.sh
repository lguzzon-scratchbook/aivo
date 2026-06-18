#!/usr/bin/env bash
# Fallback E2E Integration Test
#
# Tests that aivo's fallback retry mechanism correctly falls through from
# a failing provider (hyper:deepseek-v4-flash → 401 malformed token) to a
# succeeding one (kiloGateway:kilo-auto/small → exit 0).
#
# Validates:
#   1. Fallback definition is created and resolved
#   2. First target fails, fallback is triggered ("fallback target 1/2 failed")
#   3. Second target succeeds (exit 0, "Using key: kiloGateway")
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AIVO="$REPO_ROOT/target/debug/aivo"

echo "=== Step 1: Build ==="
cargo build --quiet 2>&1

echo "=== Step 2: Source .env ==="
set -a; source "$REPO_ROOT/.env"; set +a

echo "=== Step 3: Check/add keys ==="
KEYS_BEFORE="$($AIVO keys 2>&1)"
echo "$KEYS_BEFORE"
ADDED_HYPER=false; ADDED_KILO=false

if ! echo "$KEYS_BEFORE" | grep -qw "hyper"; then
  $AIVO keys add --name hyper --base-url "$HYPER_OPENAI_BASE_URL" --key "$HYPER_API_KEY"
  ADDED_HYPER=true
fi
if ! echo "$KEYS_BEFORE" | grep -qw "kiloGateway"; then
  $AIVO keys add --name kiloGateway --base-url "$KILO_OPENAI_BASE_URL" --key "$KILO_API_KEY"
  ADDED_KILO=true
fi

echo "=== Step 4: Create fallback ==="
FALLBACK_ID="e2e-test-fallback"
$AIVO fallback --set "$FALLBACK_ID" -- hyper:deepseek-v4-flash kiloGateway:kilo-auto/small

echo "=== Step 5: Run with fallback ==="
# Capture stdout and stderr separately since claude -p output goes through MCP
STDERR_FILE=$(mktemp)
STDOUT_FILE=$(mktemp)
set +e
"$AIVO" claude --model "$FALLBACK_ID" -p "Write out only OK" > "$STDOUT_FILE" 2>"$STDERR_FILE"
RC=$?
set -e
echo "EXIT CODE: $RC"
echo "=== STDERR ==="
cat "$STDERR_FILE"
echo "=== STDOUT ==="
cat "$STDOUT_FILE"

echo "=== Step 6: Validate ==="
ERRORS=0
if [ "$RC" -ne 0 ]; then
  echo "FAIL: Exit code $RC (expected 0)"
  ERRORS=1
fi
# Check that fallback was triggered (first target failed)
if grep -q "fallback target 1/2 failed" "$STDERR_FILE"; then
  echo "  ✓ Fallback triggered (hyper failed)"
else
  echo "  ? First target may have succeeded (fallback not triggered)"
fi
# Check that second target succeeded
if grep -q "Using key: kiloGateway" "$STDERR_FILE"; then
  echo "  ✓ kiloGateway target was used"
else
  echo "FAIL: kiloGateway target was not reached"
  ERRORS=1
fi

rm -f "$STDOUT_FILE" "$STDERR_FILE"

if [ "$ERRORS" -ne 0 ]; then
  echo "FAIL"
  exit 1
fi
echo "PASS"

echo "=== Step 7: Cleanup ==="
$AIVO fallback --rm "$FALLBACK_ID" 2>/dev/null || true
if $ADDED_HYPER; then $AIVO keys rm hyper 2>/dev/null || true; fi
if $ADDED_KILO; then $AIVO keys rm kiloGateway 2>/dev/null || true; fi
