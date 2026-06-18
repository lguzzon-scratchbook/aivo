#!/usr/bin/env bash
# Fallback Validation E2E Test
#
# Tests that the CLI enforces fallback naming and content constraints:
#   1. Empty name is rejected
#   2. Whitespace-only name is rejected
#   3. '@' in name is rejected (ambiguity with @fallback reference syntax)
#   4. ':' in name is rejected (conflict with provider:model format)
#   5. ':' in @fallback reference target is rejected
#   6. Empty provider in target is rejected
#   7. Valid fallback creation succeeds
#   8. Valid fallback removal works
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AIVO="$REPO_ROOT/target/debug/aivo"
ERRORS=0

echo "=== Step 1: Build ==="
cargo build --quiet 2>&1

#
# Test 1: Reject empty fallback name
#
echo ""
echo "=== Test 1: Reject empty fallback name ==="
OUTPUT=$("$AIVO" fallback --set "" -- anthropic:claude-sonnet-4-6 2>&1) && {
    echo "FAIL: Expected error for empty name, got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "cannot be empty" 2>/dev/null; then
        echo "  ✓ Rejected empty name with descriptive error"
    else
        echo "  ? Rejected empty name but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 2: Reject whitespace-only fallback name
#
echo ""
echo "=== Test 2: Reject whitespace-only fallback name ==="
OUTPUT=$("$AIVO" fallback --set "   " -- anthropic:claude-sonnet-4-6 2>&1) && {
    echo "FAIL: Expected error for whitespace-only name, got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "cannot be empty\|whitespace" 2>/dev/null; then
        echo "  ✓ Rejected whitespace-only name with descriptive error"
    else
        echo "  ? Rejected whitespace-only name but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 3: Reject '@' in fallback name
#
echo ""
echo "=== Test 3: Reject '@' in fallback name ==="
OUTPUT=$("$AIVO" fallback --set "@primary" -- anthropic:claude-sonnet-4-6 2>&1) && {
    echo "FAIL: Expected error for '@primary', got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "reserved.*'@'\|'@'\|@.*reserved" 2>/dev/null; then
        echo "  ✓ Rejected '@primary' with descriptive error"
    elif echo "$OUTPUT" | grep -qi "Validation failed" 2>/dev/null; then
        echo "  ✓ Rejected '@primary' via validation"
    else
        echo "  ? Rejected '@primary' but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 4: Reject colon in fallback name
#
echo ""
echo "=== Test 4: Reject ':' in fallback name ==="
OUTPUT=$("$AIVO" fallback --set "bad:name" -- anthropic:claude-sonnet-4-6 2>&1) && {
    echo "FAIL: Expected error for 'bad:name', got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "reserved\|colon\|':'" 2>/dev/null; then
        echo "  ✓ Rejected 'bad:name' with descriptive error"
    elif echo "$OUTPUT" | grep -qi "Validation failed" 2>/dev/null; then
        echo "  ✓ Rejected 'bad:name' via validation (fallback message)"
    else
        echo "  ? Rejected 'bad:name' but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 5: Reject colon in fallback reference target (@ref)
#
echo ""
echo "=== Test 5: Reject ':' in @fallback reference target ==="
OUTPUT=$("$AIVO" fallback --set "test-fb" -- "@bad:ref" 2>&1) && {
    echo "FAIL: Expected error for @bad:ref, got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "reserved\|colon\|':'" 2>/dev/null; then
        echo "  ✓ Rejected @bad:ref with descriptive error"
    elif echo "$OUTPUT" | grep -qi "Validation failed" 2>/dev/null; then
        echo "  ✓ Rejected @bad:ref via validation"
    else
        echo "  ? Rejected @bad:ref but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 6: Reject empty provider in target
#
echo ""
echo "=== Test 6: Reject empty provider in target ==="
OUTPUT=$("$AIVO" fallback --set "test-fb" -- ":gpt-4o" 2>&1) && {
    echo "FAIL: Expected error for empty provider, got success"
    ERRORS=$((ERRORS + 1))
} || {
    if echo "$OUTPUT" | grep -qi "empty provider\|Invalid target\|empty.*provider" 2>/dev/null; then
        echo "  ✓ Rejected empty provider with descriptive error"
    else
        echo "  ? Rejected empty provider but unexpected error: $OUTPUT"
        ERRORS=$((ERRORS + 1))
    fi
}

#
# Test 7: Valid fallback creation succeeds
#
echo ""
echo "=== Test 7: Valid fallback creation succeeds ==="
OUTPUT=$("$AIVO" fallback --set "e2e-valid" -- anthropic:claude-sonnet-4-6 2>&1)
echo "$OUTPUT"

if "$AIVO" fallback 2>&1 | grep -qw "e2e-valid"; then
    echo "  ✓ 'e2e-valid' appears in fallback list"
else
    echo "FAIL: 'e2e-valid' not found in fallback list"
    ERRORS=$((ERRORS + 1))
fi

#
# Test 8: Remove valid fallback
#
echo ""
echo "=== Test 8: Remove fallback ==="
OUTPUT=$("$AIVO" fallback --rm "e2e-valid" 2>&1)
echo "$OUTPUT"

if "$AIVO" fallback 2>&1 | grep -qw "e2e-valid"; then
    echo "FAIL: 'e2e-valid' still in fallback list after removal"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ 'e2e-valid' removed successfully"
fi

#
# Summary
#
echo ""
echo "=== Results ==="
if [ "$ERRORS" -ne 0 ]; then
    echo "FAIL ($ERRORS error(s))"
    exit 1
fi
echo "PASS"
