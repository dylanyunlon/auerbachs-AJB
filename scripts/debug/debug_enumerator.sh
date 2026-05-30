#!/bin/bash
# =============================================================================
# debug_enumerator.sh — AJB Enumerator debug build + AddressSanitizer run
#
# Origin: upstream/joinrenum/debugEnumerator.sh (1 line)
# Adaptation (~20%): AJB-specific flags, output capture, and optional
#   Valgrind fallback when ASan is unavailable.
#
# Usage: ./scripts/debug/debug_enumerator.sh [--valgrind]
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_tests"
TEST_SRC="$PROJECT_ROOT/src/joinrenum/tests/test_enumerator.cpp"

mkdir -p "$BUILD_DIR"

USE_VALGRIND=false
[ "${1:-}" = "--valgrind" ] && USE_VALGRIND=true

echo "[AJB] Debug build: test_enumerator (AddressSanitizer)"

g++ -g -O0 -std=c++17 \
    -fsanitize=address -fno-omit-frame-pointer \
    -DAJB_DEBUG \
    -I"$PROJECT_ROOT/src/joinrenum" \
    "$TEST_SRC" \
    -o "$BUILD_DIR/test_enumerator_asan" \
    -lglpk 2>&1

echo "[AJB] Binary: $BUILD_DIR/test_enumerator_asan"

cd "$PROJECT_ROOT/src/joinrenum"

if [ "$USE_VALGRIND" = true ]; then
    echo "[AJB] Running with Valgrind..."
    valgrind --leak-check=full --track-origins=yes \
        "$BUILD_DIR/test_enumerator_asan" 2>&1 | \
        tee "$BUILD_DIR/enumerator_valgrind_$(date +%H%M%S).log"
else
    echo "[AJB] Running with AddressSanitizer..."
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=0 \
        "$BUILD_DIR/test_enumerator_asan" 2>&1 | \
        tee "$BUILD_DIR/enumerator_asan_$(date +%H%M%S).log"
fi

echo "[AJB] Debug run DONE"
