#!/bin/bash
# =============================================================================
# build_and_trace.sh — AJB debug build + trace execution
#
# Builds the AJB benchmark in debug mode with all trace macros enabled,
# then runs it (or any specified target) with structured output.
#
# Usage:
#   ./scripts/debug/build_and_trace.sh                    # build only
#   ./scripts/debug/build_and_trace.sh --run [args...]    # build + run
#   ./scripts/debug/build_and_trace.sh --target TARGET    # specific cmake target
#   ./scripts/debug/build_and_trace.sh --cpu-only         # CPU-only test build
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build_debug"

TARGET="ajb_benchmark"
RUN=false
CPU_ONLY=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)      RUN=true; shift; EXTRA_ARGS="$*"; break ;;
    --target)   TARGET="$2"; shift 2 ;;
    --cpu-only) CPU_ONLY=true; shift ;;
    *)          shift ;;
  esac
done

echo "[AJB_BUILD] Project: $PROJECT_ROOT"
echo "[AJB_BUILD] Target:  $TARGET"
echo "[AJB_BUILD] Mode:    $([ "$CPU_ONLY" = true ] && echo 'CPU-only' || echo 'CUDA debug')"

if [ "$CPU_ONLY" = true ]; then
    # CPU-only test build (no CUDA required)
    echo "[AJB_BUILD] Compiling CPU-only test..."
    mkdir -p "$BUILD_DIR"
    g++ -x c++ -std=c++17 -O0 -g \
        -DCPU_ONLY_TEST -DAJB_DEBUG -DAJB_TRACE_DECISIONS -DAJB_ENABLE_COUNTERS \
        -I"$PROJECT_ROOT/src" \
        -I"$PROJECT_ROOT/src/ajb_join" \
        -I"$PROJECT_ROOT/src/joinrenum" \
        "$PROJECT_ROOT/src/ajb_benchmark.cu" \
        -o "$BUILD_DIR/ajb_test_cpu" \
        2>&1 || {
            echo "[AJB_BUILD] CPU-only build failed — this is expected if ajb_benchmark.cu"
            echo "            has CUDA-specific includes not guarded by CPU_ONLY_TEST."
            echo "            Use the CMake path for full builds."
            exit 1
        }
    echo "[AJB_BUILD] CPU-only binary: $BUILD_DIR/ajb_test_cpu"
else
    # Full CUDA debug build
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$PROJECT_ROOT" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CUDA_FLAGS="-DAJB_DEBUG -DAJB_TRACE_DECISIONS -DAJB_TRACE_TRANSFERS -DAJB_ENABLE_COUNTERS" \
        2>&1 | tail -5
    make -j$(nproc) "$TARGET" 2>&1
    echo "[AJB_BUILD] Binary: $BUILD_DIR/$TARGET"
fi

if [ "$RUN" = true ]; then
    echo ""
    echo "[AJB_RUN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[AJB_RUN] Starting $TARGET with trace output"
    echo "[AJB_RUN] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BIN="$BUILD_DIR/$TARGET"
    [ "$CPU_ONLY" = true ] && BIN="$BUILD_DIR/ajb_test_cpu"

    # Pipe through tee so we get both console and log
    $BIN $EXTRA_ARGS 2>&1 | tee "$BUILD_DIR/${TARGET}_trace_$(date +%Y%m%d_%H%M%S).log"

    echo "[AJB_RUN] Trace log: $BUILD_DIR/${TARGET}_trace_*.log"
fi
