#!/bin/bash
# =============================================================================
# build.sh — AJB project builder
#
# Origin: upstream/multi-gpu-sort-merge-join/build.sh
# Algorithm changes vs upstream:
#   1. Incremental build: hash CMakeLists.txt+compiler version into a stamp
#      file; skip cmake configure when stamp matches (upstream always re-runs)
#   2. Parallel jobs: probe /proc/meminfo for available RAM, cap jobs at
#      RAM_MB/2048 to avoid OOM on large CUDA builds (upstream hardcodes -j8)
#   3. Two-pass build on failure: if first pass fails, retry with -j1 to get
#      a clean error (parallel builds scramble diagnostics)
#   4. ccache probe: if ccache exists, inject it as CMAKE_CUDA_COMPILER_LAUNCHER
# =============================================================================

set -euo pipefail

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

BUILD_TYPE="Release"
if [[ "${1:-}" = "Debug" ]]; then
    BUILD_TYPE="$1"
fi

# --- algorithm 1: incremental configure via content-hash stamp ---
STAMP_FILE="$BUILD_DIR/.configure_stamp"
CML_HASH=$(sha256sum CMakeLists.txt 2>/dev/null | cut -c1-16)
CXX_VER=$(${CXX:-g++} --version 2>/dev/null | head -1 | tr -d ' ')
NVCC_VER=$(nvcc --version 2>/dev/null | tail -1 | tr -d ' ' || echo "none")
CURRENT_STAMP="${BUILD_TYPE}:${CML_HASH}:${CXX_VER}:${NVCC_VER}"

NEED_CONFIGURE=1
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$CURRENT_STAMP" ]; then
    NEED_CONFIGURE=0
fi

# --- algorithm 2: memory-aware parallelism ---
if [ -f /proc/meminfo ]; then
    AVAIL_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    MEM_JOBS=$((AVAIL_MB / 2048))
    [ "$MEM_JOBS" -lt 1 ] && MEM_JOBS=1
    CPU_JOBS=$(nproc 2>/dev/null || echo 4)
    JOBS=$((CPU_JOBS < MEM_JOBS ? CPU_JOBS : MEM_JOBS))
else
    JOBS=$(nproc 2>/dev/null || echo 8)
fi

# --- algorithm 4: ccache injection ---
CCACHE_FLAGS=""
if command -v ccache &>/dev/null; then
    CCACHE_FLAGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
fi

cd "$BUILD_DIR"

if [ "$NEED_CONFIGURE" -eq 1 ]; then
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE $CCACHE_FLAGS ..
    echo "$CURRENT_STAMP" > "$STAMP_FILE"
fi

# --- algorithm 3: two-pass build with fallback ---
if ! cmake --build . -- -j${JOBS} 2>&1; then
    echo "parallel build failed (j=$JOBS), retrying j=1 for clean diagnostics..." >&2
    cmake --build . -- -j1
fi

cd ..
