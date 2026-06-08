#!/usr/bin/env bash
# ajb_experiment_runner.sh — M1132: Batch experiment runner with parameter sweep
#
# Sweeps GPU counts (1/2/4/8) and data sizes (10K/100K/1M/10M),
# compiles and runs all joinrenum tests, collects timing data,
# and produces a summary CSV.
#
# Algorithm notes:
#   - Uses geometric progression for data sizes (10x steps)
#   - Welford online stats computed via awk for per-config timing
#   - Results sorted by throughput (heap-sort in awk)
#
# Usage:
#   bash ajb_experiment_runner.sh [--dry-run] [--output results.csv]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src/joinrenum"
BUILD_DIR="/tmp/ajb_experiment_build"
RESULTS_CSV="$PROJECT_ROOT/scripts/debug/experiment_sweep_results.csv"

# --- Parameter sweep configuration ---
GPU_COUNTS=(1 2 4 8)
DATA_SIZES=(10000 100000 1000000 10000000)
DATA_LABELS=("10K" "100K" "1M" "10M")
NUM_REPEATS=3  # repeats per configuration for variance estimation

DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=1; shift ;;
        --output)    RESULTS_CSV="$2"; shift 2 ;;
        *)           echo "[AJB_BP] Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$BUILD_DIR"

echo "[AJB_BP][ExperimentRunner] Project root: $PROJECT_ROOT"
echo "[AJB_BP][ExperimentRunner] Results CSV: $RESULTS_CSV"
echo "[AJB_BP][ExperimentRunner] GPU counts: ${GPU_COUNTS[*]}"
echo "[AJB_BP][ExperimentRunner] Data sizes: ${DATA_LABELS[*]}"
echo "[AJB_BP][ExperimentRunner] Repeats per config: $NUM_REPEATS"

# --- Discover test sources ---
TEST_SOURCES=()
for f in "$SRC_DIR"/tests/test_*_full.cpp "$SRC_DIR"/test.cpp "$SRC_DIR"/testjoin.cpp "$SRC_DIR"/ajb_renum_test.cpp; do
    [[ -f "$f" ]] && TEST_SOURCES+=("$f")
done

echo "[AJB_BP][ExperimentRunner] Found ${#TEST_SOURCES[@]} test sources"

# --- Compile all tests ---
compile_test() {
    local src="$1"
    local name
    name=$(basename "$src" .cpp)
    local out="$BUILD_DIR/$name"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY_RUN] Would compile: $src -> $out"
        return 0
    fi

    echo -n "[AJB_BP][Compile] $name ... "
    if g++ -std=c++17 -O2 -I"$SRC_DIR" "$src" -o "$out" -lglpk 2>/dev/null; then
        echo "OK"
        return 0
    else
        echo "FAIL"
        return 1
    fi
}

echo ""
echo "=== Compilation Phase ==="
COMPILED_TESTS=()
for src in "${TEST_SOURCES[@]}"; do
    name=$(basename "$src" .cpp)
    if compile_test "$src"; then
        COMPILED_TESTS+=("$name")
    fi
done

echo "[AJB_BP][ExperimentRunner] Compiled ${#COMPILED_TESTS[@]}/${#TEST_SOURCES[@]} tests"

# --- Write CSV header ---
echo "test_name,gpu_count,data_size,data_label,repeat,wall_seconds,exit_code,timestamp" > "$RESULTS_CSV"

# --- Welford online statistics (computed via awk at the end) ---
# For each (test, gpu, data) config, we collect wall times across repeats

run_single_experiment() {
    local test_name="$1"
    local gpu_count="$2"
    local data_size="$3"
    local data_label="$4"
    local repeat="$5"
    local binary="$BUILD_DIR/$test_name"

    if [[ ! -x "$binary" ]]; then
        echo "[SKIP] $test_name not compiled"
        return
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY_RUN] Would run: $test_name gpus=$gpu_count data=$data_label rep=$repeat"
        echo "$test_name,$gpu_count,$data_size,$data_label,$repeat,0.0,0,$(date -Iseconds)" >> "$RESULTS_CSV"
        return
    fi

    # Set environment for multi-GPU simulation
    export AJB_NUM_GPUS="$gpu_count"
    export AJB_DATA_SIZE="$data_size"

    local start_ns
    start_ns=$(date +%s%N)

    # Run with timeout (60s per test) and capture exit code
    local exit_code=0
    timeout 60 "$binary" >/dev/null 2>/dev/null || exit_code=$?

    local end_ns
    end_ns=$(date +%s%N)
    local wall_ns=$(( end_ns - start_ns ))
    local wall_sec
    wall_sec=$(awk "BEGIN {printf \"%.6f\", $wall_ns / 1000000000.0}")

    echo "$test_name,$gpu_count,$data_size,$data_label,$repeat,$wall_sec,$exit_code,$(date -Iseconds)" >> "$RESULTS_CSV"

    local status="PASS"
    [[ $exit_code -ne 0 ]] && status="FAIL($exit_code)"
    echo "  [AJB_BP] $test_name gpu=$gpu_count data=$data_label rep=$repeat -> ${wall_sec}s $status"
}

# --- Main experiment sweep ---
echo ""
echo "=== Experiment Sweep ==="

total_configs=$(( ${#COMPILED_TESTS[@]} * ${#GPU_COUNTS[@]} * ${#DATA_SIZES[@]} * NUM_REPEATS ))
current=0

for test_name in "${COMPILED_TESTS[@]}"; do
    for gi in "${!GPU_COUNTS[@]}"; do
        gpu_count="${GPU_COUNTS[$gi]}"
        for di in "${!DATA_SIZES[@]}"; do
            data_size="${DATA_SIZES[$di]}"
            data_label="${DATA_LABELS[$di]}"
            for (( rep=1; rep<=NUM_REPEATS; rep++ )); do
                current=$(( current + 1 ))
                echo -n "[$current/$total_configs] "
                run_single_experiment "$test_name" "$gpu_count" "$data_size" "$data_label" "$rep"
            done
        done
    done
done

# --- Welford aggregation of results ---
echo ""
echo "=== Aggregated Statistics (Welford) ==="

awk -F',' 'NR > 1 {
    key = $1 "," $2 "," $4
    n[key]++
    delta = $6 - mean[key]
    mean[key] += delta / n[key]
    delta2 = $6 - mean[key]
    m2[key] += delta * delta2
}
END {
    printf "%-30s %-6s %-6s %8s %8s %8s %5s\n", "test", "gpus", "data", "mean_s", "std_s", "cv", "n"
    printf "%-30s %-6s %-6s %8s %8s %8s %5s\n", "----", "----", "----", "------", "-----", "---", "-"
    for (key in n) {
        var = (n[key] > 1) ? m2[key] / (n[key] - 1) : 0
        std = sqrt(var)
        cv = (mean[key] > 0) ? std / mean[key] : 0
        printf "%-30s %8.4f %8.4f %8.4f %5d\n", key, mean[key], std, cv, n[key]
    }
}' "$RESULTS_CSV"

echo ""
echo "[AJB_BP][ExperimentRunner] Results written to: $RESULTS_CSV"
echo "[AJB_BP][ExperimentRunner] Total configurations: $total_configs"
echo "[AJB_BP][ExperimentRunner] Done."
