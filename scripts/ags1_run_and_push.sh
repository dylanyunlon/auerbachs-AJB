#!/usr/bin/env bash
# =================================================================
# AJB Experiment Runner: compile + run + auto-push results
# 
# 用法: bash scripts/ags1_run_and_push.sh [--skip-build] [--runs N]
#
# 流程:
#   1. 编译 ajb_benchmark (CUDA) 和 joinrenum CPU tests
#   2. 运行CPU tests验证基线
#   3. 运行GPU benchmark (多种数据分布 × 多种K_x值)
#   4. 收集结果到 experiment_data/
#   5. 自动git push到main, 让下游Claude拉取分析
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AJB_ROOT"

NUM_RUNS="${2:-5}"
SKIP_BUILD=false
[[ "${1:-}" == "--skip-build" ]] && SKIP_BUILD=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiment_data/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "[AJB_STATE] === AJB Experiment Run $TIMESTAMP ==="
echo "[AJB_STATE] NUM_RUNS=$NUM_RUNS SKIP_BUILD=$SKIP_BUILD"

# ---- Step 1: Build ----
if [ "$SKIP_BUILD" = false ]; then
    echo "[AJB_STATE] Building..."
    
    # CPU tests
    echo "[AJB_STATE] Compiling CPU tests..."
    cd src/joinrenum
    PASS=0; FAIL=0
    for name in test_bucket_pool test_count_oracle test_unordered_map test_join_tree test_index test_rr_access_tree test_enumerator test_join_baseline test_join_triangle test_renum_baseline; do
        if g++ -std=c++17 -O2 -DAJB_DEBUG -I. "tests/${name}.cpp" -lglpk -o "/tmp/$name" 2>/dev/null; then
            timeout 60 "/tmp/$name" > /dev/null 2>&1 && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
        else
            FAIL=$((FAIL+1))
        fi
    done
    echo "[AJB_STATE] CPU tests: $PASS PASS, $FAIL FAIL" | tee "${AJB_ROOT}/${RESULTS_DIR}/cpu_tests.txt"
    cd "$AJB_ROOT"
    
    # CUDA benchmark (如果nvcc可用)
    if command -v nvcc &>/dev/null; then
        echo "[AJB_STATE] Compiling CUDA benchmark..."
        nvcc -std=c++17 -O2 -arch=sm_80 \
            -I src/ -I third_party/moderngpu/src \
            src/ajb_benchmark.cu \
            -o build/ajb_benchmark \
            -lcudart -lnvToolsExt \
            2>&1 | tee "${RESULTS_DIR}/build.log"
    else
        echo "[AJB_BP] nvcc not available, skipping CUDA build"
    fi
fi

# ---- Step 2: Run GPU benchmarks ----
if [ -x build/ajb_benchmark ]; then
    echo "[AJB_STATE] Running GPU benchmarks..."
    
    # Baseline对比: 4种数据分布 × 3种K_x值 × 4种方法
    DISTRIBUTIONS=("uniform" "zipfian" "foreign_key" "many_to_many")
    KX_VALUES=(16 64 256)
    INPUT_SIZES=("1000000000" "7000000000" "13000000000")
    
    for dist in "${DISTRIBUTIONS[@]}"; do
        for kx in "${KX_VALUES[@]}"; do
            for input_size in "${INPUT_SIZES[@]}"; do
                echo "[AJB_STATE] Running: dist=$dist kx=$kx input=$input_size"
                
                # AJB方法
                timeout 3600 build/ajb_benchmark \
                    --method=ajb \
                    --distribution="$dist" \
                    --kx="$kx" \
                    --input-size="$input_size" \
                    --num-runs="$NUM_RUNS" \
                    --output="${RESULTS_DIR}/ajb_${dist}_kx${kx}_n${input_size}.csv" \
                    2>> "${RESULTS_DIR}/ajb_${dist}_kx${kx}_stderr.log" || \
                    echo "[AJB_BP] FAILED: ajb dist=$dist kx=$kx" >> "${RESULTS_DIR}/failures.txt"
                
                # Baseline: uniform-cadence
                timeout 3600 build/ajb_benchmark \
                    --method=uniform_cadence \
                    --distribution="$dist" \
                    --kx="$kx" \
                    --input-size="$input_size" \
                    --num-runs="$NUM_RUNS" \
                    --output="${RESULTS_DIR}/uniform_${dist}_kx${kx}_n${input_size}.csv" \
                    2>> "${RESULTS_DIR}/uniform_${dist}_kx${kx}_stderr.log" || \
                    echo "[AJB_BP] FAILED: uniform dist=$dist kx=$kx" >> "${RESULTS_DIR}/failures.txt"
                
                # Baseline: eager repartition
                timeout 3600 build/ajb_benchmark \
                    --method=eager \
                    --distribution="$dist" \
                    --kx="$kx" \
                    --input-size="$input_size" \
                    --num-runs="$NUM_RUNS" \
                    --output="${RESULTS_DIR}/eager_${dist}_kx${kx}_n${input_size}.csv" \
                    2>> "${RESULTS_DIR}/eager_${dist}_kx${kx}_stderr.log" || \
                    echo "[AJB_BP] FAILED: eager dist=$dist kx=$kx" >> "${RESULTS_DIR}/failures.txt"
                
                # Baseline: pin-local
                timeout 3600 build/ajb_benchmark \
                    --method=pin_local \
                    --distribution="$dist" \
                    --kx="$kx" \
                    --input-size="$input_size" \
                    --num-runs="$NUM_RUNS" \
                    --output="${RESULTS_DIR}/pinlocal_${dist}_kx${kx}_n${input_size}.csv" \
                    2>> "${RESULTS_DIR}/pinlocal_${dist}_kx${kx}_stderr.log" || \
                    echo "[AJB_BP] FAILED: pin_local dist=$dist kx=$kx" >> "${RESULTS_DIR}/failures.txt"
            done
        done
    done
else
    echo "[AJB_BP] build/ajb_benchmark not found, skipping GPU benchmarks"
fi

# ---- Step 3: Aggregate results ----
echo "[AJB_STATE] Aggregating results..."
python3 << PYEOF
import os, glob, csv
results_dir = "${RESULTS_DIR}"
all_csvs = sorted(glob.glob(f"{results_dir}/*.csv"))
if not all_csvs:
    print("[AJB_STATE] No CSV results found")
    exit(0)

# Merge all CSVs into one summary
with open(f"{results_dir}/summary.csv", "w") as out:
    writer = None
    for csvfile in all_csvs:
        method = os.path.basename(csvfile).split("_")[0]
        with open(csvfile) as inp:
            reader = csv.DictReader(inp)
            if writer is None:
                fieldnames = ["source_file", "method"] + reader.fieldnames
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writeheader()
            for row in reader:
                row["source_file"] = os.path.basename(csvfile)
                row["method"] = method
                writer.writerow(row)
print(f"[AJB_STATE] Summary written: {results_dir}/summary.csv ({len(all_csvs)} files merged)")
PYEOF

# ---- Step 4: Auto-push ----
echo "[AJB_STATE] Pushing results to GitHub..."
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
git add "$RESULTS_DIR/"
git commit -m "experiment: $TIMESTAMP — auto-collected benchmark data"
git push origin main 2>&1 || echo "[AJB_BP] git push failed"

echo "[AJB_STATE] === Experiment run $TIMESTAMP complete ==="
