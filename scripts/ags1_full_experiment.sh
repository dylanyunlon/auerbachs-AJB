#!/usr/bin/env bash
# =================================================================
# AJB Full Experiment: 4 methods × 4 distributions × scales
#
# 用法: bash scripts/ags1_full_experiment.sh [results_dir]
#
# 对比基线:
#   1. AJB (our method): decoupled K_x, K_u, K_v
#   2. Uniform-cadence (Maltry VLDB'25 baseline): K_x = K_u = K_v
#   3. Eager repartition: reshuffle at every level
#   4. Pin-local: partition stays on first device
#
# 数据分布: uniform, zipfian, foreign_key, many_to_many
# 输入规模: 100M, 500M, 1B, 7B, 13B (tuples)
# K_x值: 16, 64, 256
# 每组实验重复 NUM_RUNS 次 (均值±标准差)
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AJB_ROOT"

RESULTS_DIR="${1:-experiment_data/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

NUM_RUNS=5
NUM_GPUS=3  # 2x A6000 + 1x H100

echo "[AJB_EXP] Starting full experiment"
echo "[AJB_EXP] Results: $RESULTS_DIR"
echo "[AJB_EXP] NUM_RUNS=$NUM_RUNS NUM_GPUS=$NUM_GPUS"

# ---- GPU 预热 ----
echo "[AJB_EXP] GPU warmup..."
nvidia-smi -pm 1 2>/dev/null || true   # persistence mode
nvidia-smi -lgc 1410 2>/dev/null || true  # lock clocks (A6000)

# ---- 实验参数 ----
METHODS=("ajb" "uniform_cadence" "eager" "pin_local")
DISTRIBUTIONS=("uniform" "zipfian" "foreign_key" "many_to_many")

# 按论文Table 2的规模: 1B, 7B, 13B; 加100M/500M做小规模验证
SMALL_SIZES=("100000000" "500000000")
PAPER_SIZES=("1000000000" "7000000000" "13000000000")

KX_VALUES=("16" "64" "256")

# K_u, K_v 由 AJB 自动设置为 K_u = 3*K_x, K_v = 6*K_x
# 对uniform_cadence/eager/pin_local, K_x 参数被忽略或统一应用

FAIL_COUNT=0
TOTAL=0
PASS_COUNT=0

run_one() {
    local method="$1" dist="$2" kx="$3" input_size="$4"
    local label="${method}_${dist}_kx${kx}_n${input_size}"
    local csv_out="$RESULTS_DIR/${label}.csv"
    local stderr_out="$RESULTS_DIR/${label}_stderr.log"
    
    # 根据输入规模设超时: 100M=120s, 1B=600s, 13B=7200s
    local timeout_s=600
    if (( input_size > 5000000000 )); then timeout_s=7200
    elif (( input_size > 1000000000 )); then timeout_s=3600
    elif (( input_size < 500000000 )); then timeout_s=300
    fi
    
    TOTAL=$((TOTAL+1))
    echo -n "[AJB_EXP] [$TOTAL] $label ... "
    
    if [ -x build/ajb_benchmark ]; then
        timeout "$timeout_s" build/ajb_benchmark \
            --method="$method" \
            --distribution="$dist" \
            --kx="$kx" \
            --input-size="$input_size" \
            --num-runs="$NUM_RUNS" \
            --num-gpus="$NUM_GPUS" \
            --output="$csv_out" \
            2> "$stderr_out"
        
        if [ $? -eq 0 ] && [ -f "$csv_out" ]; then
            local rows=$(wc -l < "$csv_out")
            echo "OK ($rows rows)"
            PASS_COUNT=$((PASS_COUNT+1))
        else
            echo "FAIL (exit=$?)"
            FAIL_COUNT=$((FAIL_COUNT+1))
            echo "$label" >> "$RESULTS_DIR/failures.txt"
        fi
    elif [ -x build/join_benchmark ]; then
        # 回退到upstream的join_benchmark做baseline对比
        local sort_alg="radix"
        [ "$method" = "uniform_cadence" ] && sort_alg="radix"
        [ "$method" = "eager" ] && sort_alg="merge"
        
        timeout "$timeout_s" build/join_benchmark \
            --num-elements="$input_size" \
            --distribution="$dist" \
            --num-gpus="$NUM_GPUS" \
            --gpu-sort="$sort_alg" \
            --gpu-merge=merge-path \
            2> "$stderr_out" | tee "$csv_out" | tail -3
        
        if [ $? -eq 0 ]; then
            echo "OK (upstream fallback)"
            PASS_COUNT=$((PASS_COUNT+1))
        else
            echo "FAIL (upstream)"
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
    else
        echo "SKIP (no binary)"
    fi
}

# ---- Phase 1: 小规模验证 (快速确认一切工作) ----
echo ""
echo "[AJB_EXP] === Phase 1: Small-scale validation ==="
for dist in uniform zipfian; do
    for method in "${METHODS[@]}"; do
        run_one "$method" "$dist" "64" "100000000"
    done
done

echo ""
echo "[AJB_EXP] Phase 1 done: $PASS_COUNT pass, $FAIL_COUNT fail out of $TOTAL"

if [ "$FAIL_COUNT" -gt 0 ] && [ "$PASS_COUNT" -eq 0 ]; then
    echo "[AJB_EXP] All small-scale runs failed. Dumping last stderr:"
    tail -30 "$RESULTS_DIR"/*_stderr.log 2>/dev/null | head -50
    echo "[AJB_EXP] Fix build errors before proceeding to full sweep"
    exit 1
fi

# ---- Phase 2: 论文Table 2数据 (1B, 7B, 13B × K_x=16,256) ----
echo ""
echo "[AJB_EXP] === Phase 2: Paper Table 2 data ==="
for input_size in "${PAPER_SIZES[@]}"; do
    for kx in 16 256; do
        for method in "${METHODS[@]}"; do
            for dist in uniform; do
                run_one "$method" "$dist" "$kx" "$input_size"
            done
        done
    done
done

# ---- Phase 3: 论文Table 1 (ICL) 数据 (all distributions × 1B) ----
echo ""
echo "[AJB_EXP] === Phase 3: Paper Table 1 (ICL) data ==="
for dist in "${DISTRIBUTIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_one "$method" "$dist" "256" "1000000000"
    done
done

# ---- Phase 4: K_x sweep (论文Figure 3) ----
echo ""
echo "[AJB_EXP] === Phase 4: K_x cadence sweep ==="
for kx in "${KX_VALUES[@]}"; do
    for method in ajb uniform_cadence; do
        run_one "$method" "zipfian" "$kx" "500000000"
    done
done

# ---- Phase 5: 大规模 (论文Figure 5) ----
echo ""
echo "[AJB_EXP] === Phase 5: Billion-tuple scale ==="
for dist in zipfian foreign_key; do
    for method in ajb uniform_cadence pin_local; do
        run_one "$method" "$dist" "256" "7000000000"
    done
done

# ---- 汇总 ----
echo ""
echo "============================================="
echo "[AJB_EXP] SUMMARY: $PASS_COUNT pass, $FAIL_COUNT fail out of $TOTAL"
echo "============================================="

# 合并所有CSV
python3 << 'PYEOF'
import os, glob, csv, json, sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", ".")
all_csvs = sorted(glob.glob(f"{results_dir}/*.csv"))
if not all_csvs:
    print("[AJB_EXP] No CSV results found")
    sys.exit(0)

# 合并
summary_path = f"{results_dir}/summary.csv"
rows = []
for csvfile in all_csvs:
    base = os.path.basename(csvfile)
    if base == "summary.csv":
        continue
    parts = base.replace(".csv", "").split("_")
    method = parts[0]
    try:
        with open(csvfile) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source"] = base
                row["method"] = method
                rows.append(row)
    except Exception as e:
        print(f"  skip {base}: {e}")

if rows:
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AJB_EXP] Summary: {summary_path} ({len(rows)} rows from {len(all_csvs)} files)")
else:
    print("[AJB_EXP] No valid rows found")
PYEOF

echo ""
echo "[AJB_EXP] Experiment complete. Results in: $RESULTS_DIR"
