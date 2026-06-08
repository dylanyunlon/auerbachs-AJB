#!/usr/bin/env bash
# =============================================================================
# ags1_quickstart.sh — 在实验室机器上一键执行的完整命令序列
#
# 使用: 在ags1上复制粘贴这整个脚本, 或者:
#   bash <(curl -sL https://raw.githubusercontent.com/dylanyunlon/auerbachs-AJB/main/ags1_quickstart.sh)
# =============================================================================
set -euo pipefail

WORK="/data/jiacheng/system/cache/temp/nips2026"
cd "$WORK"

# === Step 1: 拉取最新代码 ===
if [ -d auerbachs-AJB ]; then
  cd auerbachs-AJB && git pull origin main
else
  git clone https://github.com/dylanyunlon/auerbachs-AJB.git && cd auerbachs-AJB
fi
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

echo "=== Step 1 DONE: code ready ==="

# === Step 2: 硬件探测 ===
mkdir -p experiment_data/{logs,results}
{
  echo "timestamp: $(date -Iseconds)"
  echo "=== CPU ===" && lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
  echo "=== Memory ===" && free -h
  echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,compute_cap --format=csv,noheader
  echo "=== Topology ===" && nvidia-smi topo -m
  echo "=== NUMA ===" && numactl --hardware 2>/dev/null | head -15
  echo "=== NVLink ===" && nvidia-smi nvlink -s 2>/dev/null || echo "no nvlink"
  echo "=== CUDA ===" && nvcc --version 2>/dev/null | tail -1
  echo "=== Driver ===" && nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
} > experiment_data/results/hardware_$(date +%Y%m%d).txt 2>&1
echo "=== Step 2 DONE: hardware probe ==="

# === Step 3: Joinrenum CPU 测试 (不需要CUDA) ===
cd src/joinrenum
echo "=== Compiling joinrenum tests ==="
PASS=0; FAIL=0
for f in tests/test_bucket_pool.cpp tests/test_count_oracle.cpp \
         tests/test_unordered_map.cpp tests/test_join_tree.cpp \
         tests/test_index.cpp tests/test_rr_access_tree.cpp \
         tests/test_enumerator.cpp tests/test_join_baseline.cpp \
         tests/test_join_triangle.cpp tests/test_renum_baseline.cpp; do
  name=$(basename $f .cpp)
  numactl --cpubind=1 --membind=1 \
    g++ -std=c++17 -O3 -march=znver4 -DAJB_DEBUG -I. $f -lglpk -o /tmp/$name 2>&1 && {
    echo -n "  $name: "
    T0=$(date +%s%N)
    numactl --cpubind=1 --membind=1 timeout 120 /tmp/$name \
      > ../../experiment_data/logs/${name}.txt 2>&1
    RC=$?
    T1=$(date +%s%N)
    MS=$(( (T1 - T0) / 1000000 ))
    [ $RC -eq 0 ] && { echo "PASS (${MS}ms)"; PASS=$((PASS+1)); } || { echo "FAIL rc=$RC (${MS}ms)"; FAIL=$((FAIL+1)); }
  } || { echo "  $name: COMPILE FAIL"; FAIL=$((FAIL+1)); }
done
cd ../..
echo "=== Step 3 DONE: $PASS PASS, $FAIL FAIL ==="

# === Step 4: CUDA 构建 ===
echo "=== Building CUDA targets ==="
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;90" .. 2>&1 | tee ../experiment_data/logs/cmake.log
JOBS=$(( $(nproc) / 4 ))
cmake --build . -- -j${JOBS} 2>&1 | tee ../experiment_data/logs/build.log
BUILD_RC=$?
cd ..
if [ $BUILD_RC -eq 0 ]; then
  echo "=== Step 4 DONE: CUDA build SUCCESS ==="
else
  echo "=== Step 4: CUDA build FAILED (check experiment_data/logs/build.log) ==="
fi

# === Step 5: GPU实验 (如果build成功) ===
if [ $BUILD_RC -eq 0 ] && [ -f build/sort_benchmark ]; then
  echo "=== Running GPU experiments ==="
  bash lab_experiment_runner.sh rq2_cadence
  bash lab_experiment_runner.sh rq3_volume
else
  echo "=== Skipping GPU experiments (build failed or binaries not found) ==="
  echo "=== Check CMakeLists.txt and build logs ==="
fi

# === Step 6: Push 结果 ===
git add experiment_data/
git commit -m "experiment_data: ags1 run $(date +%Y%m%d_%H%M%S)

CPU tests: $PASS/$((PASS+FAIL)) PASS
CUDA build: $([ $BUILD_RC -eq 0 ] && echo SUCCESS || echo FAILED)
Hardware: 2x A6000 + H100 NVL + 2x EPYC 9354"
git remote set-url origin https://YOUR_GITHUB_TOKEN@github.com/dylanyunlon/auerbachs-AJB.git
git push origin main

echo "==========================================="
echo "ALL DONE. Results pushed to GitHub."
echo "Next Claude can: git pull && python3 scripts/debug/parse_lab_results.py"
echo "==========================================="
