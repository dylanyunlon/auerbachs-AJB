#!/usr/bin/env bash
# =============================================================================
# ags1_quickstart.sh — 实验室一键执行: 编译 + CPU测试 + GPU构建 + push
# 不使用conda/pip, 纯系统级依赖 (cmake, g++, nvcc, libglpk-dev, libboost-dev)
# =============================================================================
set -uo pipefail

log() { echo "[$(date +%H:%M:%S)] $*"; }

# === CUDA/cmake PATH自动探测 ===
for p in /usr/local/cuda/bin /usr/local/cuda-12*/bin /usr/local/cuda-11*/bin \
         /opt/cmake*/bin /snap/bin $HOME/.local/bin; do
  [ -d "$p" ] && export PATH="$p:$PATH"
done
command -v module &>/dev/null && { module load cmake 2>/dev/null; module load cuda 2>/dev/null; } || true

WORK="${WORK:-$(pwd)}"
cd "$WORK"

# === Step 1: 拉取最新代码 ===
log "Step 1: git pull"
if [ -d auerbachs-AJB ]; then
  cd auerbachs-AJB && git pull origin main 2>&1 | tail -3
else
  git clone https://github.com/dylanyunlon/auerbachs-AJB.git && cd auerbachs-AJB
fi
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
log "Step 1 DONE"

# === Step 2: 硬件探测 ===
log "Step 2: hardware probe"
mkdir -p experiment_data/{logs,results}
{
  date -Iseconds
  echo "=== CPU ===" && lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
  echo "=== Memory ===" && free -h 2>/dev/null || cat /proc/meminfo | head -3
  echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "no GPU"
  echo "=== CUDA ===" && nvcc --version 2>/dev/null | tail -1 || echo "no nvcc"
  echo "=== Driver ===" && nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "no driver"
} > experiment_data/results/hardware_$(date +%Y%m%d).txt 2>&1
log "Step 2 DONE"

# === Step 3: Joinrenum CPU 测试 ===
log "Step 3: CPU tests (joinrenum)"
cd src/joinrenum
PASS=0; FAIL=0
TESTS="test_bucket_pool test_count_oracle test_unordered_map test_join_tree
       test_index test_rr_access_tree test_enumerator test_join_baseline
       test_join_triangle test_renum_baseline test_sample_baseline"
for name in $TESTS; do
  f="tests/${name}.cpp"
  [ -f "$f" ] || { log "  SKIP: $f not found"; continue; }
  if g++ -std=c++17 -O2 -I. "$f" -lglpk -o "/tmp/$name" 2>"/tmp/${name}_compile.log"; then
    T0=$(date +%s%N)
    timeout 120 "/tmp/$name" > "../../experiment_data/logs/${name}.txt" 2>&1
    RC=$?
    T1=$(date +%s%N)
    MS=$(( (T1 - T0) / 1000000 ))
    if [ $RC -eq 0 ]; then
      log "  PASS: $name (${MS}ms)"
      PASS=$((PASS+1))
    else
      log "  FAIL: $name rc=$RC (${MS}ms)"
      FAIL=$((FAIL+1))
    fi
  else
    log "  COMPILE FAIL: $name"
    cat "/tmp/${name}_compile.log" | head -3
    FAIL=$((FAIL+1))
  fi
done
cd ../..
log "Step 3 DONE: $PASS PASS, $FAIL FAIL"

# === Step 4: CUDA 构建 ===
log "Step 4: CUDA build"

# 自动获取 third_party 依赖 (如果不存在)
if [ ! -d third_party/thrust ]; then
  log "  Fetching third_party dependencies..."
  mkdir -p third_party
  # Thrust (NVIDIA, includes CUB)
  [ -d third_party/thrust ] || git clone --depth=1 https://github.com/NVIDIA/thrust.git third_party/thrust 2>/dev/null || true
  # cxxopts (CLI parsing)
  [ -d third_party/cxxopts ] || git clone --depth=1 https://github.com/jarro2783/cxxopts.git third_party/cxxopts 2>/dev/null || true
  # moderngpu (GPU primitives)
  [ -d third_party/moderngpu ] || git clone --depth=1 https://github.com/moderngpu/moderngpu.git third_party/moderngpu 2>/dev/null || true
  # termcolor (colored output)
  [ -d third_party/termcolor ] || git clone --depth=1 https://github.com/ikalnytskyi/termcolor.git third_party/termcolor 2>/dev/null || true
  # csv-parser
  [ -d third_party/csvparser ] || git clone --depth=1 https://github.com/vincentlaucsb/csv-parser.git third_party/csvparser 2>/dev/null || true
  # tabulate (table formatting)
  [ -d third_party/tabulate ] || git clone --depth=1 https://github.com/p-ranav/tabulate.git third_party/tabulate 2>/dev/null || true
  # parallel-hashmap
  [ -d third_party/parallel ] || git clone --depth=1 https://github.com/greg7mdp/parallel-hashmap.git third_party/parallel 2>/dev/null || true
  log "  third_party dependencies fetched"
fi

BUILD_RC=1
if command -v cmake &>/dev/null && command -v nvcc &>/dev/null; then
  # 自动检测CUDA版本,选择arch
  CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0.0")
  CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
  log "  CUDA $CUDA_VER detected"

  # sm_86 需要 CUDA>=11.1, sm_90 需要 CUDA>=11.8
  ARCHS="86"
  if [ "$CUDA_MAJOR" -gt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    ARCHS="86;90"
    log "  Using archs: sm_86 + sm_90 (H100 supported)"
  else
    log "  Using arch: sm_86 only (CUDA $CUDA_VER < 11.8, no sm_90/H100)"
  fi

  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" .. \
    2>&1 | tee ../experiment_data/logs/cmake.log
  JOBS=$(( $(nproc) / 4 ))
  [ "$JOBS" -lt 4 ] && JOBS=4
  cmake --build . -- -j${JOBS} 2>&1 | tee ../experiment_data/logs/build.log
  BUILD_RC=$?
  cd ..
  [ $BUILD_RC -eq 0 ] && log "Step 4 DONE: build SUCCESS" || log "Step 4 FAILED (see logs)"
else
  log "Step 4 SKIPPED: cmake=$(command -v cmake || echo MISSING) nvcc=$(command -v nvcc || echo MISSING)"
  log "  On ags1: export PATH=/usr/local/cuda/bin:\$PATH"
fi

# === Step 5: GPU实验 ===
if [ $BUILD_RC -eq 0 ] && [ -x build/sort_benchmark ]; then
  log "Step 5: GPU experiments"
  for gpu_id in 0 1 2; do
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu_id 2>/dev/null || echo "")
    [ -z "$gpu_name" ] && continue
    log "  GPU$gpu_id ($gpu_name): sort 1M elements"
    CUDA_VISIBLE_DEVICES=$gpu_id timeout 120 \
      build/sort_benchmark --num-elements 1000000 --num-gpus 1 \
      > "experiment_data/logs/sort_gpu${gpu_id}.txt" 2>&1 || true
    log "  GPU$gpu_id ($gpu_name): join 1Mx1M"
    CUDA_VISIBLE_DEVICES=$gpu_id timeout 120 \
      build/join_benchmark --r-num-elements 1000000 --s-num-elements 1000000 --num-gpus 1 \
      > "experiment_data/logs/join_gpu${gpu_id}.txt" 2>&1 || true
  done
  log "Step 5 DONE"
else
  log "Step 5 SKIPPED (no GPU binaries)"
fi

# === Step 6: Push 结果 ===
log "Step 6: git push"
git add experiment_data/
git commit -m "experiment_data: ags1 $(date +%Y%m%d_%H%M%S) — CPU $PASS/$((PASS+FAIL)) PASS, CUDA build=$([ $BUILD_RC -eq 0 ] && echo OK || echo FAIL)" || true
if [ -n "${GH_TOKEN:-}" ]; then
  git remote set-url origin "https://${GH_TOKEN}@github.com/dylanyunlon/auerbachs-AJB.git"
fi
git push origin main 2>&1 || log "Push failed — set GH_TOKEN env"

log "ALL DONE"
echo "============================================"
echo "CPU tests: $PASS/$((PASS+FAIL)) PASS"
echo "CUDA build: $([ $BUILD_RC -eq 0 ] && echo SUCCESS || echo FAILED)"
echo "Next: git pull experiment_data/ to check results"
echo "============================================"
