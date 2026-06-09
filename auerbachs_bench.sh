#!/usr/bin/env bash
# =============================================================================
# auerbachs_bench.sh — AJB vs Upstream Baseline Experiment Suite
#
# Runs both the AJB (proposed) and upstream (VLDB'25 Maltry et al.) binaries
# on the same hardware with identical parameters, collects CSV data for the
# paper's Tables 1-2 and Figures 2-7, and auto-pushes results to git.
#
# Usage on ags1:
#   cd /data/jiacheng/system/cache/temp/atc2026/auerbachs-AJB
#   conda activate walking3
#   bash auerbachs_bench.sh [build|sort|join|cpu|all|push]
#
# Hardware: 2x RTX A6000 (sm_86) + 1x H100 NVL (sm_90), CUDA 11.5
# =============================================================================
set -uo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AJB_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
DATA="$AJB_ROOT/experiment_data"
LOG="$DATA/logs"
RES="$DATA/results"
mkdir -p "$LOG" "$RES"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/bench_${TS}.log"; }

# ---- PATH setup ----
for p in /usr/local/cuda/bin /usr/local/cuda-12*/bin /usr/local/cuda-11*/bin; do
  [ -d "$p" ] && export PATH="$p:$PATH"
done

# ---- Conda (reuse walking3) ----
activate_conda() {
  set +eu
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate walking3 2>/dev/null || true
  set -uo pipefail
}

# ---- Hardware probe ----
probe_hw() {
  log "=== Hardware Probe ==="
  {
    date -Iseconds
    echo "--- CPU ---"
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
    echo "--- Memory ---"
    free -h 2>/dev/null || cat /proc/meminfo | head -5
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null
    echo "--- CUDA ---"
    nvcc --version 2>/dev/null | tail -1
    echo "--- Topology ---"
    nvidia-smi topo -m 2>/dev/null | head -15
  } > "$RES/hardware_${TS}.txt" 2>&1
  log "Hardware probe saved"
}

# ---- Build both upstream and AJB ----
do_build() {
  log "=== Building ==="

  # Fetch third_party deps if missing
  if [ ! -d third_party/thrust ]; then
    log "Fetching third_party..."
    mkdir -p third_party
    git clone --depth=1 --recurse-submodules https://github.com/NVIDIA/thrust.git third_party/thrust 2>/dev/null || true
    git clone --depth=1 https://github.com/jarro2783/cxxopts.git third_party/cxxopts 2>/dev/null || true
    git clone --depth=1 https://github.com/moderngpu/moderngpu.git third_party/moderngpu 2>/dev/null || true
    git clone --depth=1 https://github.com/ikalnytskyi/termcolor.git third_party/termcolor 2>/dev/null || true
    git clone --depth=1 https://github.com/vincentlaucsb/csv-parser.git third_party/csvparser 2>/dev/null || true
    git clone --depth=1 https://github.com/p-ranav/tabulate.git third_party/tabulate 2>/dev/null || true
    git clone --depth=1 https://github.com/greg7mdp/parallel-hashmap.git third_party/parallel 2>/dev/null || true
  fi

  # Detect CUDA arch
  CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "11.5")
  CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
  ARCHS="80"
  [ "$CUDA_MAJOR" -ge 12 ] && ARCHS="80;86;90"
  log "CUDA $CUDA_VER -> archs=$ARCHS"
  JOBS=$(( $(nproc) / 4 )); [ "$JOBS" -lt 4 ] && JOBS=4

  # Build AJB
  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" .. \
    > "$LOG/cmake_ajb.log" 2>&1
  cmake --build . -- -j${JOBS} > "$LOG/build_ajb.log" 2>&1
  AJB_RC=$?; cd ..
  log "AJB build: $([ $AJB_RC -eq 0 ] && echo OK || echo FAILED)"

  # Build upstream baseline (share third_party)
  if [ -d upstream/multi-gpu-sort-merge-join ]; then
    [ -L upstream/multi-gpu-sort-merge-join/third_party ] || \
      ln -sf "$AJB_ROOT/third_party" upstream/multi-gpu-sort-merge-join/third_party
    mkdir -p build_upstream && cd build_upstream
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
      ../upstream/multi-gpu-sort-merge-join > "$LOG/cmake_upstream.log" 2>&1
    cmake --build . -- -j${JOBS} > "$LOG/build_upstream.log" 2>&1
    UP_RC=$?; cd ..
    log "Upstream build: $([ $UP_RC -eq 0 ] && echo OK || echo FAILED)"
  fi
}

# ---- Sort benchmark: AJB vs Upstream ----
do_sort() {
  log "=== Sort Benchmark ==="
  local OUT="$RES/sort_comparison_${TS}.csv"
  echo "method,gpu_id,gpu_name,num_elements,distribution,sort_ms,throughput_meps" > "$OUT"

  local SIZES="1000000 10000000 100000000 1000000000"
  local DISTS="uniform zipf sorted staggered"

  for gpu_id in 0 1 2; do
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu_id 2>/dev/null || echo "")
    [ -z "$GPU_NAME" ] && continue
    GPU_NAME=$(echo "$GPU_NAME" | tr ' ' '_')

    for n in $SIZES; do
      for dist in $DISTS; do
        log "  GPU$gpu_id sort n=$n dist=$dist"

        # AJB
        if [ -x build/sort_benchmark ]; then
          RESULT=$(CUDA_VISIBLE_DEVICES=$gpu_id timeout 300 \
            build/sort_benchmark --num_elements $n --gpus 0 \
            --sort_algorithm hybrid_merge_sort \
            --key_distribution $dist --value_distribution uniform \
            2>"$LOG/sort_ajb_g${gpu_id}_${n}_${dist}.log" || echo "")
          SORT_MS=$(echo "$RESULT" | tail -1 | cut -d, -f11 2>/dev/null || echo "")
          [ -n "$SORT_MS" ] && echo "ajb,$gpu_id,$GPU_NAME,$n,$dist,$SORT_MS," >> "$OUT"
        fi

        # Upstream
        if [ -x build_upstream/sort_benchmark ]; then
          RESULT=$(CUDA_VISIBLE_DEVICES=$gpu_id timeout 300 \
            build_upstream/sort_benchmark --num_elements $n --gpus 0 \
            --sort_algorithm hybrid_merge_sort \
            --key_distribution $dist --value_distribution uniform \
            2>"$LOG/sort_up_g${gpu_id}_${n}_${dist}.log" || echo "")
          SORT_MS=$(echo "$RESULT" | tail -1 | cut -d, -f11 2>/dev/null || echo "")
          [ -n "$SORT_MS" ] && echo "upstream,$gpu_id,$GPU_NAME,$n,$dist,$SORT_MS," >> "$OUT"
        fi
      done
    done
  done
  log "Sort results: $OUT ($(wc -l < "$OUT") rows)"
}

# ---- Join benchmark: AJB vs Upstream ----
do_join() {
  log "=== Join Benchmark ==="
  local OUT="$RES/join_comparison_${TS}.csv"
  echo "method,gpu_config,r_elements,s_elements,distribution,join_ms" > "$OUT"

  local CONFIGS="1000000:1000000 10000000:10000000 100000000:100000000 1000000000:7000000000"
  local DISTS="unique_full_key_range unique_partial_key_range"
  local GPU_CONFIGS="0 0,1 0,1,2"

  for gpus in $GPU_CONFIGS; do
    NUM_GPU=$(echo "$gpus" | tr ',' '\n' | wc -l)

    for cfg in $CONFIGS; do
      R=$(echo $cfg | cut -d: -f1)
      S=$(echo $cfg | cut -d: -f2)

      for dist in $DISTS; do
        log "  GPUs=$gpus join R=$R S=$S dist=$dist"

        for method in ajb upstream; do
          BIN="build/join_benchmark"
          [ "$method" = "upstream" ] && BIN="build_upstream/join_benchmark"
          [ -x "$BIN" ] || continue

          RESULT=$(CUDA_VISIBLE_DEVICES=$gpus timeout 600 \
            $BIN --r_num_elements $R --s_num_elements $S \
            --gpus $(seq -s, 0 $((NUM_GPU-1))) \
            --join_algorithm hybrid_sort_merge_join \
            --join_distribution $dist \
            2>"$LOG/join_${method}_g${gpus}_${R}_${dist}.log" || echo "")
          JOIN_MS=$(echo "$RESULT" | tail -1 | awk -F, '{print $(NF-1)}' 2>/dev/null || echo "")
          [ -n "$JOIN_MS" ] && echo "$method,$gpus,$R,$S,$dist,$JOIN_MS" >> "$OUT"
        done
      done
    done
  done
  log "Join results: $OUT ($(wc -l < "$OUT") rows)"
}

# ---- CPU tests (joinrenum) ----
do_cpu() {
  log "=== CPU Tests ==="
  local OUT="$RES/cpu_tests_${TS}.csv"
  echo "test_name,status,time_ms,trace_events" > "$OUT"

  cd src/joinrenum
  for name in test_bucket_pool test_count_oracle test_unordered_map test_join_tree \
              test_index test_rr_access_tree test_enumerator test_join_baseline \
              test_join_triangle test_renum_baseline test_sample_baseline; do
    f="tests/${name}.cpp"
    [ -f "$f" ] || continue
    if g++ -std=c++17 -O2 -DAJB_DEBUG -I. "$f" -lglpk -o "/tmp/$name" 2>/dev/null; then
      T0=$SECONDS
      timeout 120 "/tmp/$name" > "$LOG/${name}_${TS}.txt" 2>&1
      RC=$?
      DUR=$(( (SECONDS - T0) * 1000 ))
      TRACES=$(grep -c "\[AJB_" "$LOG/${name}_${TS}.txt" 2>/dev/null || echo 0)
      STATUS=$([ $RC -eq 0 ] && echo PASS || echo FAIL)
      echo "$name,$STATUS,$DUR,$TRACES" >> "$OUT"
      log "  $STATUS: $name (${DUR}ms, ${TRACES} traces)"
    else
      echo "$name,COMPILE_FAIL,0,0" >> "$OUT"
      log "  COMPILE_FAIL: $name"
    fi
  done
  cd "$AJB_ROOT"
  log "CPU results: $OUT"
}

# ---- Git push ----
do_push() {
  log "=== Git Push ==="
  git config user.name "dylanyunlon"
  git config user.email "dogechat@163.com"
  git add experiment_data/
  git commit -m "experiment_data: bench_${TS}" || true
  GH_TOKEN="${GH_TOKEN:-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}"
  git remote set-url origin "https://${GH_TOKEN}@github.com/dylanyunlon/auerbachs-AJB.git" 2>/dev/null || true
  git push origin main 2>&1 || log "Push failed"
  log "Push done"
}

case "${1:-all}" in
  build) activate_conda; probe_hw; do_build ;;
  sort)  do_sort ;;
  join)  do_join ;;
  cpu)   do_cpu ;;
  push)  do_push ;;
  all)   activate_conda; probe_hw; do_build; do_cpu; do_sort; do_join; do_push ;;
  *)     echo "Usage: $0 [build|sort|join|cpu|all|push]" ;;
esac
log "=== DONE ==="
