#!/usr/bin/env bash
# =============================================================================
# lab_experiment_runner.sh — Run AJB experiments on lab GPU server
#
# Hardware: 2x RTX A6000 (GPU0,GPU1) + 1x H100 NVL (GPU2) + 2x EPYC 9354
# Topology: GPU0-GPU1 NODE (PCIe), GPU0/1-GPU2 NODE (PCIe), H100 PCIe Gen5
#
# Usage: bash lab_experiment_runner.sh [experiment_name]
#   experiments: all | build | rq1_drift | rq2_cadence | rq3_volume |
#                rq4_scale | rq5_skew | rq6_renum | joinrenum_cpu
# =============================================================================
set -euo pipefail

# === Auto-detect CUDA/cmake paths ===
for p in /usr/local/cuda/bin /usr/local/cuda-*/bin $HOME/.local/bin /opt/cmake*/bin; do
  [ -d "$p" ] && export PATH="$p:$PATH"
done
# conda cmake fallback
if ! command -v cmake &>/dev/null && command -v conda &>/dev/null; then
  conda install -y cmake 2>/dev/null || true
fi
# Check g++ version for -march support
GCC_VER=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
if [ "${GCC_VER:-0}" -lt 12 ]; then
  MARCH_FLAG="-march=znver3"
else
  MARCH_FLAG="-march=znver3"
fi


PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJ_DIR}/build"
DATA_DIR="${PROJ_DIR}/experiment_data"
LOG_DIR="${DATA_DIR}/logs"
RESULT_DIR="${DATA_DIR}/results"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Git config for auto-push
git -C "$PROJ_DIR" config user.name "dylanyunlon"
git -C "$PROJ_DIR" config user.email "dogechat@163.com"

timestamp() { date +%Y%m%d_%H%M%S; }
TS=$(timestamp)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/experiment_${TS}.log"; }

# =============================================================================
# Hardware probe — dump full topology for paper Table 1
# =============================================================================
probe_hardware() {
    log "=== HARDWARE PROBE ==="
    local HW="${RESULT_DIR}/hardware_${TS}.txt"
    {
        echo "=== CPU ==="
        lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
        echo ""
        echo "=== Memory ==="
        free -h
        echo ""
        echo "=== GPU ==="
        nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,compute_cap \
            --format=csv,noheader 2>/dev/null
        echo ""
        echo "=== GPU Topology ==="
        nvidia-smi topo -m 2>/dev/null
        echo ""
        echo "=== NUMA ==="
        numactl --hardware 2>/dev/null | head -20
        echo ""
        echo "=== NVLink ==="
        nvidia-smi nvlink -s 2>/dev/null || echo "no nvlink between these GPUs"
        echo ""
        echo "=== PCIe Bandwidth (P2P) ==="
        # If bandwidthTest exists from CUDA samples
        if command -v /usr/local/cuda/extras/demo_suite/bandwidthTest &>/dev/null; then
            /usr/local/cuda/extras/demo_suite/bandwidthTest --device=all 2>/dev/null | tail -20
        fi
        echo ""
        echo "=== Driver/CUDA ==="
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
        nvcc --version 2>/dev/null | tail -1
    } > "$HW" 2>&1
    log "Hardware probe saved: $HW"
}

# =============================================================================
# Build — cmake + make with CUDA
# =============================================================================
build_project() {
    log "=== BUILD ==="
    cd "$PROJ_DIR"

    # Build CUDA targets
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CUDA_ARCHITECTURES="86;90" \
          .. 2>&1 | tee "${LOG_DIR}/cmake_${TS}.log"

    local JOBS=$(( $(nproc) / 4 ))  # conservative for CUDA builds
    [ "$JOBS" -lt 1 ] && JOBS=1
    cmake --build . -- -j${JOBS} 2>&1 | tee "${LOG_DIR}/build_${TS}.log"
    log "Build complete, targets:"
    ls -la "${BUILD_DIR}"/*.exe "${BUILD_DIR}"/ajb_* 2>/dev/null || ls -la "${BUILD_DIR}"/*benchmark* 2>/dev/null
}

# =============================================================================
# Joinrenum CPU tests — compile and run all src/joinrenum tests
# =============================================================================
run_joinrenum_cpu() {
    log "=== JOINRENUM CPU TESTS ==="
    cd "${PROJ_DIR}/src/joinrenum"
    local JLOG="${LOG_DIR}/joinrenum_cpu_${TS}.log"
    local JRESULT="${RESULT_DIR}/joinrenum_cpu_${TS}.csv"

    echo "test_name,status,wall_time_ms,peak_rss_kb" > "$JRESULT"

    for f in tests/test_bucket_pool.cpp tests/test_count_oracle.cpp \
             tests/test_unordered_map.cpp tests/test_join_tree.cpp \
             tests/test_index.cpp tests/test_rr_access_tree.cpp \
             tests/test_enumerator.cpp tests/test_join_baseline.cpp \
             tests/test_join_triangle.cpp tests/test_renum_baseline.cpp \
             tests/test_sample_baseline.cpp; do
        local name=$(basename "$f" .cpp)
        log "  compiling $name..."

        # Compile with NUMA-aware optimization
        numactl --cpubind=1 --membind=1 \
            g++ -std=c++17 -O3 -march=znver3 -DAJB_DEBUG -I. "$f" -lglpk \
            -o "/tmp/${name}" 2>>"$JLOG" && {

            log "  running $name..."
            local T0=$(date +%s%N)
            numactl --cpubind=1 --membind=1 \
                timeout 300 "/tmp/${name}" > "${LOG_DIR}/${name}_${TS}.txt" 2>&1
            local RC=$?
            local T1=$(date +%s%N)
            local WALL_MS=$(( (T1 - T0) / 1000000 ))
            local RSS=$(grep -oP 'peak_RSS=\K[0-9]+' "${LOG_DIR}/${name}_${TS}.txt" 2>/dev/null || echo "0")

            if [ $RC -eq 0 ]; then
                echo "${name},PASS,${WALL_MS},${RSS}" >> "$JRESULT"
                log "  PASS: $name (${WALL_MS}ms)"
            else
                echo "${name},FAIL(rc=$RC),${WALL_MS},${RSS}" >> "$JRESULT"
                log "  FAIL: $name (rc=$RC, ${WALL_MS}ms)"
            fi
        } || {
            echo "${name},COMPILE_FAIL,0,0" >> "$JRESULT"
            log "  COMPILE FAIL: $name"
        }
    done

    log "Results: $JRESULT"
    cat "$JRESULT"
}

# =============================================================================
# RQ1: Drift rates — measure structure staleness over time
# =============================================================================
run_rq1_drift() {
    log "=== RQ1: DRIFT RATES ==="
    cd "$BUILD_DIR"
    local OUT="${RESULT_DIR}/rq1_drift_${TS}.csv"

    # Sort benchmark: measure how build-partition drift changes with K_u
    # GPU0 (A6000) as primary, GPU2 (H100) as secondary
    for K_U in 1 2 4 8 16 32 64 128 256; do
        log "  K_u=$K_U"
        CUDA_VISIBLE_DEVICES=0,2 \
        numactl --cpubind=1 --membind=1 \
            ./sort_benchmark --num_elements=100000000 --num_gpus=2 \
            --ajb_k_u=$K_U --ajb_trace=1 \
            >> "$OUT" 2>&1 || log "  sort_benchmark K_u=$K_U failed"
    done
    log "RQ1 results: $OUT"
}

# =============================================================================
# RQ2: Cadence sweep — K_u vs throughput with fixed K_x/K_v
# =============================================================================
run_rq2_cadence() {
    log "=== RQ2: CADENCE SWEEP ==="
    cd "$BUILD_DIR"
    local OUT="${RESULT_DIR}/rq2_cadence_${TS}.csv"
    echo "k_x,k_u,k_v,throughput_mtuples_s,cross_tier_bytes,wall_ms" > "$OUT"

    for K_U in 1 2 4 8 16 32 64 128 256 512; do
        log "  K_x=16 K_u=$K_U K_v=8"
        CUDA_VISIBLE_DEVICES=0,1,2 \
        numactl --cpubind=1 --membind=1 \
            ./join_benchmark --r_size=50000000 --s_size=200000000 \
            --num_gpus=3 --ajb_k_x=16 --ajb_k_u=$K_U --ajb_k_v=8 \
            --ajb_trace=1 \
            2>&1 | grep -E "AJB_RESULTS|throughput|cross_tier|wall" \
            >> "$OUT" || log "  join K_u=$K_U failed"
    done
    log "RQ2 results: $OUT"
}

# =============================================================================
# RQ3: Volume reduction — AJB vs uniform cadence baseline
# =============================================================================
run_rq3_volume() {
    log "=== RQ3: VOLUME REDUCTION ==="
    cd "$BUILD_DIR"
    local OUT="${RESULT_DIR}/rq3_volume_${TS}.csv"

    # Baseline: uniform cadence (K_x = K_u = K_v = K)
    for K in 1 4 16 64 256; do
        log "  uniform K=$K"
        CUDA_VISIBLE_DEVICES=0,1,2 \
        numactl --cpubind=1 --membind=1 \
            ./ajb_benchmark --r_size=50000000 --s_size=200000000 \
            --num_gpus=3 --ajb_k_x=$K --ajb_k_u=$K --ajb_k_v=$K \
            --ajb_mode=uniform --ajb_trace=1 \
            >> "$OUT" 2>&1 || true
    done

    # AJB: decoupled (K_x=16, K_u=sweep, K_v=8)
    for K_U in 1 4 16 64 256; do
        log "  AJB K_u=$K_U"
        CUDA_VISIBLE_DEVICES=0,1,2 \
        numactl --cpubind=1 --membind=1 \
            ./ajb_benchmark --r_size=50000000 --s_size=200000000 \
            --num_gpus=3 --ajb_k_x=16 --ajb_k_u=$K_U --ajb_k_v=8 \
            --ajb_mode=decoupled --ajb_trace=1 \
            >> "$OUT" 2>&1 || true
    done
    log "RQ3 results: $OUT"
}

# =============================================================================
# RQ4: Scale — billion-tuple joins
# =============================================================================
run_rq4_scale() {
    log "=== RQ4: SCALE ==="
    cd "$BUILD_DIR"
    local OUT="${RESULT_DIR}/rq4_scale_${TS}.csv"

    for R_SIZE in 10000000 50000000 100000000 500000000 1000000000; do
        local S_SIZE=$((R_SIZE * 4))
        log "  R=$R_SIZE S=$S_SIZE"
        CUDA_VISIBLE_DEVICES=0,1,2 \
        numactl --cpubind=1 --membind=1 \
            timeout 600 \
            ./ajb_benchmark --r_size=$R_SIZE --s_size=$S_SIZE \
            --num_gpus=3 --ajb_k_x=16 --ajb_k_u=64 --ajb_k_v=8 \
            --ajb_mode=decoupled --ajb_trace=1 \
            >> "$OUT" 2>&1 || log "  scale R=$R_SIZE timeout/fail"
    done
    log "RQ4 results: $OUT"
}

# =============================================================================
# RQ5: Skew — θ and σ sweep
# =============================================================================
run_rq5_skew() {
    log "=== RQ5: SKEW ==="
    cd "$BUILD_DIR"
    local OUT="${RESULT_DIR}/rq5_skew_${TS}.csv"

    for THETA in 0.0 0.25 0.5 0.75 1.0 1.25 1.5; do
        log "  θ=$THETA"
        CUDA_VISIBLE_DEVICES=0,1,2 \
        numactl --cpubind=1 --membind=1 \
            ./ajb_benchmark --r_size=50000000 --s_size=200000000 \
            --num_gpus=3 --skew_theta=$THETA \
            --ajb_k_x=16 --ajb_k_u=64 --ajb_k_v=8 \
            --ajb_mode=decoupled --ajb_trace=1 \
            >> "$OUT" 2>&1 || true
    done
    log "RQ5 results: $OUT"
}

# =============================================================================
# RQ6: REnum — joinrenum as build phase
# =============================================================================
run_rq6_renum() {
    log "=== RQ6: RENUM ==="
    cd "${PROJ_DIR}/src/joinrenum"
    local OUT="${RESULT_DIR}/rq6_renum_${TS}.csv"

    # Generate larger test data
    log "  generating 100K-point dataset..."
    numactl --cpubind=1 --membind=1 \
        /tmp/gen_co_data 100000 3 1000 "${PROJ_DIR}/src/joinrenum/db/data_100k.txt" \
        2>&1 | tee -a "${LOG_DIR}/rq6_gen_${TS}.log"

    # Run enumerator with different sample sizes
    for SAMPLE in 1000 5000 10000 50000; do
        log "  sample=$SAMPLE"
        numactl --cpubind=1 --membind=1 \
            timeout 120 /tmp/test_enumerator \
            >> "$OUT" 2>&1 || true
    done
    log "RQ6 results: $OUT"
}

# =============================================================================
# Git push results
# =============================================================================
push_results() {
    log "=== PUSH RESULTS ==="
    cd "$PROJ_DIR"
    git add experiment_data/ 2>/dev/null || true
    git commit -m "experiment_data: ${1:-all} run at ${TS} on ags1

Hardware: 2x A6000 + 1x H100 NVL + 2x EPYC 9354 (128 cores)
NUMA: GPU0/1/2 on node1, PCIe Gen1/1/5
Topology: all NODE (PCIe), no NVLink between GPUs" 2>/dev/null || true
    git push origin main 2>/dev/null || log "push failed (check token)"
}

# =============================================================================
# Main dispatcher
# =============================================================================
EXPERIMENT="${1:-all}"
log "Starting experiment: $EXPERIMENT (TS=$TS)"

case "$EXPERIMENT" in
    build)          build_project ;;
    probe)          probe_hardware ;;
    joinrenum_cpu)  run_joinrenum_cpu; push_results joinrenum_cpu ;;
    rq1_drift)      run_rq1_drift; push_results rq1 ;;
    rq2_cadence)    run_rq2_cadence; push_results rq2 ;;
    rq3_volume)     run_rq3_volume; push_results rq3 ;;
    rq4_scale)      run_rq4_scale; push_results rq4 ;;
    rq5_skew)       run_rq5_skew; push_results rq5 ;;
    rq6_renum)      run_rq6_renum; push_results rq6 ;;
    all)
        probe_hardware
        build_project
        run_joinrenum_cpu
        run_rq1_drift
        run_rq2_cadence
        run_rq3_volume
        run_rq4_scale
        run_rq5_skew
        run_rq6_renum
        push_results all
        ;;
    *) echo "Unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

log "=== DONE ==="
