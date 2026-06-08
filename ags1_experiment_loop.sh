#!/usr/bin/env bash
# ================================================================
# ags1_experiment_loop.sh — 服务器实验 → git push → 子Claude迭代
#
# 在 ags1 上运行:
#   bash ags1_experiment_loop.sh build      # cmake + make
#   bash ags1_experiment_loop.sh test        # 跑joinrenum CPU测试
#   bash ags1_experiment_loop.sh gpu         # 跑GPU benchmark
#   bash ags1_experiment_loop.sh push        # git push实验数据
#   bash ags1_experiment_loop.sh all         # build+test+gpu+push
#
# 子Claude通过 git pull experiment_data/ 获取日志,
# 分析 [AJB_STATE]/[AJB_TIMER]/[AJB_BP] 输出, 修改代码, git push回来
# ================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BUILD_DIR="build"
DATA_DIR="experiment_data"
LOG_DIR="$DATA_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# CUDA架构: A6000=sm_86, H100=sm_90
CUDA_ARCHS="86;90"

mkdir -p "$LOG_DIR"

# ── 工具函数 ──────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
hr()  { echo "════════════════════════════════════════════════════"; }

dump_system_info() {
    local out="$LOG_DIR/system_info_${TIMESTAMP}.txt"
    {
        echo "=== System Info ==="
        date
        uname -a
        echo ""
        echo "=== CPU ==="
        lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\):"
        echo ""
        echo "=== Memory ==="
        free -h
        echo ""
        echo "=== GPU ==="
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu,pcie.link.gen.current --format=csv 2>/dev/null || echo "nvidia-smi failed"
        echo ""
        echo "=== CUDA ==="
        nvcc --version 2>/dev/null || echo "nvcc not found"
        echo ""
        echo "=== Disk ==="
        df -h . | head -2
    } > "$out"
    log "System info → $out"
}

# ── Build ─────────────────────────────────────────────────────
do_build() {
    hr
    log "BUILD: cmake + make"
    hr

    local build_log="$LOG_DIR/build_${TIMESTAMP}.log"

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # cmake configure (支持没有CUDA的纯CPU模式)
    if command -v nvcc &>/dev/null; then
        log "CUDA detected, building full targets"
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
            2>&1 | tee "$SCRIPT_DIR/$build_log"
    else
        log "No CUDA, building joinrenum CPU targets only"
        # 只构建joinrenum CPU目标
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            2>&1 | tee "$SCRIPT_DIR/$build_log"
    fi

    # 编译
    local ncpu=$(nproc 2>/dev/null || echo 8)
    local jobs=$((ncpu > 16 ? 16 : ncpu))
    log "make -j$jobs"

    if command -v nvcc &>/dev/null; then
        make -j"$jobs" 2>&1 | tee -a "$SCRIPT_DIR/$build_log"
    else
        # CPU-only: 只build joinrenum targets
        make -j"$jobs" ajb_joinrenum_all 2>&1 | tee -a "$SCRIPT_DIR/$build_log"
    fi

    cd "$SCRIPT_DIR"

    # 检查编译结果
    local ok=0 fail=0
    for t in tests/ajb_test_*; do
        [ -f "$BUILD_DIR/$t" ] && ok=$((ok+1)) || fail=$((fail+1))
    done
    log "Build result: $ok executables OK, checking..."

    echo "BUILD_TIMESTAMP=$TIMESTAMP" > "$DATA_DIR/build_status.txt"
    echo "BUILD_OK=$ok" >> "$DATA_DIR/build_status.txt"
    echo "BUILD_FAIL=$fail" >> "$DATA_DIR/build_status.txt"
}

# ── CPU Tests (joinrenum) ─────────────────────────────────────
do_test() {
    hr
    log "TEST: joinrenum CPU tests"
    hr

    local test_log="$LOG_DIR/test_cpu_${TIMESTAMP}.log"
    local pass=0 fail=0 total=0

    cd "$SCRIPT_DIR"

    # 找到所有编译好的测试
    for exe in "$BUILD_DIR"/tests/ajb_test_*; do
        [ -x "$exe" ] || continue
        local name=$(basename "$exe")
        total=$((total+1))

        log "  Running: $name"
        local out="$LOG_DIR/${name}_${TIMESTAMP}.log"

        # 在db目录下运行(joinrenum测试需要db/*.csv)
        (cd src/joinrenum && timeout 120 "$SCRIPT_DIR/$exe" 2>&1) > "$out" 2>&1
        local rc=$?

        if [ $rc -eq 0 ]; then
            pass=$((pass+1))
            # 提取关键指标
            local timer=$(grep '\[AJB_TIMER\]' "$out" | tail -1)
            local state=$(grep '\[AJB_STATE\]' "$out" | tail -1)
            log "    PASS ($timer)"
        else
            fail=$((fail+1))
            log "    FAIL (exit=$rc)"
            tail -5 "$out" | while read line; do log "      $line"; done
        fi
    done

    log "CPU tests: PASS=$pass FAIL=$fail TOTAL=$total"

    {
        echo "TEST_TIMESTAMP=$TIMESTAMP"
        echo "TEST_PASS=$pass"
        echo "TEST_FAIL=$fail"
        echo "TEST_TOTAL=$total"
    } > "$DATA_DIR/test_status.txt"

    # 生成CSV摘要
    local csv="$DATA_DIR/test_summary_${TIMESTAMP}.csv"
    echo "test_name,status,wall_sec,hit_rate,tuples,bans" > "$csv"
    for out in "$LOG_DIR"/ajb_test_*_${TIMESTAMP}.log; do
        [ -f "$out" ] || continue
        local name=$(basename "$out" "_${TIMESTAMP}.log")
        local status="PASS"
        grep -q "FAIL\|Error\|error" "$out" && status="FAIL"
        local wall=$(grep -oP 'wall=\K[0-9.]+' "$out" | tail -1)
        local hit=$(grep -oP 'hit_rate=\K[0-9.]+' "$out" | tail -1)
        local tuples=$(grep -oP 'success=\K[0-9]+' "$out" | tail -1)
        local bans=$(grep -oP 'bans=\K[0-9]+' "$out" | tail -1)
        echo "$name,$status,${wall:-0},${hit:-0},${tuples:-0},${bans:-0}" >> "$csv"
    done
    log "Summary CSV → $csv"
}

# ── GPU Benchmarks ────────────────────────────────────────────
do_gpu() {
    hr
    log "GPU: sort/merge/join benchmarks"
    hr

    if ! command -v nvidia-smi &>/dev/null; then
        log "SKIP: no GPU detected"
        return 0
    fi

    local gpu_log="$LOG_DIR/gpu_bench_${TIMESTAMP}.log"

    # Sort benchmark (各GPU)
    for gpu_id in 0 1 2; do
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu_id 2>/dev/null || echo "N/A")
        [ "$gpu_name" = "N/A" ] && continue

        log "  GPU$gpu_id ($gpu_name): sort benchmark"
        if [ -x "$BUILD_DIR/sort_benchmark" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id timeout 300 \
                "$BUILD_DIR/sort_benchmark" --num-elements 1000000 --num-gpus 1 \
                2>&1 | tee -a "$gpu_log" || true
        fi

        log "  GPU$gpu_id ($gpu_name): join benchmark"
        if [ -x "$BUILD_DIR/join_benchmark" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id timeout 300 \
                "$BUILD_DIR/join_benchmark" --r-num-elements 1000000 --s-num-elements 1000000 --num-gpus 1 \
                2>&1 | tee -a "$gpu_log" || true
        fi
    done

    # AJB benchmark (H100)
    if [ -x "$BUILD_DIR/ajb_benchmark" ]; then
        log "  AJB benchmark on H100 (GPU2)"
        CUDA_VISIBLE_DEVICES=2 timeout 600 \
            "$BUILD_DIR/ajb_benchmark" 2>&1 | tee -a "$gpu_log" || true
    fi

    log "GPU logs → $gpu_log"
}

# ── Git Push ──────────────────────────────────────────────────
do_push() {
    hr
    log "PUSH: experiment data → github"
    hr

    cd "$SCRIPT_DIR"

    git add experiment_data/
    git add -u  # 添加修改过的tracked文件

    local msg="experiment_data: ags1 run $TIMESTAMP"
    if [ -f "$DATA_DIR/test_status.txt" ]; then
        source "$DATA_DIR/test_status.txt"
        msg="experiment_data: ${TEST_PASS:-?}/${TEST_TOTAL:-?} pass, run $TIMESTAMP"
    fi

    git commit -m "$msg" || { log "Nothing to commit"; return 0; }
    git push origin main 2>&1 || { log "Push failed — check auth"; return 1; }
    log "Pushed to origin/main"
}

# ── Main ──────────────────────────────────────────────────────
main() {
    local cmd="${1:-help}"

    dump_system_info

    case "$cmd" in
        build)  do_build ;;
        test)   do_test ;;
        gpu)    do_gpu ;;
        push)   do_push ;;
        all)    do_build; do_test; do_gpu; do_push ;;
        help|*)
            echo "Usage: $0 {build|test|gpu|push|all}"
            echo ""
            echo "  build  — cmake configure + make (auto-detect CUDA)"
            echo "  test   — run all joinrenum CPU tests"
            echo "  gpu    — run GPU sort/merge/join benchmarks"
            echo "  push   — git add+commit+push experiment_data/"
            echo "  all    — build + test + gpu + push"
            ;;
    esac
}

main "$@"
