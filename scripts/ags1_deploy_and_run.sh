#!/usr/bin/env bash
# =================================================================
# AJB 一键部署: CUDA 12 环境 + 编译 + 实验 + 自动push
#
# 在 ags1 服务器上执行:
#   cd /data/jiacheng/system/cache/temp/atc2026
#   git clone https://github.com/dylanyunlon/auerbachs-AJB.git
#   cd auerbachs-AJB
#   bash scripts/ags1_deploy_and_run.sh
#
# 硬件: 2x RTX A6000 (sm_86) + 1x H100 NVL (sm_90)
# 复用已有 conda base 环境, 安装 CUDA 12.6 toolkit
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AJB_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$AJB_ROOT/experiment_data/logs"
RESULTS_DIR="$AJB_ROOT/experiment_data/${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

exec > >(tee "${LOG_DIR}/deploy_${TIMESTAMP}.log") 2>&1

echo "============================================="
echo "[AJB] Deploy started: $TIMESTAMP"
echo "[AJB] AJB_ROOT=$AJB_ROOT"
echo "============================================="

# ---- Step 0: 系统信息 ----
echo ""
echo "[AJB] === Step 0: System topology ==="
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture" || true
free -h | head -3 || true
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null
nvidia-smi topo -m 2>/dev/null | head -15 || true
echo ""

# ---- Step 1: CUDA 12 环境 ----
echo "[AJB] === Step 1: CUDA 12 toolkit ==="

# 检查当前nvcc版本
CURRENT_NVCC=""
if command -v nvcc &>/dev/null; then
    CURRENT_NVCC=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
fi
echo "[AJB] Current nvcc: ${CURRENT_NVCC:-not found}"

NEED_CUDA12=true
if [[ "$CURRENT_NVCC" == 12.* ]]; then
    echo "[AJB] CUDA 12 already available, skipping install"
    NEED_CUDA12=false
fi

if [ "$NEED_CUDA12" = true ]; then
    echo "[AJB] Installing CUDA 12.6 via conda..."
    
    # 用conda安装CUDA toolkit 12.6 到 base 环境
    # 这不会影响系统驱动(550.144.03 支持 CUDA 12.x)
    conda install -y -c nvidia/label/cuda-12.6.3 cuda-toolkit 2>&1 | tail -10 || {
        echo "[AJB] conda cuda-toolkit install failed, trying pip..."
        pip install nvidia-cuda-toolkit 2>&1 | tail -5 || true
    }
    
    # 验证
    CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)}"
    if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        export PATH="$CONDA_PREFIX/bin:$PATH"
        export CUDA_HOME="$CONDA_PREFIX"
        echo "[AJB] nvcc from conda: $($CONDA_PREFIX/bin/nvcc --version | tail -1)"
    else
        echo "[AJB] Fallback: looking for system CUDA 12..."
        for cuda_dir in /usr/local/cuda-12* /usr/local/cuda; do
            if [ -x "$cuda_dir/bin/nvcc" ]; then
                export PATH="$cuda_dir/bin:$PATH"
                export CUDA_HOME="$cuda_dir"
                echo "[AJB] Found: $cuda_dir/bin/nvcc"
                break
            fi
        done
    fi
fi

# 最终验证
nvcc --version | tail -1 || { echo "[AJB] FATAL: nvcc not found after setup"; exit 1; }
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    ARCH_LIST="86;90"
    echo "[AJB] CUDA $CUDA_VER: native sm_86 (A6000) + sm_90 (H100)"
else
    ARCH_LIST="80"
    echo "[AJB] WARNING: CUDA $CUDA_VER < 12, H100 will NOT work natively"
fi

# ---- Step 2: 依赖 ----
echo ""
echo "[AJB] === Step 2: Dependencies ==="

# Python deps
pip install numpy pandas matplotlib seaborn 2>/dev/null | tail -3 || true

# glpk for joinrenum CPU tests
if ! dpkg -l libglpk-dev &>/dev/null 2>&1; then
    echo "[AJB] libglpk-dev not found, trying conda..."
    conda install -y -c conda-forge glpk 2>/dev/null | tail -3 || true
fi

# third_party (git clone each dep)
echo "[AJB] Cloning third_party dependencies..."
mkdir -p third_party
declare -A DEPS=(
    [thrust]="https://github.com/NVIDIA/thrust.git"
    [moderngpu]="https://github.com/moderngpu/moderngpu.git"
    [cxxopts]="https://github.com/jarro2783/cxxopts.git"
    [tabulate]="https://github.com/p-ranav/tabulate.git"
    [termcolor]="https://github.com/ikalnytskyi/termcolor.git"
    [csvparser]="https://github.com/vincentlaucsb/csv-parser.git"
    [randomdist]="https://github.com/llersch/cpp_random_distributions.git"
    [zipiterator]="https://github.com/dpellegr/ZipIterator.git"
)

for dep in "${!DEPS[@]}"; do
    if [ -d "third_party/$dep" ] && [ "$(ls -A third_party/$dep)" ]; then
        echo "  ✓ third_party/$dep"
    else
        echo "  ↓ cloning $dep..."
        rm -rf "third_party/$dep"
        git clone --depth=1 -q "${DEPS[$dep]}" "third_party/$dep" 2>/dev/null || \
            echo "  ✗ FAILED: $dep"
    fi
done

# parallel header stub
mkdir -p third_party/parallel/include

echo ""

# ---- Step 3: CMake + Build ----
echo "[AJB] === Step 3: Build ==="
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH_LIST" \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
    2>&1 | tee "$RESULTS_DIR/cmake.log" | tail -15

echo ""
echo "[AJB] Building targets..."

# 先编upstream baseline
make -j$(nproc) join_benchmark 2>&1 | tee "$RESULTS_DIR/build_join.log" | tail -10
echo "[AJB] join_benchmark: $([ -x join_benchmark ] && echo OK || echo FAIL)"

# 再编AJB
make -j$(nproc) ajb_benchmark 2>&1 | tee "$RESULTS_DIR/build_ajb.log" | tail -10
echo "[AJB] ajb_benchmark: $([ -x ajb_benchmark ] && echo OK || echo FAIL)"

# sort benchmarks
for target in gpu_sort_benchmark gpu_merge_benchmark sort_benchmark cpu_sort_benchmark cpu_merge_benchmark; do
    make -j$(nproc) $target 2>&1 | tail -3
    echo "[AJB] $target: $([ -x $target ] && echo OK || echo FAIL)"
done

cd "$AJB_ROOT"

# ---- Step 4: 最小GPU测试 ----
echo ""
echo "[AJB] === Step 4: Smoke test (1M elements) ==="

if [ -x build/join_benchmark ]; then
    echo "[AJB] Running upstream join_benchmark..."
    timeout 120 build/join_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus=2 \
        --gpu-sort=radix \
        --gpu-merge=merge-path \
        2>&1 | tee "$RESULTS_DIR/smoke_join.txt" | tail -15
fi

if [ -x build/ajb_benchmark ]; then
    echo "[AJB] Running AJB benchmark..."
    timeout 120 build/ajb_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus=2 \
        2>&1 | tee "$RESULTS_DIR/smoke_ajb.txt" | tail -15
fi

# ---- Step 5: 完整实验 ----
echo ""
echo "[AJB] === Step 5: Full benchmark sweep ==="

if [ -x build/ajb_benchmark ] && [ -x build/join_benchmark ]; then
    bash scripts/ags1_full_experiment.sh "$RESULTS_DIR" 2>&1 | \
        tee "$RESULTS_DIR/experiment.log" | tail -20
else
    echo "[AJB] Binaries not ready, skipping full experiment"
    echo "[AJB] Fix build errors and re-run: bash scripts/ags1_full_experiment.sh"
fi

# ---- Step 6: Push results ----
echo ""
echo "[AJB] === Step 6: Push to GitHub ==="
cd "$AJB_ROOT"
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
GH_TOKEN="${GH_TOKEN:-$(cat ~/.gh_token 2>/dev/null || echo '')}"
if [ -n "$GH_TOKEN" ]; then
    git remote set-url origin "https://${GH_TOKEN}@github.com/dylanyunlon/auerbachs-AJB.git"
fi
git add -A
git commit -m "experiment: ${TIMESTAMP} — GPU benchmark data from ags1 (A6000x2 + H100)" || true
git push origin main 2>&1 || echo "[AJB] push failed"

echo ""
echo "============================================="
echo "[AJB] Deploy complete: $TIMESTAMP"
echo "[AJB] Results: $RESULTS_DIR"
echo "============================================="
