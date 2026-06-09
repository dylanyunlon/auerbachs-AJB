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

# ---- Step 1: 在walking3里补装编译依赖 (绝不碰PyTorch/CUDA runtime) ----
echo "[AJB] === Step 1: Build deps in walking3 ==="

# walking3 现状:
#   torch 2.7.1+cu118, nvidia-*-cu11 — 不许动
#   系统 nvcc 11.5 — 支持 sm_80 native, sm_86 通过PTX兼容跑A6000
#   H100 (sm_90) 需要 CUDA 12 — 后续单独处理, 先用2张A6000出数据
#
# 只装: cmake(如果没有), glpk(CPU tests), matplotlib/seaborn(画图)
# 绝不装: cuda-toolkit, nvidia-*, torch, 任何会动cu11链的东西

# conda activate 在 set -e 下容易炸, 先临时关掉
set +eu
# 初始化conda (找到conda的位置, 执行shell hook)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate walking3
# 恢复strict模式
set -euo pipefail

# 验证确实进了walking3
ACTIVE_ENV=$(conda info --envs 2>/dev/null | grep '\*' | awk '{print $1}')
PYTHON_VER=$(python3 --version 2>&1)
TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>&1)
echo "[AJB] Active env: $ACTIVE_ENV"
echo "[AJB] Python: $PYTHON_VER"
echo "[AJB] torch: $TORCH_VER"

# 硬校验: 必须是walking3
if [[ "$TORCH_VER" != *"cu118"* ]]; then
    echo "[AJB] ERROR: 不在walking3里! torch=$TORCH_VER, 期望含cu118"
    echo "[AJB] 请手动执行: conda activate walking3 && bash scripts/ags1_deploy_and_run.sh"
    exit 1
fi

# cmake — 编译需要, 与PyTorch无关
# 注意: cmake 4.x 和 nvcc 11.5 不兼容 (生成的编译命令格式变了)
# 如果系统nvcc < 12, 必须用 cmake <= 3.31
CMAKE_OK=false
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    if [ "${CMAKE_MAJOR:-0}" -le 3 ]; then
        CMAKE_OK=true
        echo "[AJB] cmake $CMAKE_VER — OK for nvcc 11.x"
    else
        echo "[AJB] cmake $CMAKE_VER (v4+) — 与 nvcc 11.x 不兼容, 降级..."
        pip uninstall -y cmake 2>/dev/null || true
        CMAKE_OK=false
    fi
fi
if [ "$CMAKE_OK" = false ]; then
    echo "[AJB] Installing cmake 3.31.6 (最后兼容nvcc 11.x的版本)..."
    pip install 'cmake==3.31.6' 2>&1 | tail -3 || \
    pip install 'cmake<4' 2>&1 | tail -3 || \
    conda install -y 'cmake>=3.18,<4' 2>&1 | tail -3 || true
fi
echo "[AJB] cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"

# matplotlib/seaborn — 画图用, 与PyTorch无关
python3 -c "import matplotlib" 2>/dev/null || {
    echo "[AJB] Installing matplotlib seaborn (不影响PyTorch)..."
    pip install matplotlib seaborn 2>&1 | tail -3 || true
}

# 找nvcc — 用系统的, 不装新的
CURRENT_NVCC=""
for nvcc_path in \
    "$(command -v nvcc 2>/dev/null)" \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-11.5/bin/nvcc \
    /usr/local/cuda-11.8/bin/nvcc \
    /usr/local/cuda-12*/bin/nvcc; do
    if [ -x "$nvcc_path" ] 2>/dev/null; then
        export PATH="$(dirname "$nvcc_path"):$PATH"
        export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
        CURRENT_NVCC=$("$nvcc_path" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
        [ -n "$CURRENT_NVCC" ] && break
    fi
done
echo "[AJB] nvcc: ${CURRENT_NVCC:-not found}"

# 最终验证
nvcc --version | tail -1 || { echo "[AJB] FATAL: nvcc not found"; exit 1; }
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    ARCH_LIST="86;90"
    echo "[AJB] CUDA $CUDA_VER: native sm_86 (A6000) + sm_90 (H100)"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "${CUDA_MINOR:-0}" -ge 5 ]; then
    # CUDA 11.5: sm_80 native, A6000(sm_86)通过PTX兼容跑
    # H100(sm_90)跑不了 — 先用2张A6000出数据
    ARCH_LIST="80"
    echo "[AJB] CUDA $CUDA_VER: sm_80 native, A6000通过PTX兼容"
    echo "[AJB] NOTE: H100 (sm_90) 需要CUDA 12, 本轮先用A6000 x2"
    echo "[AJB] 设 NUM_GPUS=2 (跳过GPU2=H100)"
    export AJB_SKIP_H100=1
else
    ARCH_LIST="80"
    echo "[AJB] WARNING: CUDA $CUDA_VER 较旧, 尝试sm_80编译"
fi

# ---- Step 2: 依赖 ----
echo ""
echo "[AJB] === Step 2: Dependencies ==="

# glpk for joinrenum CPU tests
if ! dpkg -l libglpk-dev &>/dev/null 2>&1; then
    echo "[AJB] libglpk-dev not found, trying conda (不影响PyTorch)..."
    conda install -y -c conda-forge glpk 2>&1 | tail -3 || true
fi

# third_party (git clone each dep)
echo "[AJB] Cloning third_party dependencies..."
mkdir -p third_party
declare -A DEPS=(
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

# Thrust 需要特殊处理: 必须 --recursive 拉 dependencies/libcudacxx 和 dependencies/cub
# 用 1.17.2 tag (最后一个独立Thrust release, 兼容CUDA 11.x)
if [ -d "third_party/thrust" ] && [ -d "third_party/thrust/dependencies/libcudacxx" ]; then
    echo "  ✓ third_party/thrust (with libcudacxx)"
else
    echo "  ↓ cloning thrust --recursive (含libcudacxx+cub submodule)..."
    rm -rf "third_party/thrust"
    git clone --depth=1 --recursive -b 1.17.2 -q \
        https://github.com/NVIDIA/thrust.git third_party/thrust 2>/dev/null || {
        # 如果tag不存在, 试不带tag的recursive clone
        echo "  ↓ retrying thrust main branch --recursive..."
        rm -rf "third_party/thrust"
        git clone --depth=1 --recursive -q \
            https://github.com/NVIDIA/thrust.git third_party/thrust 2>/dev/null || \
            echo "  ✗ FAILED: thrust"
    }
    # 验证libcudacxx存在
    if [ -d "third_party/thrust/dependencies/libcudacxx" ]; then
        echo "  ✓ thrust/dependencies/libcudacxx present"
    else
        echo "  ✗ WARNING: libcudacxx missing after clone, cmake will fail"
    fi
fi

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

# CUDA 11.5: 只用GPU0+GPU1 (A6000x2), 跳过GPU2 (H100)
SMOKE_GPUS=2
[ "${AJB_SKIP_H100:-0}" = "1" ] && {
    export CUDA_VISIBLE_DEVICES=0,1
    SMOKE_GPUS=2
    echo "[AJB] CUDA_VISIBLE_DEVICES=0,1 (跳过H100)"
}

if [ -x build/join_benchmark ]; then
    echo "[AJB] Running upstream join_benchmark..."
    timeout 120 build/join_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus="$SMOKE_GPUS" \
        --gpu-sort=radix \
        --gpu-merge=merge-path \
        2>&1 | tee "$RESULTS_DIR/smoke_join.txt" | tail -15
fi

if [ -x build/ajb_benchmark ]; then
    echo "[AJB] Running AJB benchmark..."
    timeout 120 build/ajb_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus="$SMOKE_GPUS" \
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
