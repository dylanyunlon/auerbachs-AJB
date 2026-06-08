#!/usr/bin/env bash
# =================================================================
# AJB: 从零编译 (ags1服务器)
#
# 前提: ags1有 nvcc (CUDA 11.5), cmake>=3.18, g++, git
# 硬件: 2x A6000 (sm_86) + 1x H100 NVL (sm_90)
#
# 注意: CUDA 11.5 不支持 sm_86 和 sm_90!
#   sm_86 需要 CUDA 11.1+  → OK
#   sm_90 需要 CUDA 12.0+  → 需要升级CUDA或用PTX fallback
#
# 用法: cd /path/to/auerbachs-AJB && bash scripts/ags1_build_from_scratch.sh
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AJB_ROOT"

echo "[AJB_STATE] === Step 0: System check ==="
nvcc --version 2>/dev/null | tail -1 || { echo "ERROR: nvcc not found"; exit 1; }
cmake --version | head -1
g++ --version | head -1
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
echo ""

# 检测CUDA版本
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "[AJB_STATE] CUDA version: $CUDA_VER"

# 判断支持的架构
# CUDA 11.5: sm_80 max (sm_86通过PTX兼容)
# CUDA 12.0+: sm_90 native
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" -ge 12 ]; then
    ARCH_LIST="80;86;90"
    echo "[AJB_STATE] CUDA 12+: using native sm_80, sm_86, sm_90"
else
    # CUDA 11.x: sm_80 native, sm_86通过PTX兼容运行
    # H100 (sm_90) 不能在CUDA 11.5上编译native, 只能用PTX fallback
    ARCH_LIST="80"
    echo "[AJB_STATE] CUDA 11.x: using sm_80 (A6000 sm_86 通过PTX兼容, H100 sm_90 需升级CUDA)"
    echo "[AJB_BP] WARNING: CUDA $CUDA_VER 不原生支持 sm_86/sm_90"
    echo "  建议: conda install -c nvidia cuda-toolkit=12.6 或 module load cuda/12.x"
fi

echo ""
echo "[AJB_STATE] === Step 1: 初始化 git submodules ==="
git submodule init
git submodule update --depth 1 --recursive
echo ""

echo "[AJB_STATE] === Step 2: 验证 third_party ==="
for dep in thrust moderngpu cxxopts tabulate termcolor csvparser randomdist zipiterator; do
    if [ -d "third_party/$dep" ]; then
        echo "  ✓ third_party/$dep"
    else
        echo "  ✗ third_party/$dep MISSING"
        # 如果submodule失败,手动clone
        case $dep in
            thrust)     git clone --depth 1 https://github.com/NVIDIA/thrust.git third_party/thrust ;;
            moderngpu)  git clone --depth 1 https://github.com/moderngpu/moderngpu.git third_party/moderngpu ;;
            cxxopts)    git clone --depth 1 https://github.com/jarro2783/cxxopts.git third_party/cxxopts ;;
            tabulate)   git clone --depth 1 https://github.com/p-ranav/tabulate.git third_party/tabulate ;;
            termcolor)  git clone --depth 1 https://github.com/ikalnytskyi/termcolor.git third_party/termcolor ;;
            csvparser)  git clone --depth 1 https://github.com/vincentlaucsb/csv-parser.git third_party/csvparser ;;
            randomdist) git clone --depth 1 https://github.com/llersch/cpp_random_distributions.git third_party/randomdist ;;
            zipiterator)git clone --depth 1 https://github.com/dpellegr/ZipIterator.git third_party/zipiterator ;;
        esac
    fi
done

# parallel (gnu parallel sort) — 检查third_party/parallel
if [ ! -d "third_party/parallel" ]; then
    echo "  ✗ third_party/parallel MISSING — creating symlink to system headers"
    mkdir -p third_party/parallel/include
    # gnu parallel sort headers come from libstdc++ — they're already in include path
fi

echo ""
echo "[AJB_STATE] === Step 3: CMake configure ==="
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH_LIST" \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    2>&1 | tee cmake_config.log

echo ""
echo "[AJB_STATE] === Step 4: Build ==="
# 先编译 upstream join_benchmark (验证基础设施)
make -j$(nproc) join_benchmark 2>&1 | tail -20
echo "[AJB_STATE] join_benchmark build: $?"

# 然后编译 ajb_benchmark
make -j$(nproc) ajb_benchmark 2>&1 | tail -20
echo "[AJB_STATE] ajb_benchmark build: $?"

echo ""
echo "[AJB_STATE] === Step 5: 验证二进制 ==="
ls -la join_benchmark ajb_benchmark 2>/dev/null
file join_benchmark ajb_benchmark 2>/dev/null

echo ""
echo "[AJB_STATE] === Step 6: 最小化GPU测试 ==="
# 用最小数据量验证GPU kernel能跑
if [ -x ./join_benchmark ]; then
    echo "[AJB_STATE] Running join_benchmark with 1M elements..."
    timeout 120 ./join_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus=1 \
        --gpu-sort=radix \
        --gpu-merge=merge-path \
        --no-materialize \
        2>&1 | tail -30
    echo "[AJB_STATE] join_benchmark exit: $?"
fi

if [ -x ./ajb_benchmark ]; then
    echo "[AJB_STATE] Running ajb_benchmark with 1M elements..."
    timeout 120 ./ajb_benchmark \
        --num-elements=1000000 \
        --distribution=uniform \
        --num-gpus=1 \
        2>&1 | tail -30
    echo "[AJB_STATE] ajb_benchmark exit: $?"
fi

echo ""
echo "[AJB_STATE] === Build complete ==="
echo "If successful, run full benchmark with:"
echo "  bash scripts/ags1_run_and_push.sh"
