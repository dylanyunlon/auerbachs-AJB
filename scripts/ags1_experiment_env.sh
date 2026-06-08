#!/usr/bin/env bash
# =================================================================
# AJB Experiment Environment Setup (ags1 server)
# Target: 2x RTX A6000 + 1x H100 NVL, 2x AMD EPYC 9354, 1.5TiB RAM
#
# 用法: source scripts/ags1_experiment_env.sh
# 复用已有conda环境, 安装CUDA编译依赖
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[AJB_STATE] AJB_ROOT=$AJB_ROOT"

# ---- 系统信息采集 ----
echo "[AJB_STATE] === System Topology ==="
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture" || true
free -h | head -3
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
nvidia-smi topo -m 2>/dev/null || echo "topo not available"
nvidia-smi nvlink -s 2>/dev/null || echo "no nvlink"
uname -r
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"

# ---- Conda环境 ----
# 复用服务器已有conda环境 (如果有的话)
if command -v conda &>/dev/null; then
    echo "[AJB_STATE] conda found: $(conda --version)"
    # 检查是否已有ajb环境
    if conda env list | grep -q "ajb"; then
        echo "[AJB_STATE] activating existing 'ajb' conda env"
        conda activate ajb
    else
        echo "[AJB_STATE] creating 'ajb' conda env"
        conda create -n ajb python=3.11 -y
        conda activate ajb
        pip install numpy pandas matplotlib seaborn
    fi
else
    echo "[AJB_STATE] no conda, using system Python"
fi

# ---- CUDA编译依赖检查 ----
echo "[AJB_STATE] === Build Dependencies ==="
for cmd in nvcc g++ cmake make; do
    if command -v $cmd &>/dev/null; then
        echo "[AJB_STATE] $cmd: $(which $cmd)"
    else
        echo "[AJB_BP] MISSING: $cmd"
    fi
done

# ---- moderngpu检查 ----
if [ -d "${AJB_ROOT}/third_party/moderngpu" ]; then
    echo "[AJB_STATE] moderngpu: found"
else
    echo "[AJB_BP] moderngpu not found, cloning..."
    mkdir -p "${AJB_ROOT}/third_party"
    git clone --depth=1 https://github.com/moderngpu/moderngpu.git "${AJB_ROOT}/third_party/moderngpu"
fi

echo "[AJB_STATE] environment setup complete"
