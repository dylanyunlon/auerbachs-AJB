#!/usr/bin/env bash
# =================================================================
# 子Claude任务分派: 拉取实验数据 → 分析 → 填充tex → push
#
# 用法: bash scripts/dispatch_to_sub_claude.sh [task]
#   task = "analyze"  — 分析实验数据,填充论文
#   task = "build_fix" — 修复编译错误
#   task = "figure"   — 生成论文图表
#
# 前提: claude-hk-config 仓库中有有效的cookie
# =================================================================
set -euo pipefail

AJB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AJB_ROOT"

TASK="${1:-analyze}"
MODEL="claude-opus-4-6"

# ---- 同步claude-hk-config ----
CONFIG_DIR="/tmp/claude-hk-config"
if [ -d "$CONFIG_DIR" ]; then
    git -C "$CONFIG_DIR" pull -q 2>/dev/null || true
else
    git clone --depth=1 -q https://github.com/dylanyunlon/claude-hk-config.git "$CONFIG_DIR" 2>/dev/null || true
fi

# ---- 构造prompt ----
case "$TASK" in
    analyze)
        # 找最新实验结果
        LATEST=$(ls -d experiment_data/2026* 2>/dev/null | sort | tail -1)
        if [ -z "$LATEST" ]; then
            echo "[AJB] No experiment results found"
            exit 1
        fi
        
        PROMPT="你是AJB项目的分析Claude。
git clone https://github.com/dylanyunlon/auerbachs-AJB.git && cd auerbachs-AJB

任务:
1. 查看 $LATEST/ 下的实验CSV文件
2. 运行 python3 scripts/dispatch_experiments.py $LATEST
3. 用实测数据替换 paper/ajb_reconstructed.tex 中的Table 1和Table 2
4. 更新Abstract中的speedup数字
5. git add -A && git commit -m 'paper: fill tables with measured data' && git push

git config: user.name=dylanyunlon email=dogechat@163.com
remote url: use GH_TOKEN env var or ~/.gh_token file

不要改算法代码。只改tex中的数字。直接push到main。"
        ;;
    
    build_fix)
        PROMPT="你是AJB项目的编译修复Claude。
git clone https://github.com/dylanyunlon/auerbachs-AJB.git && cd auerbachs-AJB

服务器有CUDA 12.6, 2x RTX A6000 + 1x H100 NVL。

任务:
1. 查看 experiment_data/logs/ 下最新的build log
2. 分析编译错误
3. 修复src/下的.cuh/.cu文件使其能编译通过
4. 修改算法代码而非字符串替换
5. git commit + push

不要开新分支,不要用v2/port后缀,直接在原文件上改。
git config: user.name=dylanyunlon email=dogechat@163.com"
        ;;
    
    figure)
        PROMPT="你是AJB项目的图表Claude。
git clone https://github.com/dylanyunlon/auerbachs-AJB.git && cd auerbachs-AJB

任务:
1. 从 experiment_data/ 最新目录读取CSV
2. 用matplotlib生成论文需要的Figure 2-7 (PDF格式)
3. 放到 paper/figures/ 目录
4. 更新tex中的includegraphics路径
5. git commit + push

图表风格: NeurIPS 2026标准, 双栏宽度, 字体大小合适。
git config: user.name=dylanyunlon email=dogechat@163.com"
        ;;
    
    *)
        echo "Unknown task: $TASK"
        echo "Usage: $0 {analyze|build_fix|figure}"
        exit 1
        ;;
esac

# ---- 调用 bench_one_model.py ----
OUTPUT_FILE="/tmp/subclaude_${TASK}_$(date +%s).json"

echo "[AJB] Dispatching task '$TASK' to sub-Claude ($MODEL)..."
echo "[AJB] Output: $OUTPUT_FILE"

python3 "$AJB_ROOT/bench_one_model.py" \
    "$MODEL" \
    "dylanyunlon/auerbachs-AJB" \
    "$OUTPUT_FILE" \
    600

echo "[AJB] Sub-Claude response saved to $OUTPUT_FILE"

# ---- 检查结果 ----
if [ -f "$OUTPUT_FILE" ]; then
    python3 -c "
import json
r = json.load(open('$OUTPUT_FILE'))
print(f'Text length: {len(r.get(\"text\",\"\"))}')
print(f'Tool calls: {r.get(\"num_tool_calls\",0)}')
print(f'Elapsed: {r.get(\"elapsed_s\",0)}s')
if r.get('text'):
    print('--- First 500 chars ---')
    print(r['text'][:500])
"
fi
