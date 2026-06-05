#!/usr/bin/env bash
# ============================================================
# dispatch_to_sub_claude.sh — 向子Claude Opus 4.6发送改写任务
# 第一位Claude调度使用
# ============================================================
set -euo pipefail

ORG=$(cat /tmp/claude_hk_org.txt)
BASE="https://claude.hk.cn/api/organizations/${ORG}"
MODEL="claude-opus-4-6"
EFFORT="high"
THINKING="extended"
TIMEOUT="${TIMEOUT:-600}"
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
COOKIES=$(cat /tmp/claude_hk_cookie.txt)

CONV_ID="$1"
PROMPT_FILE="$2"

if [ -z "$CONV_ID" ] || [ -z "$PROMPT_FILE" ]; then
    echo "Usage: $0 <conv_id> <prompt_file>"
    exit 1
fi

PROMPT=$(cat "$PROMPT_FILE")

H_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
A_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

# 构建payload
python3 << PYEOF
import json

prompt = open("$PROMPT_FILE").read()

payload = {
    "prompt": prompt,
    "timezone": "Asia/Shanghai",
    "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
        "nameKey":"normal_style_name","prompt":"Normal\n",
        "summary":"Default responses from Claude",
        "summaryKey":"normal_style_summary","isDefault":True}],
    "locale": "en-US", "model": "${MODEL}", "effort": "${EFFORT}",
    "thinking_mode": "${THINKING}",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {"human_message_uuid": "${H_UUID}", "assistant_message_uuid": "${A_UUID}"},
    "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    "create_conversation_params":{"name":"","model":"${MODEL}",
        "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
        "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
}
open("/tmp/_dispatch_payload.json","w").write(json.dumps(payload))
PYEOF

echo "[DISPATCH] Sending to Opus 4.6, conv=$CONV_ID"
echo "[DISPATCH] Prompt: $(wc -c < "$PROMPT_FILE") bytes"
echo "[DISPATCH] Model: $MODEL, effort: $EFFORT, thinking: $THINKING"

RAW=$(curl -sf --max-time "$TIMEOUT" \
    "${BASE}/chat_conversations/${CONV_ID}/completion" \
    -H 'accept: text/event-stream' -H 'content-type: application/json' \
    -H 'anthropic-client-platform: web_claude_ai' \
    -b "$COOKIES" -H 'origin: https://claude.hk.cn' \
    -H 'referer: https://claude.hk.cn/new' -H "user-agent: $UA" \
    -d @/tmp/_dispatch_payload.json 2>&1)

# 保存原始响应
echo "$RAW" > /tmp/sub_claude_raw_response.txt

# 解析响应
echo "$RAW" | python3 << 'PYEOF'
import sys, json
text_parts, tool_calls = [], []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    try: d = json.loads(line[6:])
    except: continue
    t = d.get("type", "")
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta": text_parts.append(delta["text"])
        elif delta.get("type") == "tool_use_block_update_delta":
            dc = delta.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    info = json.loads(dc["json_block"])
                    if info.get("code"): tool_calls.append(f'[{info.get("language","")}] {info["code"][:200]}')
                except: pass
    elif t == "content_block_start":
        cb = d.get("content_block", {})
        if cb.get("type") == "tool_result":
            dc = cb.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    r = json.loads(dc["json_block"])
                    stdout = r.get("stdout", "")
                    if stdout: tool_calls.append(f'[stdout] {stdout[:300]}')
                except: pass

if tool_calls:
    print(f"\033[90m  [{len(tool_calls)} tool operations]\033[0m")
    for tc in tool_calls[:5]:
        print(f"\033[90m  > {tc[:200]}\033[0m")
text = "".join(text_parts)
if text.strip():
    print(f"\033[36mSub-Claude:\033[0m")
    # 只打前2000字
    if len(text) > 2000:
        print(text[:2000])
        print(f"\n... [{len(text)} chars total, truncated]")
    else:
        print(text)
print()
PYEOF

echo "[DISPATCH] Response saved to /tmp/sub_claude_raw_response.txt"
