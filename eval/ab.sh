#!/usr/bin/env bash
# A/B the pipeline speed-ups on eval/cases.jsonl. The llama-server router is restarted before
# each run so no run reuses prompt caches left by the previous one.
set -uo pipefail
cd "$(dirname "$0")/.."
SERVE=${LLAMACPP_SERVE:-$HOME/Work/moe-offload/serve/serve.sh}
restart() {
  pkill -x llama-server; sleep 3
  setsid "$SERVE" >> logs/llama-server.log 2>&1 &
  for _ in $(seq 180); do
    n=$(curl -s localhost:8080/models 2>/dev/null | python3 -c 'import json,sys; print(sum(1 for m in json.load(sys.stdin)["data"] if (m.get("status") or {}).get("value")=="loaded"))' 2>/dev/null)
    [ "$n" = 3 ] && return 0; sleep 1
  done
  echo "router did not come up"; exit 1
}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
restart
LAYA_DEVICE=cpu OVERLAP_ROUTE=0 CONTEXT_IN_USER=0 ADAPTIVE_BUDGET=0 LENGTH_ROUTING=0 \
  uv run laya-pipeline eval --label baseline 2>&1 | grep -v -i warn
base=$(ls -t logs/eval/*-baseline.json | head -1)
restart
uv run laya-pipeline eval --label tuned --compare "$base" 2>&1 | grep -v -i warn
