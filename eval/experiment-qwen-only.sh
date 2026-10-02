#!/usr/bin/env bash
# Experiment 2026-10-02: does the small model earn its VRAM? Same 17 requests, three setups:
#   multi           Nemotron 4B + Qwen3.6-35B-A3B (all experts in RAM), Laya routes between them
#   qwen-only-40    every request to Qwen, all experts in RAM (isolates the small-model path)
#   qwen-only-34    every request to Qwen, Nemotron's VRAM spent on 6 expert layers on the GPU
# The router is restarted before each run, so no run reuses another's prompt cache.
set -uo pipefail
cd "$(dirname "$0")/.."
SERVE=${LLAMACPP_SERVE:-serve/serve.sh}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
restart() {  # env for serve.sh in "$@"; waits until $1 models are loaded
  local want=$1; shift
  pkill -x llama-server; sleep 3
  env "$@" setsid "$SERVE" >> logs/llama-server.log 2>&1 &
  for _ in $(seq 180); do
    n=$(curl -s localhost:8080/models 2>/dev/null | python3 -c 'import json,sys; print(sum(1 for m in json.load(sys.stdin)["data"] if (m.get("status") or {}).get("value")=="loaded"))' 2>/dev/null)
    [ "$n" = "$want" ] && return 0; sleep 1
  done
  echo "router did not come up"; exit 1
}
run() { uv run laya-pipeline eval "$@" 2>&1 | grep -v -i warn; }

restart 3 PRESET=
run --label multi
multi=$(ls -t logs/eval/*-multi.json | head -1)
restart 2 PRESET=qwen-only N_CPU_MOE=40
SMALL_MODEL=qwen3.6-35b-a3b run --label qwen-only-40 --compare "$multi"
restart 2 PRESET=qwen-only N_CPU_MOE=34
SMALL_MODEL=qwen3.6-35b-a3b run --label qwen-only-34 --compare "$multi"
restart 3 PRESET=  # back to the default setup
