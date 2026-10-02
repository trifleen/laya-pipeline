#!/usr/bin/env bash
# Phase 0: sweep --n-cpu-moe (how many layers keep their experts in RAM) with stock llama.cpp.
# Prompt processing at 512/4096 tokens and generation of 128 tokens, flash attention on, all
# non-expert weights on the GPU. Writes JSONL to results/phase0/.
#
#   bench/sweep.sh [n-cpu-moe values, comma separated]   e.g. bench/sweep.sh 40,36,32,30,28
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-${MODELS:-$HOME/Work/moe-offload/models}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
NCMOE=${1:-40,36,32,30,28,26}
THREADS=${THREADS:-8}  # physical cores; SMT threads don't help memory-bound decode
OUT=bench/results/phase0/sweep-$(date +%Y%m%d-%H%M%S).jsonl
mkdir -p bench/results/phase0

# Snapshot of the conditions, so numbers stay comparable later.
{
  echo "# $(date -Is)  model=$MODEL  threads=$THREADS  llama.cpp=$(git -C "${LLAMA_CPP:-$HOME/Work/moe-offload/llama.cpp}" rev-parse --short HEAD)"
  free -m | sed 's/^/# /'
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/# vram before: /'
} | tee "$OUT.meta"

# Each --n-cpu-moe value runs as its own process so a value that doesn't fit in VRAM fails
# alone instead of ending the sweep.
for n in ${NCMOE//,/ }; do
  echo "== n-cpu-moe $n" >&2
  "${LLAMA_CPP:-$HOME/Work/moe-offload/llama.cpp}/build/bin/llama-bench" -m "$MODEL" -ngl 99 -fa on -t "$THREADS" \
      --n-cpu-moe "$n" -p 512,4096 -n 128 -r 3 -o jsonl >> "$OUT" \
    || echo "{\"n_cpu_moe\": $n, \"error\": \"failed (likely out of VRAM)\"}" >> "$OUT"
done
echo "$OUT"
