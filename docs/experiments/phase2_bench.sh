#!/usr/bin/env bash
# Phase 2 prototype benchmark: Qwen alone in llama-server, every expert in RAM, with and without
# a hot-expert copy on the GPU (LLAMA_HOT_EXPERTS, patched llama.cpp). Sends the 17 eval
# prompts (greedy, 192 tokens max, no prompt cache) and writes logs/phase2/bench-<config>.json.
#   docs/experiments/phase2_bench.sh baseline 8layers-k39 all-k39
set -uo pipefail
cd "$(dirname "$0")/../.."
L=${LLAMA_CPP:-$HOME/Work/moe-offload/llama.cpp}
MODEL=${MODEL:-$HOME/Work/moe-offload/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
pkill -x llama-server; sleep 3
for cfg in "$@"; do
  hot=""; [ "$cfg" != baseline ] && hot="logs/phase2/hot-$cfg.txt"
  LLAMA_HOT_EXPERTS="$hot" "$L/build/bin/llama-server" -m "$MODEL" -ngl 99 --n-cpu-moe 40 -fa on -t 8 \
    -c 16384 -b 2048 -ub 2048 -np 1 --port 18090 > "logs/phase2/server-$cfg.log" 2>&1 &
  pid=$!
  for _ in $(seq 180); do curl -sf localhost:18090/health >/dev/null && break; sleep 1; done
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  python3 - "$cfg" "$vram" <<'PY'
import json, sys, urllib.request
cfg, vram = sys.argv[1], int(sys.argv[2])
ids = {json.loads(l)["id"] for l in open("eval/cases.jsonl") if l.strip()}
meta = [json.loads(l) for l in open("logs/phase1/meta.jsonl")]
texts = [json.loads(l)["text"] for l in open("logs/phase1/prompts.jsonl")]
rows = []
for m, text in zip(meta, texts):
    if m["id"] not in ids:
        continue
    body = {"prompt": text, "n_predict": 192, "temperature": 0, "cache_prompt": False}
    r = urllib.request.Request("http://localhost:18090/completion", json.dumps(body).encode(), {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    t = d["timings"]
    rows.append({"id": m["id"], "text": d["content"], "predicted_n": t["predicted_n"], "predicted_ms": t["predicted_ms"],
                 "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"]})
gen_n = sum(r["predicted_n"] for r in rows); gen_ms = sum(r["predicted_ms"] for r in rows)
pn = sum(r["prompt_n"] for r in rows); pms = sum(r["prompt_ms"] for r in rows)
out = {"config": cfg, "vram_mb": vram, "gen_tokens": gen_n, "gen_tok_s": round(gen_n / gen_ms * 1000, 2),
       "prompt_tok_s": round(pn / pms * 1000, 1), "rows": rows}
json.dump(out, open(f"logs/phase2/bench-{cfg}.json", "w"), indent=1)
print(f"{cfg:14} vram {vram} MiB  generation {out['gen_tok_s']} tok/s ({gen_n} tokens)  prompt {out['prompt_tok_s']} tok/s")
PY
  kill $pid; wait $pid 2>/dev/null
done
