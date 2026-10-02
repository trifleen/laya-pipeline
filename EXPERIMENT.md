# Experiment: hot experts on the GPU (Phase 2 prototype, H21–H22) — dropped

*Branch `experiment/hot-experts` · 2 Oct 2026 · built on [`experiment/expert-routing`](https://github.com/trifleen/laya-pipeline/tree/experiment/expert-routing). Kept as a documented dead end; not merged and not developed further.*

**Idea:** Phase 1 showed that 39 "hot" experts per layer catch 35% of Qwen's routing picks. Copy those to the GPU, compute the rest from RAM as before, and generation should get ~22% faster (predicted 48.5 tok/s against 39.7).

**What we built:** a ~300-line llama.cpp patch (`tools/hot-experts/llama.cpp-hot-experts.patch`, for llama.cpp `4ebdf2c`). With `LLAMA_HOT_EXPERTS=<file>`, it copies the listed experts into a GPU buffer and splits each token's 8 picks: hot ones on the GPU copy, cold ones from RAM. Without the variable it does nothing.

**Result: no speed-up.** 39.8 tok/s with hot experts in all 40 layers, against 39.7 without. llama.cpp runs each layer's GPU part and CPU part one after the other, never side by side, so the extra GPU work per layer costs about what the cache saves. The idea only pays if hot and cold experts run *in parallel*, a much bigger change to llama.cpp's scheduler. So the idea was dropped.

**Two bugs worth knowing about if you try something similar:**
- llama.cpp's GPU expert kernel (MMQ) assumes a token lists each expert at most once. Pointing unused slots at one shared stand-in overflowed its buffers ("illegal memory access"); each slot now gets its own stand-in.
- Computing the hot/cold ids on the CPU added two GPU↔CPU handoffs per layer (~0.14 ms each per token); keeping those lookups on the GPU removed the loss, but still no gain.

Full write-up with the predicted-vs-measured figure: section 8 of [`docs/experiments/2026-10-02-moe-offload.md`](docs/experiments/2026-10-02-moe-offload.md).

**On this branch:**
- `tools/hot-experts/llama.cpp-hot-experts.patch`: the llama.cpp change.
- `docs/experiments/phase2_hotsets.py`: chooses hot experts on the 40 extra Phase 1 prompts and records predictions *before* measuring.
- `docs/experiments/phase2_bench.sh`: benchmarks on the 17 eval prompts; `phase2_figure.py` makes the figure.
- `docs/experiments/data/phase2/`: hot-expert lists, predictions and every benchmark run, including the first version with CPU-side ids (`*-cpu-tables.json`).
