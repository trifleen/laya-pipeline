# Experiment: does the small model earn its VRAM? (H16)

*Branch `experiment/qwen-only` · 2 Oct 2026 · not merged by design: `main` keeps the pipeline, this branch keeps the experiment.*

**Question:** Nemotron 4B takes ~3 GB of VRAM next to Qwen3.6-35B-A3B. Is the pipeline faster with it, or with Qwen alone (which could use those 3 GB for 6 expert layers on the GPU)?

**Result:** confirmed on speed. Multi-model: 172 s for the test set; best Qwen-only setup: 215 s; Qwen alone with every expert in RAM: 261 s. Qwen-only answers scored slightly better on Laya's answer check. Multi-model stays the default; Qwen-only is a preset in `serve/` for quality. Full write-up with figures: section 6 of [`docs/experiments/2026-10-02-moe-offload.md`](docs/experiments/2026-10-02-moe-offload.md).

**On this branch:**
- `eval/experiment-qwen-only.sh`: runs the 17 eval requests on the three setups, restarting the server between runs.
- `eval/results/2026-10-02/`: the raw runs behind the numbers (baseline, multi-model, Qwen-only with 40 and 34 expert layers in RAM).
