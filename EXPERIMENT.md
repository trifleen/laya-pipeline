# Experiment: which experts does Qwen actually use? (Phase 1, H17–H20)

*Branch `experiment/expert-routing` · 2 Oct 2026 · not merged by design: `main` keeps the pipeline, this branch keeps the experiment.*

**Question:** each of Qwen3.6-35B-A3B's 40 layers routes every token to 8 of 256 experts. Is that routing concentrated enough that a few "hot" experts per layer could live on the GPU?

**Result:** moderately. 39 hot experts per layer catch 35% of picks on prompts not used to choose them (an even spread would catch 15%). Code and creative writing have their own hot experts; a prompt batch touches 232 of 256 experts, so caching can't speed up prompt reading; a dynamic LRU cache is limited by PCIe bandwidth. Estimated +15–25% generation speed with ~3 GB of VRAM. That estimate was tested, and failed, on [`experiment/hot-experts`](https://github.com/trifleen/laya-pipeline/tree/experiment/hot-experts). Full write-up with figures: section 7 of [`docs/experiments/2026-10-02-moe-offload.md`](docs/experiments/2026-10-02-moe-offload.md).

**On this branch:**
- `tools/expert-trace/`: a small C++ tool on llama.cpp's API that records every token's 8 experts per layer (`build.sh`, `expert_trace.cpp`), and `make_prompts.py`, which builds the 57-prompt set labelled with Laya's task.
- `docs/experiments/phase1_analysis.py` → `phase1-results.json` and the Phase 1 figures.
- `docs/experiments/data/phase1/meta.jsonl`: the prompts and their task labels. The 11 MB trace isn't committed; the tool regenerates it in ~8 minutes.
