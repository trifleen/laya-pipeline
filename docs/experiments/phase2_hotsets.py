"""Phase 2 prototype: hot-expert files for LLAMA_HOT_EXPERTS, and the speed the model predicts.

    uv run --with numpy python docs/experiments/phase2_hotsets.py logs/phase1 logs/phase2

Hot experts are chosen from the decode routing of the 40 extra Phase 1 prompts (ids not in
eval/cases.jsonl); predictions use the 17 eval prompts, which the benchmark then runs. Each
file is "<layer> <expert> ..." lines plus "s <layer> <expert>" naming the cold stand-in (the
layer's most used cold expert).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase1_analysis import FIXED_MS, N_EXPERTS, N_LAYERS, RAM_MS, counts, load  # noqa: E402

CONFIGS = {  # name: (layers, experts per layer)
    "8layers-k39": (list(range(2, 40, 5)), 39),
    "all-k39": (list(range(N_LAYERS)), 39),
}


def main(trace_dir, out_dir):
    meta, recs = load(trace_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_ids = {json.loads(l)["id"] for l in (Path(__file__).parents[2] / "eval/cases.jsonl").read_text().splitlines() if l.strip()}
    test = {i for i, m in enumerate(meta) if m["id"] in eval_ids}
    train = set(range(len(meta))) - test
    ctr, cte = counts(recs, 1, train), counts(recs, 1, test)

    pred = {"baseline": {"layers": 0, "share_on_gpu": 0.0, "tok_s": round(1000 / (FIXED_MS + RAM_MS), 1)}}
    for name, (layers, k) in CONFIGS.items():
        lines, hit = [], 0.0
        for l in layers:
            order = np.argsort(-ctr[l])
            hot, cold = order[:k], order[k:]
            lines.append(f"{l} " + " ".join(map(str, hot)))
            lines.append(f"s {l} {cold[0]}")
            hit += cte[l, hot].sum() / cte[l].sum()
        share = hit / N_LAYERS  # share of all picks (all 40 layers) served from the GPU
        (out / f"hot-{name}.txt").write_text("\n".join(lines) + "\n")
        pred[name] = {"layers": len(layers), "k": k, "vram_mb": round(len(layers) * k * 465 / 256),
                      "share_on_gpu": round(share, 3), "tok_s": round(1000 / (FIXED_MS + RAM_MS * (1 - share)), 1)}
    (out / "predictions.json").write_text(json.dumps(pred, indent=1))
    print(json.dumps(pred, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
