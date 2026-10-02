"""Build report.html (the visual experiment log) from the eval runs and the llama-bench results.

    python docs/experiments/build_report.py BASELINE TUNED MULTI QWEN40 QWEN34

Each argument is a logs/eval/*.json file. Numbers from llama-bench (phase 0) are copied from
bench/results/phase0 below, with the file each one came from.
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load(path):
    return json.loads(Path(path).read_text())


def first_token(rows):
    return {r["id"]: r["time_to_answer_s"] for r in rows}


def total(rows):
    return {r["id"]: r["total_s"] for r in rows}


def summary_row(label, run):
    rows, s = run["rows"], run["summary"]
    ctx = [r for r in rows if r["context_tokens"]]
    nctx = [r for r in rows if not r["context_tokens"]]
    return [label,
            f'{statistics.median(r["time_to_answer_s"] for r in nctx):.2f} s',
            f'{statistics.median(r["time_to_answer_s"] for r in ctx):.2f} s',
            f'{s["total_s_sum"]:.0f} s',
            s["expect_pass"],
            f'{s["p_bad_mean"]:.3f}',
            str(s.get("p_agrees_mean", "–"))]


HYPOTHESES = [
    ("H1", "Offloading the KV cache to RAM gets more out of the 9B", "more context or a bigger model",
     "not needed: compressed prompts are ~2k tokens and the hybrid model's KV cache is small",
     "rejected", "looked at expert offload instead"),
    ("H2", "glm53-flash-offload runs on this laptop", "maybe", "needs 128+ GB RAM, a 24 GB GPU and ~24 cores",
     "rejected", "kept the idea, chose a 22 GB model"),
    ("H3", "A 35B-A3B MoE with all experts in RAM generates at a usable speed", "12–25 tok/s",
     "42.3 tok/s with an empty context, 44 tok/s at 8k", "confirmed", "Phase 0 baseline"),
    ("H4", "Moving expert layers onto the GPU speeds it up a lot", "a large gain",
     "about +0.8 tok/s per layer at ~465 MB each", "weak", "spent VRAM on other models"),
    ("H5", "KV cache and experts compete for VRAM", "yes", "32k context fits with 32 layers in RAM, not 30",
     "confirmed", "context size is part of the placement"),
    ("H6", "Reading the prompt is the bottleneck, not generating", "yes",
     "237–293 tok/s, so a 2k prompt waits ~7 s", "confirmed", "tuned the batch size"),
    ("H7", "Bigger prompt batches read faster", "some gain", "273 → 699 tok/s from ub 512 to 2048",
     "confirmed", "ubatch-size 2048"),
    ("H8", "Pinned RAM (no mmap) copies experts faster", "a large gain", "+3%", "rejected",
     "kept mmap: no 19 GB of locked RAM"),
    ("H9", "Prompt time is linear in length", "yes",
     "no: ~1.5 s fixed per 2,048-token batch, plus ~800 tok/s", "rejected",
     "cost model in routing, budgets sized to batches"),
    ("H10", "The prompt cache works on a hybrid (recurrent) model", "unsure",
     "a repeated 2,203-token prompt: 5.8 s → 0.13 s", "confirmed", "context moved to the last message"),
    ("H11", "Prefilling while Laya routes is worth building", "some gain",
     "at most 0.1 s, the history is already cached", "rejected", "overlapped routing with embedding instead"),
    ("H12", "All three models fit in VRAM together", "unsure", "6.5 GB, all loaded in 9 s", "confirmed",
     "no model swapping"),
    ("H13", "Laya's ~300 ms per decision doesn't matter", "yes",
     "judging 6 chunks took ~5 s on the CPU", "rejected", "moved Laya to the GPU"),
    ("H14", "Laya on the GPU is much faster", "yes",
     "first try silently fell back to the CPU (out of VRAM); loaded as bf16: 18 ms per route, 130 ms for 6 chunks",
     "confirmed", "LAYA_DEVICE=cuda16 with CPU fallback"),
    ("H15", "Keeping only the top 2 chunks by similarity is enough", "yes",
     "2 of 3 lookup answers ranked 3rd–4th", "rejected", "top two-thirds of the budget kept by similarity"),
]


def main(baseline, tuned, multi, q40, q34):
    b, t, m, a, c = (load(p) for p in (baseline, tuned, multi, q40, q34))

    # ---- Result 1
    bf, tf = first_token(b["rows"]), first_token(t["rows"])
    r1_rows = [{"id": r["id"], "baseline": bf[r["id"]], "tuned": tf[r["id"]]} for r in b["rows"]]
    r1_head = ["run", "first token, no doc", "first token, doc", "total", "facts", "p(bad)", "agrees"]
    r1_table = [summary_row("baseline", b), summary_row("tuned", t)]

    # ---- Result 2 (H16)
    runs = [("multi-model", m, "--vram"), ("Qwen only, 40 in RAM", a, "--ram"), ("Qwen only, 34 in RAM", c, "--muted")]
    tot = {name: total(run["rows"]) for name, run, _ in runs}
    keys = ["multi", "qo40", "qo34"]
    r2_rows = [{"id": r["id"], **{k: tot[name][r["id"]] for k, (name, _, _) in zip(keys, runs)}} for r in m["rows"]]
    r2_series = [{"key": k, "name": name, "color": col, "hollow": k == "multi"} for k, (name, _, col) in zip(keys, runs)]
    r2_table = [summary_row(name, run) for name, run, _ in runs]
    def med(run, ctx):
        rows = [r for r in run["rows"] if bool(r["context_tokens"]) == ctx]
        return statistics.median(r["time_to_answer_s"] for r in rows)
    def tot_wo(run, skip="poem"):
        return sum(r["total_s"] for r in run["rows"] if r["id"] != skip)
    verdict = (f"Confirmed on speed. Without one outlier (Qwen spent 181 s on a 450-character poem once and "
               f"didn't repeat it in 4 retries), all 17 requests took {tot_wo(m):.0f} s multi-model against "
               f"{tot_wo(c):.0f} s for the best Qwen-only setup and {tot_wo(a):.0f} s with every expert in RAM. "
               f"The small model wins most on easy requests and long documents: \"Where does Laya run?\" "
               f"{tot['multi-model']['laya-cpu']} s against {tot['Qwen only, 34 in RAM']['laya-cpu']} s. "
               f"Qwen-only answers score a little better on Laya's check (p(bad) {c['summary']['p_bad_mean']:.3f} "
               f"against {m['summary']['p_bad_mean']:.3f}) with the same facts found, and spending Nemotron's VRAM "
               f"on 6 expert layers makes Qwen ~3 tok/s faster. Multi-model stays the default; Qwen-only with 34 "
               f"layers in RAM is the quality option.")
    hyps = [dict(zip(("id", "claim", "expected", "measured", "verdict", "did"), h)) for h in HYPOTHESES]
    hyps.append({"id": "H16", "claim": "The 4B small model earns its ~3 GB of VRAM", "expected": "yes",
                 "measured": f"{tot_wo(m):.0f} s for all 17 requests vs {tot_wo(c):.0f} s for the best Qwen-only setup; Qwen-only answers score slightly better", "verdict": H16_VERDICT, "did": H16_DID})

    data = {
        "headline": [
            ["42–50 tok/s", "35B MoE generating, experts in RAM"],
            [f"{statistics.median(r['time_to_answer_s'] for r in t['rows']):.1f} s",
             f"median first token (was {statistics.median(r['time_to_answer_s'] for r in b['rows']):.1f} s)"],
            [f"{med(t, True):.1f} s", f"first token with a document (was {med(b, True):.1f} s)"],
            [t["summary"]["expect_pass"].replace("/", " / "), f"expected facts found (was {b['summary']['expect_pass'].replace('/', ' / ')})"],
        ],
        "memory": [
            {"label": "VRAM 7.8 GB", "segs": [["Qwen attention + KV", 3.1, "v strong"], ["Nemotron 4B", 3.0, "v"],
                                             ["Laya", 1.0, "v strong"], ["embed", 0.3, "v"], ["free", 0.4, "free"]]},
            {"label": "RAM 30 GB", "segs": [["Qwen experts (page cache)", 19.0, "r strong"], ["desktop + apps", 5.5, "r"],
                                           ["prompt cache", 1.0, "r"], ["free", 4.5, "free"]]},
        ],
        "hypotheses": hyps,
        # results/phase0/ubatch.txt (n-cpu-moe 36, 2,048-token prompt; 512 measured at n-cpu-moe 32)
        "ubatch": [[512, 273], [1024, 421], [2048, 699], [4096, 699]],
        # results/phase0/SUMMARY.md, generation of 128 tokens with an empty context
        "ncmoe": [[30, 49.8], [32, 48.2], [36, 45.5], [40, 42.3]],
        # Laya timings measured 2026-10-02 (scratchpad laya_gpu3.py): CPU fp32 vs GPU bf16
        "laya": [["route, CPU", 443, "--ram"], ["route, GPU", 18, "--vram"],
                 ["6 chunks, CPU", 5089, "--ram"], ["6 chunks, GPU", 130, "--vram"]],
        # results/phase0/loadmode.txt (llama-bench) and llama-server timings for longer prompts
        "batch": [[256, 1.8], [1024, 2.6], [2048, 3.2], [2527, 6.2], [4434, 9.7]],
        "r1": {"rows": r1_rows, "head": r1_head, "table": r1_table},
        "r2": {"intro": ("Nemotron 4B takes ~3 GB of VRAM. Without it, Qwen can keep 6 of its 40 expert layers "
                         "on the GPU. Same 17 requests, three setups, prompt caches cleared between runs: the "
                         "current multi-model setup, Qwen alone with all experts in RAM (isolates the small "
                         "model), and Qwen alone with Nemotron's VRAM spent on experts."),
               "caption": "<b>Total time per request</b> (log scale). Filled dots far right of the hollow ones are requests the small model answers much faster.",
               "rows": r2_rows, "series": r2_series, "head": r1_head, "table": r2_table, "verdict": verdict},
        "sources": [Path(p).name for p in (baseline, tuned, multi, q40, q34)],
    }
    html = (HERE / "report.template.html").read_text().replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    (HERE / "report.html").write_text(html)
    print(verdict)


# Filled in after reading the H16 results (see the experiment log).
H16_VERDICT = "confirmed"
H16_DID = "multi-model stays the default; Qwen-only with 34 layers in RAM is the quality option"

if __name__ == "__main__":
    main(*sys.argv[1:6])
