"""Phase 1: how concentrated is Qwen3.6-35B-A3B's expert routing, and would a GPU cache of hot experts pay?

    uv run --with matplotlib --with numpy python docs/experiments/phase1_analysis.py logs/phase1

Reads trace.bin (tools/expert-trace) and meta.jsonl (tools/expert-trace/make_prompts.py).
Writes phase1-results.json and figures img/phase1-*-{light,dark}.svg next to this script.

Speed model (fitted to results/phase0, n-cpu-moe 40/36/32/30 → 42.3/45.5/48.2/49.8 tok/s):
    ms per generated token = 9.6 + 14.0 × (share of expert picks computed from RAM)
It reproduces the four measurements within 1%. Each expert is 465 MB / 256 = 1.82 MB.
"""

import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N_LAYERS, N_EXPERTS, K_USED = 40, 256, 8
EXPERT_MB = 465 / 256
FIXED_MS, RAM_MS = 9.6, 14.0
# A dynamic cache must copy every expert it admits over PCIe 3.0 (~12 GB/s): 0.148 ms per expert.
ADMIT_MS = EXPERT_MB / 1024 / 12 * 1000
# VRAM that could hold experts: what's free next to all three models, and what dropping
# Nemotron would free (its 3.0 GB, minus headroom for Laya's peaks).
BUDGETS_GB = {"free today (0.4 GB)": 0.4, "without Nemotron (2.8 GB)": 2.8}


def tok_s(cpu_share):
    return 1000 / (FIXED_MS + RAM_MS * cpu_share)


def per_layer_slots(gb):
    return int(gb * 1024 / EXPERT_MB / N_LAYERS)


def load(trace_dir):
    d = Path(trace_dir)
    meta = [json.loads(l) for l in (d / "meta.jsonl").read_text().splitlines() if l.strip()]
    raw = (d / "trace.bin").read_bytes()
    recs, i = [], 0
    while i + 7 <= len(raw):
        prompt = int.from_bytes(raw[i:i + 2], "little"); phase = raw[i + 2]; layer = raw[i + 3]
        nt = int.from_bytes(raw[i + 4:i + 6], "little"); nu = raw[i + 6]; i += 7
        if i + nt * nu > len(raw):  # a trace still being written
            break
        ids = np.frombuffer(raw, dtype=np.uint8, count=nt * nu, offset=i).reshape(nt, nu); i += nt * nu
        recs.append((prompt, phase, layer, ids))
    return meta, recs


def counts(recs, phase, prompts=None):
    c = np.zeros((N_LAYERS, N_EXPERTS))
    for p, ph, l, ids in recs:
        if ph == phase and (prompts is None or p in prompts):
            np.add.at(c[l], ids.ravel(), 1)
    return c


def coverage(train, test, k):
    """Share of `test` picks served by each layer's top-k experts chosen on `train`."""
    hit = tot = 0
    for l in range(N_LAYERS):
        hot = np.argsort(-train[l])[:k]
        hit += test[l, hot].sum(); tot += test[l].sum()
    return hit / tot


def lru_hit_rate(recs, k, prompts):
    """Per-layer LRU cache of k experts, run over each prompt's generated tokens in order."""
    hit = tot = 0
    for p in prompts:
        caches = [OrderedDict() for _ in range(N_LAYERS)]
        for pp, ph, l, ids in recs:
            if pp != p or ph != 1:
                continue
            cache = caches[l]
            for row in ids:
                for e in row:
                    tot += 1
                    if e in cache:
                        hit += 1; cache.move_to_end(e)
                    else:
                        cache[e] = True
                        if len(cache) > k:
                            cache.popitem(last=False)
    return hit / tot


def main(trace_dir):
    meta, recs = load(trace_dir)
    n = max(r[0] for r in recs) + 1  # prompts with data (all of them once the trace is complete)
    meta = meta[:n]
    dec = counts(recs, 1)
    pre = counts(recs, 0)
    n_dec_tokens = int(dec.sum() / N_LAYERS / K_USED)
    n_pre_tokens = int(pre.sum() / N_LAYERS / K_USED)

    ks = [1, 2, 4, 8, 14, 16, 32, 41, 64, 96, 128, 192, 256]
    # 1. concentration, in-sample (upper bound) and held out (even/odd prompts, both directions)
    even, odd = set(range(0, n, 2)), set(range(1, n, 2))
    de, do = counts(recs, 1, even), counts(recs, 1, odd)
    in_sample = {k: coverage(dec, dec, k) for k in ks}
    held_out = {k: (coverage(de, do, k) + coverage(do, de, k)) / 2 for k in ks}

    # 2. per-layer spread: how many experts cover half / 80% of a layer's picks
    def experts_for(share, c):
        s = np.sort(c)[::-1].cumsum() / c.sum()
        return int(np.searchsorted(s, share) + 1)
    per_layer = [{"layer": l, "top14": float(np.sort(dec[l])[::-1][:14].sum() / dec[l].sum()),
                  "experts_for_50": experts_for(0.5, dec[l]), "experts_for_80": experts_for(0.8, dec[l])}
                 for l in range(N_LAYERS)]

    # 3. task-specific vs global hot sets, leave-one-prompt-out, decode picks
    by_prompt = {p: counts(recs, 1, {p}) for p in range(n)}
    tasks = defaultdict(list)
    for p, m in enumerate(meta):
        tasks[m["laya_task"]].append(p)
    task_rows = {}
    for k in sorted({14, 41} | {per_layer_slots(gb) for gb in BUDGETS_GB.values()}):
        rows = {}
        for task, ps in tasks.items():
            if len(ps) < 3:
                continue
            g = t = 0.0
            for p in ps:
                rest_all = dec - by_prompt[p]
                rest_task = sum(by_prompt[q] for q in ps if q != p)
                g += coverage(rest_all, by_prompt[p], k); t += coverage(rest_task, by_prompt[p], k)
            rows[task] = {"n": len(ps), "global": g / len(ps), "task": t / len(ps)}
        task_rows[k] = rows
    # overlap of hot sets between tasks (mean Jaccard of top-41 over layers)
    tc = {task: sum(by_prompt[p] for p in ps) for task, ps in tasks.items() if len(ps) >= 3}
    names = sorted(tc)
    jac = {}
    for a in names:
        for b in names:
            js = []
            for l in range(N_LAYERS):
                A = set(np.argsort(-tc[a][l])[:41]); B = set(np.argsort(-tc[b][l])[:41])
                js.append(len(A & B) / len(A | B))
            jac[f"{a}|{b}"] = float(np.mean(js))

    # 4. dynamic LRU cache over the generated tokens
    lru = {k: lru_hit_rate(recs, k, range(n)) for k in (5, 14, 41, 64)}

    # 5. prompt reading: distinct experts a batch touches per layer (prompts > 512 tokens)
    touched = []
    for p, ph, l, ids in recs:
        if ph == 0 and ids.shape[0] >= 512:
            touched.append(len(np.unique(ids)))
    touched_mean = float(np.mean(touched)) if touched else None

    # 6. speed estimates for the two VRAM budgets
    est = {}
    for name, gb in BUDGETS_GB.items():
        k = per_layer_slots(gb)
        layers = gb / 0.465
        est[name] = {
            "experts_per_layer": k,
            "whole_layers": round(layers, 1),
            "tok_s_whole_layers": round(tok_s(1 - layers / N_LAYERS), 1),
            "tok_s_hot_static": round(tok_s(1 - held_out_k(de, do, k)), 1),
            "tok_s_hot_task": round(tok_s(1 - task_mean(task_rows, k)), 1) if k in task_rows else None,
            # LRU: misses are computed from RAM *and* copied in. Serial = copies add to every token;
            # overlapped = copies run fully in parallel with the rest (best case).
            **lru_bounds(lru_hit_rate(recs, k, range(n))),
            "hit_static": round(held_out_k(de, do, k), 3),
            "hit_lru": round(lru_hit_rate(recs, k, range(n)), 3),
        }

    res = {
        "prompts": n, "decode_tokens": n_dec_tokens, "prefill_tokens": n_pre_tokens,
        "uniform": {k: k / N_EXPERTS for k in ks},
        "in_sample": in_sample, "held_out": held_out,
        "per_layer": per_layer, "tasks": task_rows, "jaccard41": jac, "lru": lru,
        "prefill_distinct_experts_per_batch": touched_mean, "estimates": est,
        "baseline_tok_s": round(tok_s(1.0), 1),
    }
    out = HERE / "phase1-results.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_layer",)}, indent=1, default=float))
    figures(res)


def lru_bounds(hit):
    miss_per_token = N_LAYERS * K_USED * (1 - hit)
    admit = miss_per_token * ADMIT_MS
    compute = FIXED_MS + RAM_MS * (1 - hit)
    return {"tok_s_lru_serial": round(1000 / (compute + admit), 1),
            "tok_s_lru_overlapped": round(1000 / max(compute, admit), 1),
            "lru_copy_ms_per_token": round(admit, 1)}


def held_out_k(de, do, k):
    return (coverage(de, do, k) + coverage(do, de, k)) / 2


def task_mean(task_rows, k):
    rows = task_rows[k]
    tot = sum(r["n"] for r in rows.values())
    return sum(r["task"] * r["n"] for r in rows.values()) / tot


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE))
    from make_figures import THEMES, save, style, label

    for mode, t in THEMES.items():
        style(t)
        # coverage curve
        ks = [int(k) for k in res["held_out"]]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.plot(ks, [100 * res["uniform"][k] for k in ks], color=t["base"], linewidth=2, linestyle="--",
                label="if routing were uniform")
        ax.plot(ks, [100 * res["in_sample"][k] for k in ks], color=t["q40"], linewidth=2,
                label="hot set chosen on the same prompts (upper bound)")
        ax.plot(ks, [100 * res["held_out"][k] for k in ks], color=t["multi"], linewidth=2.5, marker="o",
                markersize=5, label="hot set chosen on other prompts (realistic)")
        for name, e in res["estimates"].items():
            k = e["experts_per_layer"]
            ax.axvline(k, color=t["grid"], linewidth=1, zorder=0)
            label(ax, k, 62 if k < 16 else 6, f" {k}/layer:\n {name}", t, fontsize=8.2, color=t["ink2"])
        ax.set_xscale("log", base=2)
        ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.minorticks_off()
        ax.set_xlim(1, 256)
        ax.set_ylim(0, 100)
        ax.set_xlabel("experts kept on the GPU per layer (of 256)")
        ax.set_ylabel("% of generated tokens' expert picks")
        ax.grid(axis="y", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8.8)
        ax.set_title("H17 · How much of the routing would a hot-expert cache catch?")
        save(fig, "phase1-coverage", mode)

        # speed estimates, one group per VRAM budget
        rows, groups = [], []
        for name, e in res["estimates"].items():
            groups.append((len(rows), name))
            rows += [(f"whole expert layers ({e['whole_layers']:g})", e["tok_s_whole_layers"], t["base"]),
                     (f"hot experts, one set ({e['experts_per_layer']}/layer)", e["tok_s_hot_static"], t["multi"]),
                     ("hot experts, a set per task", e["tok_s_hot_task"], t["q34"]),
                     ("LRU cache, copies one after another", e["tok_s_lru_serial"], t["q40"]),
                     ("LRU cache, copies fully overlapped", e["tok_s_lru_overlapped"], t["q40"]),
                     None]
        rows = rows[:-1]
        fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(rows) + 1.2))
        ylab, ypos = [], []
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            if r is None:
                continue
            name, v, col = r
            ax.barh(y, v, height=0.62, color=col)
            label(ax, v, y, f"{v:.1f}", t, va="center", xytext=(5, 0), textcoords="offset points")
            ylab.append(name); ypos.append(y)
        for start, name in groups:
            label(ax, 0, len(rows) - 1 - start + 0.62, f"VRAM for experts: {name}", t, fontsize=9, fontweight="bold")
        ax.axvline(res["baseline_tok_s"], color=t["ink2"], linewidth=1, linestyle="--")
        label(ax, res["baseline_tok_s"], -0.9, f"today: {res['baseline_tok_s']}", t, fontsize=8.2,
              color=t["ink2"], ha="center")
        ax.set_yticks(ypos, ylab, fontsize=8.8)
        ax.set_ylim(-1.2, len(rows) + 0.2)
        ax.set_xlim(0, max(r[1] for r in rows if r) * 1.12)
        ax.set_xlabel("estimated generated tokens / s (speed model fitted to Phase 0)")
        ax.grid(axis="x", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title("H17–H20 · What each way of spending the VRAM would buy")
        save(fig, "phase1-speed", mode)

        # task-specific vs one shared hot set, at the "without Nemotron" budget
        k = max(e["experts_per_layer"] for e in res["estimates"].values())
        rows = sorted(res["tasks"][k].items(), key=lambda kv: kv[1]["task"] - kv[1]["global"], reverse=True)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        for i, (task, r) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y + 0.19, 100 * r["global"], height=0.36, color=t["multi"])
            ax.barh(y - 0.19, 100 * r["task"], height=0.36, color=t["q34"])
            label(ax, 100 * r["global"], y + 0.19, f"{100 * r['global']:.0f}%", t, va="center", fontsize=8.6,
                  xytext=(4, 0), textcoords="offset points")
            label(ax, 100 * r["task"], y - 0.19, f"{100 * r['task']:.0f}%  (+{100 * (r['task'] - r['global']):.0f} points)", t,
                  va="center", fontsize=8.6, xytext=(4, 0), textcoords="offset points")
        ax.set_yticks(range(len(rows)), [f"{task}  ({r['n']} prompts)" for task, r in reversed(rows)])
        ax.set_xlim(0, 85)
        ax.set_xlabel(f"% of picks caught by {k} hot experts per layer, on prompts not used to pick them")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=t["multi"], label="one shared hot set"),
                           Patch(color=t["q34"], label="a hot set per Laya task")], loc="lower right", fontsize=8.8)
        ax.grid(axis="x", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title("H18 · Code and creative writing use their own experts")
        save(fig, "phase1-tasks", mode)


if __name__ == "__main__":
    main(sys.argv[1])
