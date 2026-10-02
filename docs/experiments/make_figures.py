"""Figures for the experiment log, in a light and a dark version for GitHub's <picture> switch.

    uv run --with matplotlib python docs/experiments/make_figures.py BASELINE TUNED QWEN40 QWEN34

Arguments are logs/eval/*.json runs (TUNED is the multi-model run). llama-bench numbers are
copied from bench/results/phase0, with the file each came from.
Writes docs/experiments/img/<name>-light.svg and -dark.svg.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

OUT = Path(__file__).parent / "img"

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, both modes),
# plus a neutral for "before". Text never wears a series colour.
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e8eb",
                  multi="#2a78d6", q40="#eb6834", q34="#1baf7a", base="#8c959f", tint="#c9dcf4"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#262c36",
                 multi="#3987e5", q40="#d95926", q34="#199e70", base="#6e7681", tint="#1f3a5c"),
}


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "grid.color": t["grid"], "font.size": 10.5, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path",
        "font.family": "DejaVu Sans", "legend.frameon": False, "legend.labelcolor": t["ink"],
    })


def save(fig, name, mode):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def label(ax, x, y, text, t, **kw):
    kw = {"color": t["ink"], "fontsize": 9.5, **kw}
    ax.annotate(text, (x, y), **kw)


# ---- figures -----------------------------------------------------------------------------------

def fig_memory(t, mode):
    """Where every byte lives, each bar to the scale of its own memory."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 2.6), gridspec_kw={"hspace": 0.9})
    rows = [
        ("GPU memory · 7.8 GB", 7.8, [("Qwen: attention, KV", 3.1, t["multi"], "white"), ("Nemotron 4B", 3.0, t["base"], t["bg"]),
                                     ("Laya", 1.0, t["ink2"], t["bg"]), ("embed", 0.3, t["grid"], t["ink"])]),
        ("System RAM · 30 GB", 30, [("Qwen: 40 layers × 256 experts", 19.0, t["tint"], t["ink"]),
                                    ("desktop, apps", 5.5, t["grid"], t["ink"]), ("prompt cache", 1.0, t["base"], t["bg"])]),
    ]
    for ax, (title, cap, segs) in zip(axes, rows):
        x = 0
        for name, gb, col, fg in segs:
            ax.barh(0, gb, left=x, height=0.62, color=col, edgecolor=t["bg"], linewidth=2)
            if gb / cap > 0.06:
                ax.text(x + gb / 2, 0, f"{name}\n{gb:g} GB", ha="center", va="center", fontsize=8.6, color=fg)
            x += gb
        ax.barh(0, cap - x, left=x, height=0.62, color="none", edgecolor=t["grid"], linewidth=1, hatch="")
        ax.text(x + (cap - x) / 2, 0, f"free\n{cap - x:.1f} GB", ha="center", va="center", fontsize=8.6, color=t["ink2"])
        ax.set_xlim(0, cap)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
        small = [f"{n} {gb:g} GB" for n, gb, _, _ in segs if gb / cap <= 0.06]
        if small:
            ax.text(x, -0.55, "unlabelled: " + ", ".join(small), ha="right", va="top", fontsize=8.2, color=t["ink2"])
    fig.suptitle("Qwen spans both memories: what every token needs stays on the GPU, the experts stay in RAM",
                 x=0.125, ha="left", fontsize=11, fontweight="bold", y=1.08)
    save(fig, "memory", mode)


def fig_ubatch(t, mode):
    # results/phase0/ubatch.txt: 2,048-token prompt (512 measured at n-cpu-moe 32, the rest at 36)
    data = [(512, 273), (1024, 421), (2048, 699), (4096, 699)]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    xs = range(len(data))
    ax.bar(xs, [v for _, v in data], width=0.6, color=t["multi"])
    for i, (_, v) in enumerate(data):
        label(ax, i, v, f"{v}", t, ha="center", xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(list(xs), [f"{ub}" for ub, _ in data])
    ax.set_xlabel("prompt batch size (ubatch, tokens)")
    ax.set_ylabel("prompt tokens / s")
    ax.set_ylim(0, 800)
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("H7 · Bigger batches read prompts 2.5× faster")
    save(fig, "ubatch", mode)


def fig_layers(t, mode):
    # results/phase0/SUMMARY.md: 128 generated tokens, empty context
    data = [(40, 42.3), (36, 45.5), (32, 48.2), (30, 49.8)]
    on_gpu = [40 - n for n, _ in data]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(on_gpu, [v for _, v in data], color=t["multi"], linewidth=2, marker="o", markersize=7,
            markeredgecolor=t["bg"], markeredgewidth=2)
    for g, (_, v) in zip(on_gpu, data):
        label(ax, g, v, f"{v}", t, ha="center", xytext=(0, 9), textcoords="offset points")
        label(ax, g, 33.6, f"+{g * 0.465:.1f} GB" if g else "0 GB", t, ha="center", fontsize=8.2, color=t["ink2"])
    ax.set_xticks(on_gpu)
    ax.set_xlabel("expert layers on the GPU (of 40) · extra VRAM below")
    ax.set_ylabel("generated tokens / s")
    ax.set_ylim(32, 55)
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("H4 · Each layer on the GPU: ~465 MB for ~0.8 tok/s")
    save(fig, "layers", mode)


def fig_prompt_cost(t, mode):
    """H9: prompt time steps up per 2,048-token batch; the routing cost model is drawn as a line."""
    bench = [(256, 1.83), (1024, 2.64), (2048, 3.16)]  # results/phase0/loadmode.txt
    server = [(1012, 3.35), (2203, 5.82), (2296, 5.70), (2527, 6.18), (4434, 9.66)]  # llama-server timings
    ns = list(range(1, 4600, 8))
    big = [1.5 * math.ceil(n / 2048) + n / 800 for n in ns]
    small = [n / 3300 for n in ns]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ns, big, color=t["multi"], linewidth=2, label="35B MoE, cost model: 1.5 s per batch + 800 tok/s")
    ax.plot(ns, small, color=t["q34"], linewidth=2, label="4B on the GPU: 3,300 tok/s")
    ax.scatter(*zip(*server), s=48, color=t["multi"], edgecolor=t["bg"], linewidth=2, zorder=3, label="measured, llama-server")
    ax.scatter(*zip(*bench), s=48, marker="s", facecolor=t["bg"], edgecolor=t["multi"], linewidth=2, zorder=3,
               label="measured, llama-bench")
    for b in (2048, 4096):
        ax.axvline(b, color=t["grid"], linewidth=1, linestyle="--", zorder=0)
        label(ax, b, 0.25, f" batch boundary\n {b} tokens", t, ha="left", fontsize=8.2, color=t["ink2"])
    ax.set_xlim(0, 4600)
    ax.set_ylim(0, 11.2)
    ax.set_xlabel("new prompt tokens")
    ax.set_ylabel("seconds to read the prompt")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("H9 · Prompt time jumps at every 2,048-token batch: the RAM experts are copied once per batch")
    save(fig, "prompt-cost", mode)


def fig_laya(t, mode):
    # measured 2026-10-02, Laya 420M: CPU fp32 vs GPU bf16 (same inputs)
    rows = [("one routing decision", 443, 18), ("judging 6 context chunks", 5089, 130)]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, (name, cpu, gpu) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + 0.18, cpu, height=0.32, color=t["base"])
        ax.barh(y - 0.18, gpu, height=0.32, color=t["multi"])
        label(ax, cpu, y + 0.18, f"CPU {cpu:,} ms", t, va="center", xytext=(5, 0), textcoords="offset points")
        label(ax, gpu, y - 0.18, f"GPU {gpu} ms  ({cpu / gpu:.0f}× faster)", t, va="center", xytext=(5, 0),
              textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(8, 60000)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10 ms", "100 ms", "1 s", "10 s"])
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.grid(axis="x", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("H13–H14 · Laya on the CPU was the hidden bottleneck (log scale)")
    save(fig, "laya", mode)


def _per_request(ax, t, ids, series, xlabel, doc, xmax=400):
    """One row per request; series = [(name, {id: seconds}, colour, marker, hollow)]."""
    for i, rid in enumerate(ids):
        y = len(ids) - 1 - i
        vals = [s[1][rid] for s in series]
        ax.plot([min(vals), max(vals)], [y, y], color=t["grid"], linewidth=2.5, zorder=1, solid_capstyle="round")
        for name, d, col, mk, hollow in series:
            ax.scatter(d[rid], y, s=58, marker=mk, zorder=3, linewidth=2,
                       facecolor=t["bg"] if hollow else col, edgecolor=col)
    ax.set_xscale("log")
    ax.set_xlim(0.08, xmax)
    ticks = [x for x in (0.1, 0.3, 1, 3, 10, 30, 100, 300) if x < xmax]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x:g} s" for x in ticks])
    ax.set_yticks(range(len(ids)), [r + ("  · doc" if doc[r] else "") for r in reversed(ids)])
    ax.grid(axis="x", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    for name, _, col, mk, hollow in series:
        ax.scatter([], [], s=58, marker=mk, linewidth=2, facecolor=t["bg"] if hollow else col, edgecolor=col, label=name)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), fontsize=9.2)


def fig_r1(t, mode, b, m):
    bt = {r["id"]: r["time_to_answer_s"] for r in b["rows"]}
    mt = {r["id"]: r["time_to_answer_s"] for r in m["rows"]}
    doc = {r["id"]: bool(r["context_tokens"]) for r in m["rows"]}
    ids = sorted(bt, key=lambda k: (doc[k], bt[k]))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    _per_request(ax, t, ids, [("baseline", bt, t["base"], "o", True), ("tuned", mt, t["multi"], "o", False)],
                 "time to first token (log scale)", doc, xmax=25)
    fig.suptitle("Result 1 · Time to first token per request, before and after", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", y=1.04)
    save(fig, "result1", mode)


def fig_r2(t, mode, m, a, c):
    tot = lambda run: {r["id"]: r["total_s"] for r in run["rows"]}  # noqa: E731
    mt, at, ct = tot(m), tot(a), tot(c)
    doc = {r["id"]: bool(r["context_tokens"]) for r in m["rows"]}
    ids = sorted(mt, key=lambda k: (doc[k], mt[k]))
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    _per_request(ax, t, ids, [("multi-model", mt, t["multi"], "o", False),
                              ("Qwen only, 40 in RAM", at, t["q40"], "s", False),
                              ("Qwen only, 34 in RAM", ct, t["q34"], "^", False)],
                 "total time per request (log scale)", doc)
    y = len(ids) - 1 - ids.index("poem")
    label(ax, at["poem"], y, "one-off runaway: 181 s for a short poem  ", t, fontsize=8.4, color=t["ink2"],
          xytext=(-10, 9), textcoords="offset points", ha="right", va="bottom")
    fig.suptitle("Result 2 · Total time per request in the three setups", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", y=1.04)
    save(fig, "result2", mode)


def fig_r2_totals(t, mode, m, a, c):
    def wo(run):
        return sum(r["total_s"] for r in run["rows"] if r["id"] != "poem")
    rows = [("multi-model", wo(m), m["summary"]["p_bad_mean"], t["multi"]),
            ("Qwen only, 34 in RAM", wo(c), c["summary"]["p_bad_mean"], t["q34"]),
            ("Qwen only, 40 in RAM", wo(a), a["summary"]["p_bad_mean"], t["q40"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.4), gridspec_kw={"wspace": 0.08})
    for i, (name, tot, pbad, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax1.barh(y, tot, height=0.55, color=col)
        label(ax1, tot, y, f"{tot:.0f} s", t, va="center", xytext=(5, 0), textcoords="offset points")
        ax2.barh(y, pbad, height=0.55, color=col)
        label(ax2, pbad, y, f"{pbad:.3f}", t, va="center", xytext=(5, 0), textcoords="offset points")
    ax1.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax2.set_yticks([])
    ax1.set_xlim(0, 320)
    ax2.set_xlim(0, 0.22)
    ax1.set_title("total for 16 requests (poem left out) ↓", fontsize=10)
    ax2.set_title("Laya answer check, mean p(bad) ↓", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(axis="x", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Result 2 · The small model buys speed; Qwen alone scores slightly better", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", y=1.12)
    save(fig, "result2-totals", mode)


def main(baseline, tuned, q40, q34):
    b, m, a, c = (json.loads(Path(p).read_text()) for p in (baseline, tuned, q40, q34))
    for mode, t in THEMES.items():
        style(t)
        fig_memory(t, mode)
        fig_ubatch(t, mode)
        fig_layers(t, mode)
        fig_prompt_cost(t, mode)
        fig_laya(t, mode)
        fig_r1(t, mode, b, m)
        fig_r2(t, mode, m, a, c)
        fig_r2_totals(t, mode, m, a, c)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
