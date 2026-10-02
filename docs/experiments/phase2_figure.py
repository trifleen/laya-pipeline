"""Phase 2 figure: predicted vs measured generation speed of the hot-expert prototype.

    uv run --with matplotlib python docs/experiments/phase2_figure.py logs/phase2
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import THEMES, label, save, style  # noqa: E402


def main(d):
    d = Path(d)
    pred = json.loads((d / "predictions.json").read_text())
    meas = {c: json.loads((d / f"bench-{c}.json").read_text())["gen_tok_s"] for c in ("baseline", "8layers-k39", "all-k39")}
    first = json.loads((d / "bench-all-k39-cpu-tables.json").read_text())["gen_tok_s"]
    base = meas["baseline"]
    rows = [("baseline (every expert in RAM)", base, base),
            ("8 layers × 39 hot experts (0.6 GB)", base * pred["8layers-k39"]["tok_s"] / pred["baseline"]["tok_s"], meas["8layers-k39"]),
            ("40 layers × 39 hot experts (2.8 GB)", base * pred["all-k39"]["tok_s"] / pred["baseline"]["tok_s"], meas["all-k39"])]
    for mode, t in THEMES.items():
        style(t)
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        for i, (name, p, m) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y + 0.19, p, height=0.36, color=t["base"])
            ax.barh(y - 0.19, m, height=0.36, color=t["multi"])
            label(ax, p, y + 0.19, f"predicted {p:.1f}", t, va="center", fontsize=8.6, xytext=(4, 0), textcoords="offset points")
            label(ax, m, y - 0.19, f"measured {m:.1f}", t, va="center", fontsize=8.6, xytext=(4, 0), textcoords="offset points")
        y = 0
        ax.scatter([first], [y - 0.19], marker="|", s=260, color=t["q40"], zorder=3)
        label(ax, first, y - 0.55, f"first try, ids on the CPU: {first:.1f}", t, fontsize=8.2, color=t["ink2"], ha="center")
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
        ax.set_xlim(0, 60)
        ax.set_ylim(-0.9, len(rows) - 0.4)
        ax.set_xlabel("generated tokens / s, 17 eval prompts, greedy (prediction scaled to the measured baseline)")
        ax.grid(axis="x", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title("H21 · The hot-expert prototype: predicted +22%, measured ±0")
        save(fig, "phase2-prototype", mode)


if __name__ == "__main__":
    main(sys.argv[1])
