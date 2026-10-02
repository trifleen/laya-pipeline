"""The README diagrams, each in a light and a dark version (for GitHub's <picture> switch).

    python docs/img/make_diagrams.py

Writes docs/img/diagram-{flow,memory,loop}-{light,dark}.svg. Plain SVG, no dependencies.
Every diagram ends with a short glossary strip, so it reads without prior knowledge.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).parent
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", line="#d0d7de", ink="#1f2328", muted="#59636e",
                  laya="#8250df", laya_soft="#f3eeff", model="#2a78d6", model_soft="#e8f1fc",
                  embed="#1a7f64", embed_soft="#e6f4ef", ram="#a8650b", ram_soft="#fbf0dc",
                  free="#ffffff", warn="#cf222e"),
    "dark": dict(bg="#0d1117", card="#161b22", line="#30363d", ink="#e6edf3", muted="#9198a1",
                 laya="#a371f7", laya_soft="#231a36", model="#4c9aff", model_soft="#13233a",
                 embed="#3fb98f", embed_soft="#11291f", ram="#e3a53b", ram_soft="#2e2310",
                 free="#0d1117", warn="#ff7b72"),
}


class Svg:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.parts = w, h, t, []

    def rect(self, x, y, w, h, fill, stroke=None, r=10, sw=1.2, dash=None):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{s}{d}/>')

    def text(self, x, y, lines, size=14, color=None, weight=400, anchor="start", lh=1.35):
        if isinstance(lines, str):
            lines = [lines]
        color = color or self.t["ink"]
        spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh:.1f}">{escape(l)}</tspan>'
                        for i, l in enumerate(lines))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
                          f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{spans}</text>')

    def arrow(self, points, color, sw=2, dash=None, head=True):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
                          f'stroke-linecap="round" stroke-linejoin="round"/>')
        if head:
            (x1, y1), (x2, y2) = points[-2], points[-1]
            import math
            a = math.atan2(y2 - y1, x2 - x1)
            p = [(x2, y2), (x2 - 10 * math.cos(a - 0.45), y2 - 10 * math.sin(a - 0.45)),
                 (x2 - 10 * math.cos(a + 0.45), y2 - 10 * math.sin(a + 0.45))]
            self.parts.append(f'<polygon points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in p)}" fill="{color}"/>')

    def chip(self, x, y, label, color, soft):
        w = 8 * len(label) + 18
        self.rect(x, y, w, 22, soft, color, r=11, sw=1)
        self.text(x + w / 2, y + 15.5, label, size=12, color=color, weight=600, anchor="middle")
        return w

    def glossary(self, y, items, title="Words used here"):
        t = self.t
        n = len(items)
        cols = 2 if n == 4 else min(n, 3)
        rows = (n + cols - 1) // cols
        colw = (self.w - 48 - (cols - 1) * 16) / cols
        row_h = 30 + 17 * max(len(d) for _, d in items)
        h = 40 + rows * row_h
        self.rect(24, y, self.w - 48, h, t["card"], t["line"], r=10)
        self.text(40, y + 22, title, size=12, color=t["muted"], weight=700)
        for i, (term, desc) in enumerate(items):
            cx = 40 + (i % cols) * (colw + 16)
            cy = y + 48 + (i // cols) * row_h
            self.text(cx, cy, term, size=13, weight=700)
            self.text(cx, cy + 19, desc, size=12.5, color=t["muted"])
        self.h = y + h + 20  # the canvas ends just below the glossary
        return y + h

    def save(self, name, mode):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">\n<rect width="100%" height="100%" fill="{self.t["bg"]}"/>\n'
               + "\n".join(self.parts) + "\n</svg>\n")
        (OUT / f"diagram-{name}-{mode}.svg").write_text(svg)


def flow(mode, t):
    s = Svg(960, 590, t)
    s.text(24, 34, "How a request flows", size=20, weight=700)
    s.text(24, 56, "Laya makes the decisions in milliseconds; the language models do the writing.", size=13.5, color=t["muted"])

    # input and output pills
    s.rect(24, 112, 108, 64, t["card"], t["line"], r=32)
    s.text(78, 139, ["Your question", "+ files"], size=13, anchor="middle", weight=600)
    s.rect(840, 112, 96, 64, t["card"], t["line"], r=32)
    s.text(888, 148, "Answer", size=14, anchor="middle", weight=700)

    stages = [
        ("1", "Compress", ["Cut files into chunks.", "Keep the ones that", "help answer."], [("embedder", "embed"), ("Laya", "laya")]),
        ("2", "Route", ["Pick the model, and", "whether it should", "think first."], [("Laya", "laya")]),
        ("3", "Generate", ["Write the answer,", "streamed word by", "word."], [("4B or 35B model", "model")]),
        ("4", "Check", ["Does the answer", "address the", "question?"], [("Laya", "laya")]),
    ]
    x0, w, gap, y = 152, 150, 18, 86
    for i, (num, title, desc, who) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, 176, t["card"], t["line"], r=12)
        s.rect(x + 14, y + 14, 26, 26, t["ink"], r=13)
        s.text(x + 27, y + 32, num, size=13, color=t["bg"], weight=700, anchor="middle")
        s.text(x + 48, y + 33, title, size=16, weight=700)
        s.text(x + 14, y + 66, desc, size=13, color=t["muted"])
        for k, (label, role) in enumerate(who):  # who does it, stacked from the bottom
            s.chip(x + 14, y + 140 - k * 28, label, t[role], t[f"{role}_soft"])
        if i < 3:
            s.arrow([(x + w + 2, y + 88), (x + w + gap - 2, y + 88)], t["muted"])
    s.arrow([(132, 144), (x0 - 4, 144)], t["muted"])
    last = x0 + 3 * (w + gap) + w
    s.arrow([(last + 2, 144), (838, 144)], t["muted"])

    # retry loop: from Check back to Generate
    gx = x0 + 2 * (w + gap) + w / 2
    cx = x0 + 3 * (w + gap) + w / 2
    s.arrow([(cx, y + 176), (cx, y + 214), (gx, y + 214), (gx, y + 180)], t["warn"], dash="6 5")
    s.text((gx + cx) / 2, y + 234, "flagged? retry once: 35B model, thinking on", size=12.5, color=t["warn"],
           anchor="middle", weight=600)

    s.glossary(352, [
        ("Laya", ["A small model (420M parameters) that answers", "multiple-choice questions about your request in", "~20 ms. It decides; it never writes the answer."]),
        ("Chunk", ["A ~1,200-character piece of an attached file.", "Only the useful chunks are sent on, so the", "model reads less and answers faster."]),
        ("Embedder", ["Turns text into a list of numbers so that", "similar meaning gives similar numbers. Used", "to rank chunks against your question."]),
        ("4B / 35B model", ["The language models that write. 4B (billion", "parameters) is fast; 35B is smarter but slower.", "Easy requests go to the 4B one."]),
        ("Thinking", ["The model first writes hidden reasoning,", "then the answer. Better on hard problems,", "much slower, so only used when needed."]),
    ])
    s.save("flow", mode)


def memory(mode, t):
    s = Svg(960, 560, t)
    s.text(24, 34, "What runs where: a 35B model on an 8 GB graphics card", size=20, weight=700)
    s.text(24, 56, "The big model is split: the parts every word needs sit on the GPU, its 'experts' wait in RAM.",
           size=13.5, color=t["muted"])

    def bar(y, label, sub, total, segs):
        s.text(24, y + 20, label, size=14, weight=700)
        s.text(24, y + 39, sub, size=12, color=t["muted"])
        x, W = 190, 746
        s.rect(x, y, W, 52, t["free"], t["line"], r=8)
        for name, gb, fill, ink, note in segs:
            w = W * gb / total
            s.rect(x + 1, y + 1, w - 2, 50, fill, r=7)
            if w > 70:
                s.text(x + 10, y + 22, name, size=12.5, color=ink, weight=700)
                s.text(x + 10, y + 40, note, size=11.5, color=ink)
            x += w
        s.text(x + (190 + W - x) / 2, y + 31, "free", size=12, color=t["muted"], anchor="middle")

    bar(92, "GPU memory", "8 GB · very fast", 7.8, [
        ("Qwen 35B: core", 3.1, t["model"], t["bg"], "used for every word"),
        ("Nemotron 4B", 3.0, t["model_soft"], t["ink"], "the fast small model"),
        ("Laya", 1.0, t["laya"], t["bg"], "decisions"),
        ("", 0.3, t["embed"], t["bg"], ""),
    ])
    s.text(190 + 746 * 7.1 / 7.8 + 4, 160, "embedder", size=11.5, color=t["embed"], weight=600)

    bar(204, "System RAM", "30 GB · ~10× slower", 30, [
        ("Qwen 35B: experts", 19.0, t["ram"], t["bg"], "40 layers × 256 experts, ~19 GB"),
        ("Linux, apps", 5.5, t["ram_soft"], t["ink"], "everything else"),
    ])

    # the key fact
    s.rect(24, 290, 912, 84, t["card"], t["line"], r=12)
    s.text(44, 318, "Why this is fast enough", size=14, weight=700)
    s.text(44, 340, ["Each word only uses 8 of the 256 experts in every layer, so the GPU reads ~0.6 GB from RAM per word,",
                     "not all 19 GB. Result: 40–50 words per second from a model that couldn't fit in GPU memory at all."],
           size=13, color=t["muted"])

    s.glossary(394, [
        ("GPU memory (VRAM)", ["The graphics card's own memory. Very fast,", "but small: 8 GB here."]),
        ("System RAM", ["The computer's main memory. Bigger, but", "the GPU reads it ~10× slower."]),
        ("Parameters / 35B", ["The numbers a model learned; 35B = 35", "billion. More usually means smarter."]),
        ("Experts", ["Small sub-networks inside the model. A router", "picks 8 of 256 per word, per layer."]),
        ("Word (token)", ["Models write in tokens, pieces of words.", "'Words per second' is tokens per second."]),
        ("Layer", ["One of the model's 40 stacked processing", "steps; every word passes through all 40."]),
    ])
    s.save("memory", mode)


def loop(mode, t):
    s = Svg(960, 470, t)
    s.text(24, 34, "Teaching Laya your preferences", size=20, weight=700)
    s.text(24, 56, "Out of the box Laya is general-purpose. A few clicks per answer make it fit how you work.",
           size=13.5, color=t["muted"])

    def node(x, y, w, title, lines, color, soft):
        s.rect(x, y, w, 92, soft, color, r=12, sw=1.5)
        s.text(x + 16, y + 28, title, size=15, weight=700, color=color)
        s.text(x + 16, y + 50, lines, size=12.5, color=t["ink"])

    node(24, 96, 190, "1 · Use it", ["Ask questions as usual", "on the dashboard."], t["model"], t["model_soft"])
    node(254, 96, 200, "2 · Label", ["One click per decision:", "was Laya right?"], t["laya"], t["laya_soft"])
    # three tuning tools stacked
    ty = [76, 140, 204]
    for (title, lines), yy in zip([("What if…", "try new thresholds on past decisions"),
                                   ("Calibrate", "make '70% sure' mean right 70% of the time"),
                                   ("Edit questions", "reword what Laya is asked, score it first")], ty):
        s.rect(494, yy, 252, 52, t["card"], t["line"], r=10)
        s.text(510, yy + 22, title, size=13.5, weight=700)
        s.text(510, yy + 40, lines, size=11.5, color=t["muted"])
        s.arrow([(454, 142), (474, 142), (474, yy + 26), (492, yy + 26)], t["muted"])
    node(786, 96, 150, "3 · Better", ["routing for your", "kind of requests"], t["embed"], t["embed_soft"])
    for yy in ty:
        s.arrow([(746, yy + 26), (766, yy + 26), (766, 142), (784, 142)], t["muted"], head=(yy == ty[1]))
    s.arrow([(214, 142), (252, 142)], t["muted"])
    # back to use
    s.arrow([(861, 188), (861, 284), (119, 284), (119, 190)], t["muted"], dash="6 5")
    s.text(490, 302, "repeat: every label makes the next decisions better", size=12.5, color=t["muted"], anchor="middle")

    s.glossary(330, [
        ("Label", ["Your verdict on one of Laya's decisions,", "e.g. 'this was hard', 'this answer was bad'."]),
        ("Threshold", ["How sure Laya must be before acting,", "e.g. use the big model at ≥ 60% 'hard'."]),
        ("Calibration", ["Adjusts Laya's confidence so its", "percentages match how often it's right."]),
        ("Fine-tuning", ["After a few hundred labels, retrain Laya", "itself on them (free Kaggle notebook)."]),
    ])
    s.save("loop", mode)


if __name__ == "__main__":
    for mode, t in THEMES.items():
        flow(mode, t)
        memory(mode, t)
        loop(mode, t)
    print("wrote", sorted(p.name for p in OUT.glob("diagram-*.svg")))
