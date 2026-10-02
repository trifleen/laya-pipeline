"""Build the prompt set for expert-trace: eval/cases.jsonl plus 40 extra requests.

    uv run python tools/expert-trace/make_prompts.py OUT_DIR

Writes OUT_DIR/prompts.jsonl ({"text": ...}, chat-formatted for Qwen3.6 with thinking off) and
OUT_DIR/meta.jsonl (id, question, intended task, Laya's predicted task and its probability).
Documents are cut to their first 6,000 characters so every prompt fits one 2,048-token batch.
"""

import json
import sys
from pathlib import Path

from laya_pipeline import pipeline, questions

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are a helpful, precise assistant."

EXTRA = {
    "code": [
        "Write a Python function that merges two sorted lists in linear time.",
        "Why does this JavaScript print 3 three times? for (var i = 0; i < 3; i++) setTimeout(() => console.log(i));",
        "Write a SQL query that returns the second highest salary per department.",
        "Explain the difference between a mutex and a semaphore with a C example.",
        "Write a Rust function that parses '12:34:56' into seconds and returns a Result.",
        "My Docker container exits immediately with code 0. How do I find out why?",
        "Write a bash script that backs up ~/Documents to a dated tar.gz and keeps the last 7.",
        "Refactor this to be more idiomatic Python: result = []\nfor x in data:\n    if x % 2 == 0:\n        result.append(x * x)",
        "How do I fix 'TypeError: Cannot read properties of undefined (reading map)' in React?",
        "Implement an LRU cache in Go with Get and Put in O(1).",
    ],
    "factual": [
        "Why is the sky blue?",
        "What caused the fall of the Western Roman Empire?",
        "How does a heat pump move heat from a cold place to a warm one?",
        "What is the difference between a virus and a bacterium?",
        "Explain how vaccines train the immune system.",
        "What is 17% of 2,350? Show the calculation.",
        "Solve for x: 3x^2 - 12x + 9 = 0.",
        "A tank fills in 6 hours with pipe A and 4 hours with pipe B. How long with both?",
        "What is the integral of x * e^x?",
        "How many ways can 5 people sit in a row if two of them must sit together?",
    ],
    "creative": [
        "Write a haiku about a server room at night.",
        "Write the opening paragraph of a mystery novel set in a fishing village in northern Norway.",
        "Invent a name and a slogan for a coffee shop run by retired astronauts.",
        "Write a limerick about a cat who learns to code.",
        "Describe a sunrise over the fjords for a travel brochure, in 80 words.",
        "Write a short bedtime story about a robot who is afraid of the dark.",
        "Brainstorm ten names for a board game about trading spices.",
        "Write a toast for my best friend's wedding, warm and a little funny.",
        "Write a product description for noise-cancelling headphones aimed at students.",
        "Write a dialogue between a lighthouse and the moon.",
    ],
    "chat": [
        "Good morning! What's a nice way to start a rainy Saturday?",
        "I'm feeling a bit stressed about my exams. Any advice?",
        "What's your favourite season and why?",
        "Can you recommend a few podcasts about history?",
        "Thanks for the help yesterday!",
        "I just adopted a puppy. Any tips for the first week?",
        "Do you think it's better to read the book or watch the movie first?",
        "What should I cook tonight with eggs, spinach and feta?",
        "I'm bored. Suggest something fun to do indoors.",
        "How do I politely decline an invitation to a party?",
    ],
}


def chatml(question, context=""):
    user = question
    if context:
        user = (f"Context:\n\n{context[:6000]}\n\n---\n\nAnswer from the context. If it does not contain "
                f"the answer, say so instead of guessing.\n\nQuestion: {question}")
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in (json.loads(l) for l in (ROOT / "eval/cases.jsonl").read_text().splitlines() if l.strip()):
        ctx = (ROOT / "eval" / case["context"]).read_text() if case.get("context") else ""
        rows.append({"id": case["id"], "question": case["question"], "intended": "document" if ctx else "eval",
                     "text": chatml(case["question"], ctx), "has_doc": bool(ctx)})
    for task, qs in EXTRA.items():
        for i, q in enumerate(qs):
            rows.append({"id": f"{task}-{i}", "question": q, "intended": task, "text": chatml(q), "has_doc": False})

    preds = pipeline.router().predict_batch([{"state": r["question"], "questions": {"task": questions.ROUTE["task"]}}
                                            for r in rows])
    with (out / "prompts.jsonl").open("w") as fp, (out / "meta.jsonl").open("w") as fm:
        for r, p in zip(rows, preds):
            a = p["answers"]["task"]
            fp.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"id": r["id"], "question": r["question"], "intended": r["intended"],
                                 "has_doc": r["has_doc"], "laya_task": a["choice"],
                                 "p_task": round(a["probabilities"][a["choice"]], 3)}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
