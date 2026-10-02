"""`laya-pipeline eval`: run a fixed set of requests through the pipeline and score speed and quality.

Every speed-up has a switch in config.py (OVERLAP_ROUTE, CONTEXT_IN_USER, ADAPTIVE_BUDGET,
LENGTH_ROUTING), so the same set can be run with and without it:

    OVERLAP_ROUTE=0 CONTEXT_IN_USER=0 ADAPTIVE_BUDGET=0 LENGTH_ROUTING=0 laya eval --label baseline
    laya eval --label tuned --compare logs/eval/<baseline>.json

Quality is scored three ways, none needing a human:
- expect:  the answer must contain every listed string (case-insensitive), for facts we know;
- p_bad:   Laya's answer check, as in the pipeline;
- agrees:  with --compare, Laya judges whether each answer says the same as the baseline's answer.
Runs here are kept out of the decision log and the dashboard history.
"""

import json
import statistics
import time
from pathlib import Path

from . import config, pipeline

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "eval" / "cases.jsonl"
SWITCHES = ["OVERLAP_ROUTE", "CONTEXT_IN_USER", "ADAPTIVE_BUDGET", "LENGTH_ROUTING"]

AGREES = {
    "agrees": {
        "type": "choice",
        "instructions": "Compare the new answer with the reference answer to the same question.",
        "criteria": {
            "A": "the new answer gives the same facts or conclusion as the reference",
            "B": "the new answer contradicts the reference or misses its key point",
        },
    },
}


def _load_cases(only=None):
    rows = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    return [r for r in rows if not only or r["id"] in only]


def _context(case):
    return (CASES.parent / case["context"]).read_text() if case.get("context") else ""


def _row(case, answer, trace, wall_s):
    gens = [g for g in (trace.get("generate"), trace.get("retry")) if g]
    final = gens[-1]
    ro = trace.get("route", {})
    text = answer.lower()
    expect = case.get("expect")
    return {
        "id": case["id"], "question": case["question"], "answer": answer,
        "model": final["model"], "think": ro.get("think"), "reason": ro.get("reason"),
        "scope": ro.get("scope"), "retried": "retry" in trace,
        "chunks_out": trace.get("compress", {}).get("chunks_out"),
        "context_tokens": trace.get("compress", {}).get("tokens_out"),
        "prompt_tokens": sum(g.get("prompt_tokens") or 0 for g in gens),
        "cached_tokens": sum(g.get("cached_tokens") or 0 for g in gens),
        "prompt_ms": sum(g.get("prompt_ms") or 0 for g in gens),
        "first_token_ms": trace["generate"]["first_token_ms"],
        "tokens_per_s": final.get("tokens_per_s"),
        "total_s": round(wall_s, 2),
        "time_to_answer_s": round(sum(s["ms"] for s in trace.get("stages", [])
                                      if s["stage"] in ("compress", "route")) / 1000
                                  + trace["generate"]["first_token_ms"] / 1000, 2),
        "expect_ok": None if not expect else all(e.lower() in text for e in expect),
        "p_bad": trace.get("checks", [{}])[-1].get("p_bad"),
    }


def run(label, only=None, compare=None):
    # Keep eval runs out of the decision log (fine-tuning data) and the dashboard history.
    pipeline.log_decision = lambda *a, **k: None
    pipeline.log_run = lambda *a, **k: None

    cases = _load_cases(only)
    switches = {name: getattr(config, name) for name in SWITCHES}
    print(f"eval '{label}': {len(cases)} cases, switches {switches}")
    print("warming up (Laya + models)…")
    pipeline.ask("Say OK.")

    rows, history = [], []
    for case in cases:
        if not case.get("followup"):
            history = []
        t = time.perf_counter()
        answer, trace = pipeline.ask(case["question"], _context(case), history)
        row = _row(case, answer, trace, time.perf_counter() - t)
        rows.append(row)
        history = [*history, {"role": "user", "content": case["question"]},
                   {"role": "assistant", "content": answer}]
        ok = {None: " ", True: "✓", False: "✗"}[row["expect_ok"]]
        print(f"  {ok} {row['id']:<16} {row['model']:<20} {row['total_s']:6.1f} s  "
              f"first token {row['time_to_answer_s']:5.1f} s  prompt {row['prompt_tokens']:>5} "
              f"(+{row['cached_tokens']} cached)  {row['tokens_per_s'] or 0:5.1f} tok/s")

    if compare:
        base = {r["id"]: r for r in json.loads(Path(compare).read_text())["rows"]}
        states = [{"question": r["question"], "reference": base[r["id"]]["answer"][:1500],
                   "answer": r["answer"][:1500]} for r in rows if r["id"] in base]
        results = pipeline.router().predict_batch([{"state": s, "questions": AGREES} for s in states])
        it = iter(results)
        for r in rows:
            if r["id"] in base:
                r["p_agrees"] = next(it)["answers"]["agrees"]["probabilities"]["A"]

    out = {"label": label, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "switches": switches,
           "models": {"small": config.SMALL_MODEL, "big": config.BIG_MODEL}, "summary": summarize(rows),
           "rows": rows}
    path = config.LOG_DIR / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print_summary(out, json.loads(Path(compare).read_text()) if compare else None)
    print(f"\nsaved {path}")
    return out


def summarize(rows):
    checked = [r["expect_ok"] for r in rows if r["expect_ok"] is not None]
    ctx = [r for r in rows if r["context_tokens"]]
    return {
        "n": len(rows),
        "expect_pass": f"{sum(checked)}/{len(checked)}",
        "total_s_sum": round(sum(r["total_s"] for r in rows), 1),
        "total_s_median": round(statistics.median(r["total_s"] for r in rows), 2),
        "first_token_s_median": round(statistics.median(r["time_to_answer_s"] for r in rows), 2),
        "context_first_token_s_median": round(statistics.median(r["time_to_answer_s"] for r in ctx), 2) if ctx else None,
        "prompt_tokens_sum": sum(r["prompt_tokens"] for r in rows),
        "cached_tokens_sum": sum(r["cached_tokens"] for r in rows),
        "p_bad_mean": round(statistics.mean(r["p_bad"] for r in rows if r["p_bad"] is not None), 3),
        "big_model_share": f"{sum(r['model'] == config.BIG_MODEL for r in rows)}/{len(rows)}",
        "retries": sum(r["retried"] for r in rows),
        **({"p_agrees_mean": round(statistics.mean(r["p_agrees"] for r in rows if "p_agrees" in r), 3)}
           if any("p_agrees" in r for r in rows) else {}),
    }


def print_summary(out, base=None):
    s = out["summary"]
    b = base["summary"] if base else {}
    print(f"\n{'':32}{out['label']:>14}" + (f"{base['label']:>14}" if base else ""))
    for k, v in s.items():
        print(f"  {k:30}{str(v):>14}" + (f"{str(b.get(k, '')):>14}" if base else ""))
