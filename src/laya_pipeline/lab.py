"""The Lab: test Laya's questions on labelled examples, and calibrate its probabilities.

Labelled examples come from two places: decisions you labelled in the History panel
(logs/decisions.jsonl) and test cases added in the Lab (logs/lab_cases.jsonl).

Calibration is temperature scaling: p_i ∝ p_i^(1/T), one T per question. T > 1 softens
over-confident answers, T < 1 sharpens timid ones. It never changes which option wins,
only how sure Laya claims to be, which is what the thresholds act on.
"""

import json
import math
import threading
import uuid

from . import config, log, questions, settings

_lock = threading.Lock()

# Hand-labelled starter set (the prompts used to pick the "hard" wording); delete freely.
SEED_CASES = [
    ("hard", "What's the capital of Norway?", "B"),
    ("hard", "Hi, how are you?", "B"),
    ("hard", "Translate 'good morning' to Spanish.", "B"),
    ("hard", "Write a bash one-liner that lists files by size.", "B"),
    ("hard", "What does HTTP stand for?", "B"),
    ("hard", "A train leaves at 14:10 going 80 km/h; a second leaves at 14:40 going 120 km/h. "
             "When does it catch up? Show steps.", "A"),
    ("hard", "Design a database schema for a multi-tenant SaaS with billing, roles and audit logs, "
             "and explain the trade-offs.", "A"),
    ("hard", "Prove that the square root of 2 is irrational.", "A"),
    ("hard", "My Python async server deadlocks under load only when two clients upload at once. "
             "What could cause it and how do I debug it?", "A"),
    ("hard", "Compare three strategies for migrating a monolith to microservices for a 20-person "
             "team and recommend one.", "A"),
    ("task", "What's the capital of Norway?", "factual"),
    ("task", "Hi, how are you?", "chat"),
    ("task", "Write a bash one-liner that lists files by size.", "code"),
    ("task", "My Python async server deadlocks under load. How do I debug it?", "code"),
    ("task", "Write a short poem about autumn in Bergen.", "creative"),
    ("task", "What does HTTP stand for?", "factual"),
    ("scope", "What port does the server listen on by default?", "A"),
    ("scope", "Who is the author of this report?", "A"),
    ("scope", "What does the --n-cpu-moe flag do?", "A"),
    ("scope", "How do I configure authentication and what are the security caveats?", "B"),
    ("scope", "Which settings affect memory use, and how?", "B"),
    ("scope", "Explain how the caching works and when it gets invalidated.", "B"),
    ("scope", "Summarize this document.", "C"),
    ("scope", "Give me an overview of the whole design and its main trade-offs.", "C"),
    ("scope", "Review this paper: what are its strengths and weaknesses?", "C"),
]


# ---- labelled examples -----------------------------------------------------------------------

def _cases_path():
    return config.LOG_DIR / "lab_cases.jsonl"


def _load_cases():
    path = _cases_path()
    if not path.exists():
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        rows = [{"id": uuid.uuid4().hex[:12], "question": q, "state": s, "label": lab, "source": "seed"}
                for q, s, lab in SEED_CASES]
        _save_cases(rows)
        return rows
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Seed questions added after the file was created (deleting a question's seeds sticks only
    # while at least one case for that question is left).
    have = {r["question"] for r in rows}
    new = [{"id": uuid.uuid4().hex[:12], "question": q, "state": s, "label": lab, "source": "seed"}
           for q, s, lab in SEED_CASES if q not in have]
    if new:
        rows += new
        _save_cases(rows)
    return rows


def _save_cases(rows):
    _cases_path().write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def add_case(question, state, label):
    if label not in questions.all_questions()[question]["criteria"]:
        raise ValueError(f"{label!r} is not an option of {question}")
    with _lock:
        rows = _load_cases()
        row = {"id": uuid.uuid4().hex[:12], "question": question, "state": state, "label": label,
               "source": "lab"}
        rows.append(row)
        _save_cases(rows)
    return row


def delete_case(case_id):
    with _lock:
        rows = _load_cases()
        kept = [r for r in rows if r["id"] != case_id]
        _save_cases(kept)
    return len(kept) != len(rows)


def examples(question):
    """Every labelled example for one question: lab cases + labelled decisions (deduplicated)."""
    out, seen = [], set()
    for c in _load_cases():
        if c["question"] == question:
            out.append({"id": c["id"], "source": c["source"], "state": c["state"], "label": c["label"]})
            seen.add(json.dumps(c["state"], sort_keys=True))
    for d in log._load():
        if d["question"] != question or d["label"] is None:
            continue
        key = json.dumps(d["state"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": d.get("id"), "source": "run", "state": d["state"], "label": d["label"]})
    return out


_case_cache = {}


def case_predictions(question):
    """Laya's current (raw) probabilities for this question's test cases, cached until the
    cases or the question wording change."""
    rows = [c for c in _load_cases() if c["question"] == question]
    schema = questions.all_questions()[question]
    key = (json.dumps(schema, sort_keys=True), tuple(c["id"] for c in rows))
    if _case_cache.get(question, (None,))[0] != key:
        _case_cache[question] = (key, predict_all(question, schema, rows))
    return list(zip(rows, _case_cache[question][1]))


def counts():
    return {name: len(examples(name)) for name in questions.all_questions()}


# ---- evaluation ------------------------------------------------------------------------------

def predict_all(question, schema, rows):
    """Run Laya with `schema` on every example. Returns raw probability dicts."""
    from .pipeline import router

    if not rows:
        return []
    results = router().predict_batch(
        [{"state": r["state"], "questions": {question: schema}} for r in rows], batch_size=16)
    return [res["answers"][question]["probabilities"] for res in results]


def score(rows, probs, temperature=1.0):
    """Accuracy, log loss and per-example results for one set of predictions."""
    items, correct, nll = [], 0, 0.0
    for r, p in zip(rows, probs):
        p = scale(p, temperature)
        pred = max(p, key=p.get)
        correct += pred == r["label"]
        nll -= math.log(max(p.get(r["label"], 0.0), 1e-9))
        items.append({"id": r["id"], "source": r["source"], "state": r["state"], "label": r["label"],
                      "predicted": pred, "probabilities": p})
    n = max(len(rows), 1)
    return {"n": len(rows), "accuracy": correct / n, "log_loss": nll / n, "items": items}


def evaluate(question, schema=None):
    """Score the live question and, if given, an edited version on the same examples."""
    rows = examples(question)
    current = questions.all_questions()[question]
    out = {"current": score(rows, predict_all(question, current, rows), temperature(question))}
    if schema is not None:
        out["edited"] = score(rows, predict_all(question, schema, rows), temperature(question))
    return out


def save_question(question, schema):
    """Make an edited question live (None restores the default)."""
    if schema is not None:
        default = questions.DEFAULTS[question]
        if set(schema["criteria"]) != set(default["criteria"]):
            raise ValueError("Option keys can't change; edit their descriptions instead")
        schema = {"instructions": schema["instructions"], "criteria": schema["criteria"]}
    data = settings.update("questions", question, schema)
    questions.apply_overrides(data.get("questions", {}))
    # Calibration was fitted to the old wording.
    if settings.load().get("calibration", {}).get(question):
        set_temperature(question, None)


# ---- calibration -----------------------------------------------------------------------------

def scale(probs, t):
    if not t or t == 1.0:
        return dict(probs)
    logits = {k: math.log(max(v, 1e-9)) / t for k, v in probs.items()}
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def temperature(question):
    return settings.load().get("calibration", {}).get(question, {}).get("T", 1.0)


def temperatures():
    return {name: temperature(name) for name in questions.all_questions()}


def calibrated(question, answer):
    """Laya answer with calibrated probabilities (choice is unchanged by temperature scaling)."""
    t = temperature(question)
    if t == 1.0:
        return answer
    return {**answer, "probabilities": scale(answer["probabilities"], t)}


def reliability(rows, probs, t, bins=5):
    """Bucket predictions by confidence; compare to how often they were right."""
    buckets = [[] for _ in range(bins)]
    for r, p in zip(rows, probs):
        p = scale(p, t)
        pred = max(p, key=p.get)
        conf = p[pred]
        buckets[min(int(conf * bins), bins - 1)].append((conf, pred == r["label"]))
    out, ece, n = [], 0.0, max(len(rows), 1)
    for i, b in enumerate(buckets):
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
        out.append({"lo": i / bins, "hi": (i + 1) / bins, "n": len(b), "confidence": conf, "accuracy": acc})
    return {"bins": out, "ece": ece}


def fit(question):
    """Fit T by minimising log loss over a grid; report before/after."""
    rows = examples(question)
    if len(rows) < 8:
        raise ValueError(f"Need at least 8 labelled examples for {question}, have {len(rows)}")
    probs = predict_all(question, questions.all_questions()[question], rows)
    grid = [round(0.25 * 1.05 ** i, 4) for i in range(80)]  # 0.25 .. ~12
    best = min(grid, key=lambda t: score(rows, probs, t)["log_loss"])
    before, after = score(rows, probs, 1.0), score(rows, probs, best)
    return {
        "question": question, "n": len(rows), "T": best, "current_T": temperature(question),
        "before": {"log_loss": before["log_loss"], "accuracy": before["accuracy"],
                   **reliability(rows, probs, 1.0)},
        "after": {"log_loss": after["log_loss"], "accuracy": after["accuracy"],
                  **reliability(rows, probs, best)},
    }


def set_temperature(question, t):
    settings.update("calibration", question, None if t is None else {"T": float(t)})
