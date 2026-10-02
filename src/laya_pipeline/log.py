"""Decision log for later fine-tuning.

Every Laya answer is appended to logs/decisions.jsonl. `laya-pipeline review` walks the
unlabelled ones so you can confirm or correct them; `laya-pipeline export` writes the labelled
ones to logs/dataset.jsonl.
"""

import json
import threading
import time
import uuid

from . import config, questions

def schemas():
    return questions.all_questions()

_lock = threading.Lock()  # the dashboard labels from a web thread while the pipeline appends


def _path():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    return config.LOG_DIR / "decisions.jsonl"


def log_decision(question_name, state, answer, run_id):
    """Append one Laya answer; returns its id so it can be labelled later."""
    record = {
        "id": uuid.uuid4().hex[:12],
        "run": run_id,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question_name,
        "state": state,
        "predicted": answer["choice"],
        "confidence": answer["confidence"],
        "probabilities": answer.get("probabilities"),
        "label": None,
    }
    with _lock, _path().open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["id"]


def log_run(question, answer, trace):
    """Append a whole pipeline run (for the dashboard's history)."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": question,
              "answer": answer, "trace": trace}
    with _lock, (config.LOG_DIR / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def recent_runs(n=30):
    path = config.LOG_DIR / "runs.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-n:]
    return [json.loads(line) for line in reversed(lines) if line.strip()]


def label(decision_id, value):
    """Set the correct answer for one logged decision. Returns False if the id is unknown."""
    with _lock:
        records = _load()
        for r in records:
            if r.get("id") == decision_id:
                if value not in schemas()[r["question"]]["criteria"]:
                    raise ValueError(f"{value!r} is not an option of {r['question']}")
                r["label"] = value
                _save(records)
                return True
    return False


def label_stats():
    records = _load()
    labelled = [r for r in records if r["label"] is not None]
    return {"decisions": len(records), "labelled": len(labelled),
            "laya_wrong": sum(r["label"] != r["predicted"] for r in labelled)}


def _load():
    path = _path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _save(records):
    _path().write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def _show_state(state):
    if isinstance(state, str):
        return state[:800]
    return "\n".join(f"  {k}: {str(v)[:600]}" for k, v in state.items())


def review():
    records = _load()
    todo = [r for r in records if r["label"] is None]
    print(f"{len(todo)} unlabelled decisions. Enter = Laya was right, a key = correct answer, "
          "s = skip, q = quit.\n")
    for r in todo:
        criteria = schemas()[r["question"]]["criteria"]
        print("-" * 70)
        print(_show_state(r["state"]))
        print(f"\n{r['question']}: {schemas()[r['question']]['instructions']}")
        for key, desc in criteria.items():
            mark = "<- Laya" if key == r["predicted"] else ""
            print(f"  [{key}] {desc} {mark}")
        print(f"  (confidence {r['confidence']})")
        while True:
            reply = input("> ").strip()
            if reply in ("q", "s") or reply == "" or reply in criteria:
                break
            print(f"  type one of: {', '.join(criteria)}, Enter, s or q")
        if reply == "q":
            break
        if reply == "s":
            continue
        r["label"] = reply or r["predicted"]
        _save(records)  # save after each answer so quitting loses nothing
    labelled = [r for r in records if r["label"] is not None]
    wrong = sum(r["label"] != r["predicted"] for r in labelled)
    print(f"\n{len(labelled)} labelled, Laya wrong on {wrong}.")


def export():
    labelled = [r for r in _load() if r["label"] is not None]
    out = config.LOG_DIR / "dataset.jsonl"
    with out.open("w") as f:
        for r in labelled:
            row = {"state": r["state"],
                   "questions": {r["question"]: schemas()[r["question"]]},
                   "answers": {r["question"]: {"choice": r["label"]}}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labelled)} examples to {out}")
