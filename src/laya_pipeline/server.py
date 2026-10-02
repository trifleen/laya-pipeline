"""Live dashboard: `laya-pipeline dashboard`, then open http://localhost:8765."""

import asyncio
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, lab, log, pipeline, questions, settings
from .version import code_version

STATIC = Path(__file__).parent / "static"

CODE_VERSION = code_version()  # lets `laya-pipeline up` spot a stale dashboard

_busy = threading.Lock()  # one pipeline run at a time: there is one GPU
_history = []  # conversation turns sent to the LLM, like `laya-pipeline chat`
_laya = {"state": "loading"}


def _warm_up():
    """Load Laya so the first question doesn't wait ~10 s for it."""
    try:
        pipeline.router().predict("warm up", pipeline.questions.ROUTE)
        _laya.update(state="ready", device=pipeline.laya_device())
    except Exception as e:
        _laya.update(state="error", error=str(e))


@asynccontextmanager
async def _lifespan(app):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class Ask(BaseModel):
    question: str
    context: str = ""
    remember: bool = True


@app.post("/api/ask")
async def ask(req: Ask):
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "A question is already running")

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def emit(event, data):
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def run():
        try:
            history = _history[-6:] if req.remember else []
            answer, trace = pipeline.ask(req.question, req.context, history, emit)
            if req.remember:
                _history.extend([{"role": "user", "content": req.question},
                                 {"role": "assistant", "content": answer}])
            emit("done", {"answer": answer, "trace": trace})
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            _busy.release()
            emit(None, None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        while True:
            event, data = await queue.get()
            if event is None:
                return
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/reset")
def reset():
    _history.clear()
    return {"ok": True}


def _gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        name, used, total, util, temp, power = [x.strip() for x in out.strip().split(",")]
        return {"name": name, "used_mb": int(used), "total_mb": int(total), "util": int(util),
                "temp_c": int(temp), "power_w": float(power)}
    except Exception:
        return None


def _lmstudio():
    """Model server status (LM Studio or the llama-server router), in one shape for the header."""
    base = config.LLM_URL.removesuffix("/v1")
    try:
        if config.BACKEND == "llamacpp":
            data = httpx.get(f"{base}/models", timeout=2).json()["data"]
            return {"up": True, "models": [
                {"id": m["id"], "state": "loaded" if (m.get("status") or {}).get("value") == "loaded" else "not-loaded",
                 "type": "embeddings" if "embed" in m["id"] else "llm", "quant": None} for m in data]}
        data = httpx.get(f"{base}/api/v0/models", timeout=2).json()["data"]
        return {"up": True, "models": [{"id": m["id"], "state": m.get("state"), "type": m.get("type"),
                                        "quant": m.get("quantization")} for m in data]}
    except Exception:
        return {"up": False, "models": []}


@app.get("/api/status")
def status():
    return {
        "version": CODE_VERSION, "gpu": _gpu(), "lmstudio": _lmstudio(), "laya": _laya, "busy": _busy.locked(),
        "labels": log.label_stats(), "turns": len(_history) // 2,
        "config": {"small": config.SMALL_MODEL, "big": config.BIG_MODEL,
                   "hard_min_prob": config.HARD_MIN_PROB, "task_min_prob": config.TASK_MIN_PROB,
                   "bad_answer_min_prob": config.BAD_ANSWER_MIN_PROB,
                   "relevance_min_prob": config.RELEVANCE_MIN_PROB, "keep_chunks": config.KEEP_CHUNKS,
                   "backend": config.BACKEND},
    }


@app.get("/api/history")
def history():
    runs = log.recent_runs(40)
    by_run = {}
    for d in log._load():
        if d.get("id") and d["question"] != "relevant":
            by_run.setdefault(d["run"], []).append(
                {k: d[k] for k in ("id", "question", "predicted", "probabilities", "label")})
    for r in runs:
        r["decisions"] = by_run.get(r["trace"]["id"], [])
    return {"runs": runs, "options": {name: list(q["criteria"]) for name, q in log.schemas().items()},
            "temperatures": lab.temperatures(), "thresholds": _thresholds()}


class Label(BaseModel):
    id: str
    value: str


@app.post("/api/label")
def set_label(req: Label):
    try:
        found = log.label(req.id, req.value)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not found:
        raise HTTPException(404, "Unknown decision id")
    return log.label_stats()


# ---- Lab -------------------------------------------------------------------------------------

def _thresholds():
    return {name: getattr(config, name) for name in config.THRESHOLDS}


def _laya_job(fn, *args):
    """Lab jobs use Laya too, so they wait for no running question and block new ones."""
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "A question is running; try again when it finishes")
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        _busy.release()


@app.get("/api/lab")
def lab_data():
    saved = settings.load()
    return {
        "questions": questions.all_questions(), "defaults": questions.DEFAULTS,
        "edited": sorted(saved.get("questions", {})), "counts": lab.counts(),
        "temperatures": lab.temperatures(), "thresholds": _thresholds(),
        "threshold_defaults": config._DEFAULTS, "env_locked": [n for n in config.THRESHOLDS if n in os.environ],
        "cases": lab._load_cases(),
    }


@app.get("/api/lab/decisions")
def lab_decisions():
    """Every logged Laya decision with raw probabilities, for the what-if sliders."""
    out = []
    for d in log._load():
        if not d.get("probabilities"):
            continue
        st = d["state"]
        if isinstance(st, str):
            text = st
        else:  # relevance: the passage matters; check: the answer
            text = f"{st.get('question', '')} → {st.get('passage') or st.get('answer') or ''}"
        out.append({"id": d.get("id"), "run": d["run"], "question": d["question"], "time": d["time"],
                    "probabilities": d["probabilities"], "label": d["label"], "text": text[:200]})
    # Test cases have labels but no logged probabilities: score them now (cached), unless a
    # question is running, in which case they're left out of this refresh.
    if _busy.acquire(blocking=False):
        try:
            for name in questions.all_questions():
                for c, probs in lab.case_predictions(name):
                    out.append({"id": c["id"], "run": None, "question": name, "time": None, "source": c["source"],
                                "probabilities": probs, "label": c["label"], "text": f"[test case] {c['state'][:190]}"})
        finally:
            _busy.release()
    return out


class Schema(BaseModel):
    instructions: str
    criteria: dict[str, str]


class Evaluate(BaseModel):
    question: str
    schema_: Schema | None = Field(None, alias="schema")


@app.post("/api/lab/evaluate")
def lab_evaluate(req: Evaluate):
    schema = None
    if req.schema_:
        schema = {**questions.all_questions()[req.question], **req.schema_.model_dump()}
    return _laya_job(lab.evaluate, req.question, schema)


class SaveQuestion(BaseModel):
    question: str
    schema_: Schema | None = Field(None, alias="schema")  # None = restore default


@app.post("/api/lab/question")
def lab_save_question(req: SaveQuestion):
    try:
        lab.save_question(req.question, req.schema_.model_dump() if req.schema_ else None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return lab_data()


class Case(BaseModel):
    question: str
    state: str
    label: str


@app.post("/api/lab/cases")
def lab_add_case(req: Case):
    try:
        lab.add_case(req.question, req.state, req.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return lab_data()


@app.delete("/api/lab/cases/{case_id}")
def lab_delete_case(case_id: str):
    lab.delete_case(case_id)
    return lab_data()


class Calibrate(BaseModel):
    question: str


@app.post("/api/lab/calibrate")
def lab_calibrate(req: Calibrate):
    return _laya_job(lab.fit, req.question)


class Temperature(BaseModel):
    question: str
    T: float | None  # None = remove calibration


@app.post("/api/lab/temperature")
def lab_temperature(req: Temperature):
    lab.set_temperature(req.question, req.T)
    return lab_data()


class Thresholds(BaseModel):
    values: dict[str, float | None]  # None = back to default


@app.post("/api/lab/thresholds")
def lab_thresholds(req: Thresholds):
    for name, value in req.values.items():
        if name not in config.THRESHOLDS:
            raise HTTPException(400, f"Unknown threshold {name}")
        if value is not None and not 0 <= value <= 1:
            raise HTTPException(400, f"{name} must be between 0 and 1")
        settings.update("thresholds", name, value)
    config.apply_saved_thresholds()
    return lab_data()


# ---- Stats -----------------------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    runs = []
    path = config.LOG_DIR / "runs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t = r["trace"]
            g = t.get("retry") or t.get("generate") or {}
            runs.append({
                "time": r["time"], "task": (t.get("route") or {}).get("task"),
                "hard": (t.get("route") or {}).get("hard"), "model": (t.get("route") or {}).get("model"),
                "final_model": g.get("model"), "retried": "retried_with" in t, "seconds": t.get("seconds"),
                "tps": g.get("tokens_per_s"), "first_token_ms": g.get("first_token_ms"),
                "answer_tokens": g.get("answer_tokens"), "reasoning_tokens": g.get("reasoning_tokens"),
                "compressed": t.get("compress"),
            })
    labelled = [{"time": d["time"], "question": d["question"], "correct": d["label"] == d["predicted"]}
                for d in log._load() if d["label"] is not None]
    return {"runs": runs, "labelled": labelled}


@app.get("/api/version")
def version():
    return {"version": CODE_VERSION, "pid": os.getpid()}


def serve(host="127.0.0.1", port=8765):
    import uvicorn

    print(f"Dashboard: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
