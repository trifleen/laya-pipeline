"""compress -> route -> generate -> check, with every Laya decision logged for fine-tuning."""

import math
import os
import subprocess
import threading
import time
import uuid
import warnings

os.environ.setdefault("USE_TF", "0")  # README: TensorFlow's import probe can deadlock laya.load
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Laya clamps an out-of-range calibration temperature for 11+ option questions; we never ask those.
warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")

from openai import OpenAI

from . import config, lab, questions
from .log import log_decision, log_run

_lm = OpenAI(base_url=config.LMSTUDIO_URL, api_key="lm-studio")
_router = None


def laya_device():
    if config.LAYA_DEVICE != "auto":
        return config.LAYA_DEVICE
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 2**30 >= config.LAYA_MIN_FREE_VRAM_GB else "cpu"


def router():
    global _router
    if _router is None:
        from laya import Router

        device = laya_device()
        # English and multilingual checkpoints both stay loaded on CPU (RAM is plentiful);
        # on the GPU keep one so Laya doesn't crowd out the LLM.
        _router = Router(device=device, max_loaded=1 if device == "cuda" else 2)
    return _router


def _noop(event, data):
    pass


def _prob(answer, key):
    """Probability Laya gives option `key`. (Its `confidence` field is an entropy score, not this.)"""
    return answer["probabilities"][key]


# ---- 1. compress -------------------------------------------------------------------------

def chunk(text, size=None):
    """Split on paragraphs, then pack paragraphs into chunks of about `size` characters."""
    size = size or config.CHUNK_CHARS
    chunks, cur = [], ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        while len(para) > size:  # a single huge paragraph gets cut hard
            chunks.append(para[:size])
            para = para[size:]
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def compress(question, chunks, trace, emit=_noop):
    """Keep the chunks most relevant to `question`.

    Embeddings rank everything; the top KEEP_CHUNKS - BORDERLINE_CHUNKS are kept outright,
    and Laya decides which of the next 2 * BORDERLINE_CHUNKS fill the remaining slots.
    """
    if len(chunks) <= config.KEEP_CHUNKS:
        trace["compress"] = {"chunks_in": len(chunks), "chunks_out": len(chunks)}
        emit("compress", {"chunks": [{"i": i, "text": c, "status": "kept"} for i, c in enumerate(chunks)],
                          "note": f"{len(chunks)} chunks fit, nothing to compress"})
        return chunks

    # nomic-embed expects these task prefixes.
    inputs = [f"search_query: {question}"] + [f"search_document: {c}" for c in chunks]
    vecs = [d.embedding for d in _lm.embeddings.create(model=config.EMBED_MODEL, input=inputs).data]
    sims = [_cosine(vecs[0], v) for v in vecs[1:]]
    order = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)

    sure = max(config.KEEP_CHUNKS - config.BORDERLINE_CHUNKS, 1)
    keep = order[:sure]
    borderline = order[sure : sure + 2 * config.BORDERLINE_CHUNKS]

    results = router().predict_batch(
        [{"state": {"question": question, "passage": chunks[i]}, "questions": questions.RELEVANT}
         for i in borderline]
    )
    rescued, laya = [], {}
    for i, res in zip(borderline, results):
        raw = res["answers"]["relevant"]
        did = log_decision("relevant", {"question": question, "passage": chunks[i]}, raw, trace["id"])
        ans = lab.calibrated("relevant", raw)
        laya[i] = {"p_relevant": _prob(ans, "A"), "decision_id": did}
        if _prob(ans, "A") >= config.RELEVANCE_MIN_PROB:
            rescued.append(i)
    rescued = rescued[: config.KEEP_CHUNKS - len(keep)]
    keep += rescued

    trace["compress"] = {"chunks_in": len(chunks), "chunks_out": len(keep),
                         "rescued_by_laya": len(rescued)}
    status = {i: "kept" for i in keep[:sure]} | {i: "rescued" for i in rescued}
    emit("compress", {
        "chunks": [{"i": i, "text": c, "sim": round(sims[i], 4), "rank": order.index(i),
                    "status": status.get(i, "rejected" if i in laya else "dropped"), **laya.get(i, {})}
                   for i, c in enumerate(chunks)],
        "threshold": config.RELEVANCE_MIN_PROB,
        "note": f"{len(chunks)} chunks -> {len(keep)} ({len(rescued)} rescued by Laya)",
    })
    return [chunks[i] for i in sorted(keep)]  # original order reads better than score order


# ---- 2. route ----------------------------------------------------------------------------

def route(question, trace, emit=_noop):
    t = time.perf_counter()
    raw = router().predict(question, questions.ROUTE)["answers"]
    laya_ms = round((time.perf_counter() - t) * 1000)
    # Log Laya's raw probabilities (calibration is refitted on them); decide on calibrated ones.
    ids = {name: log_decision(name, question, raw[name], trace["id"]) for name in questions.ROUTE}
    ans = {name: lab.calibrated(name, raw[name]) for name in questions.ROUTE}

    task = ans["task"]["choice"]
    sure = _prob(ans["task"], task) >= config.TASK_MIN_PROB
    hard = _prob(ans["hard"], "A") >= config.HARD_MIN_PROB
    model = config.BIG_MODEL if hard or (sure and task == "code") else config.SMALL_MODEL
    temperature = config.TEMPERATURE[task] if sure else config.UNSURE_TEMPERATURE
    trace["route"] = {"task": task if sure else f"unsure ({task}?)", "hard": hard, "model": model,
                      "think": hard, "p_task": _prob(ans["task"], task), "p_hard": _prob(ans["hard"], "A")}
    emit("route", {
        "task_probs": ans["task"]["probabilities"], "task": task, "sure": sure,
        "p_hard": _prob(ans["hard"], "A"), "hard": hard,
        "calibrated": {name: lab.temperature(name) != 1.0 for name in questions.ROUTE},
        "model": model, "temperature": temperature, "think": hard, "laya_ms": laya_ms,
        "thresholds": {"task": config.TASK_MIN_PROB, "hard": config.HARD_MIN_PROB},
        "decision_ids": ids,
    })
    return model, temperature, hard


# ---- 3. generate + 4. check --------------------------------------------------------------

def generate(model, messages, temperature, think, emit=_noop):
    """Stream the answer, emitting tokens as they arrive. Returns (answer, stats)."""
    # Thinking costs hundreds of hidden tokens per answer, so only hard requests get it.
    extra = {} if think else {"reasoning_effort": "none"}
    t0 = time.perf_counter()
    first = None
    parts, n_answer, n_reasoning = [], 0, 0
    stream = _lm.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                         extra_body=extra, stream=True)
    for ev in stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if first is None and (reasoning or delta.content):
            first = time.perf_counter()
            emit("model_loaded", {"model": model, "wait_ms": round((first - t0) * 1000)})
        if reasoning:
            n_reasoning += 1
            emit("reasoning", {"text": reasoning})
        if delta.content:
            n_answer += 1
            parts.append(delta.content)
            emit("token", {"text": delta.content})
    end = time.perf_counter()
    first = first or end
    gen_s = max(end - first, 1e-6)
    stats = {"model": model, "first_token_ms": round((first - t0) * 1000),
             "answer_tokens": n_answer, "reasoning_tokens": n_reasoning,
             "tokens_per_s": round((n_answer + n_reasoning) / gen_s, 1)}
    return "".join(parts), stats


def check(question, answer, trace, emit=_noop):
    state = {"question": question, "answer": answer[:1500]}
    raw = router().predict(state, questions.CHECK)["answers"]["answers"]
    did = log_decision("answers", state, raw, trace["id"])
    ans = lab.calibrated("answers", raw)
    ok = _prob(ans, "B") < config.BAD_ANSWER_MIN_PROB
    trace.setdefault("checks", []).append({"ok": ok, "p_bad": _prob(ans, "B")})
    emit("check", {"ok": ok, "p_bad": _prob(ans, "B"), "threshold": config.BAD_ANSWER_MIN_PROB,
                   "attempt": len(trace["checks"]), "decision_id": did})
    return ok


class _Stage:
    """Times a pipeline stage and tells the listener when it starts and ends."""

    def __init__(self, name, emit, trace, t0):
        self.name, self.emit, self.trace, self.t0 = name, emit, trace, t0

    def __enter__(self):
        self.t = time.perf_counter()
        self.emit("stage", {"stage": self.name, "status": "running",
                            "start_ms": round((self.t - self.t0) * 1000)})

    def __exit__(self, exc_type, *_):
        ms = round((time.perf_counter() - self.t) * 1000)
        start = round((self.t - self.t0) * 1000)
        self.trace.setdefault("stage_ms", {})[self.name] = ms
        self.trace.setdefault("stages", []).append({"stage": self.name, "start_ms": start, "ms": ms})
        self.emit("stage", {"stage": self.name, "status": "error" if exc_type else "done",
                            "start_ms": start, "ms": ms})


class _GpuSampler:
    """Samples VRAM, power and load every 250 ms during a run (one long-lived nvidia-smi)."""

    def __init__(self, emit, t0):
        self.emit, self.t0, self.samples, self.proc = emit, t0, [], None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "250"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            return self
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                mem, power, util = (float(x) for x in line.split(","))
            except ValueError:
                continue
            s = {"t_ms": round((time.perf_counter() - self.t0) * 1000), "vram_mb": mem,
                 "power_w": power, "util": util}
            self.samples.append(s)
            self.emit("gpu", s)

    def __exit__(self, *_):
        if self.proc:
            self.proc.terminate()


def ask(question, context="", history=(), emit=_noop):
    """Run the whole pipeline. Returns (answer, trace).

    `emit(event, data)` receives live progress: stage, compress, route, model_loaded,
    reasoning, token, check, retry, gpu. The CLI ignores it; the dashboard streams it.
    """
    trace = {"id": uuid.uuid4().hex[:12]}
    t0 = time.perf_counter()
    emit("start", {"id": trace["id"], "question": question})
    with _GpuSampler(emit, t0) as gpu:
        answer = _run(question, context, history, emit, trace, t0)
    trace["gpu"] = gpu.samples
    trace["seconds"] = round(time.perf_counter() - t0, 2)
    log_run(question, answer, trace)
    return answer, trace


def _run(question, context, history, emit, trace, t0):
    if context.strip():
        with _Stage("compress", emit, trace, t0):
            kept = compress(question, chunk(context), trace, emit)
    else:
        kept = []
        emit("stage", {"stage": "compress", "status": "skipped"})
    with _Stage("route", emit, trace, t0):
        model, temperature, think = route(question, trace, emit)

    system = "You are a helpful, precise assistant."
    if kept:
        temperature = min(temperature, config.CONTEXT_TEMPERATURE)
        system += ("\n\nAnswer from the context below. If it does not contain the answer, say so "
                   "instead of guessing.\n\n" + "\n\n---\n\n".join(kept))
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": question}]

    with _Stage("generate", emit, trace, t0):
        answer, trace["generate"] = generate(model, messages, temperature, think, emit)
    with _Stage("check", emit, trace, t0):
        ok = check(question, answer, trace, emit)

    if not ok and (model, think) != (config.BIG_MODEL, True):
        trace["retried_with"] = f"{config.BIG_MODEL} + thinking"
        emit("retry", {"model": config.BIG_MODEL, "think": True})
        with _Stage("retry", emit, trace, t0):
            answer, trace["retry"] = generate(config.BIG_MODEL, messages, temperature, True, emit)
            check(question, answer, trace, emit)
    else:
        emit("stage", {"stage": "retry", "status": "skipped"})
    return answer
