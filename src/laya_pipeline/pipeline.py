"""compress -> route -> generate -> check, with every Laya decision logged for fine-tuning."""

import math
import os
import subprocess
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("USE_TF", "0")  # README: TensorFlow's import probe can deadlock laya.load
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Laya clamps an out-of-range calibration temperature for 11+ option questions; we never ask those.
warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")

from openai import OpenAI

from . import config, lab, questions
from .log import log_decision, log_run

_lm = OpenAI(base_url=config.LLM_URL, api_key="local")
_router = None
_speed = dict(config.PROMPT_SPEED)  # prompt tokens/s per model, refined from real requests


def est_tokens(text):
    return round(len(text) / config.CHARS_PER_TOKEN)


def _batches(model, tokens):
    return math.ceil(tokens / config.UBATCH.get(model, 512))


def prompt_seconds(model, tokens):
    """Estimated time to read `tokens` new prompt tokens (see PROMPT_BATCH_COST_S in config)."""
    if tokens <= 0:
        return 0.0
    cost = config.PROMPT_BATCH_COST_S.get(model, 0.0) * _batches(model, tokens)
    return cost + tokens / _speed.get(model, 1000.0)


def _note_speed(model, timings):
    """Fold a request's measured prompt speed into the estimate (short prompts are too noisy)."""
    n, ms = timings.get("prompt_n", 0) if timings else 0, timings.get("prompt_ms", 0) if timings else 0
    if n < 256 or not ms:
        return
    net = ms / 1000 - config.PROMPT_BATCH_COST_S.get(model, 0.0) * _batches(model, n)
    if net > 0:
        _speed[model] = 0.7 * _speed.get(model, n / net) + 0.3 * n / net


def laya_device():
    if config.LAYA_DEVICE == "cuda16":
        return _gpu16["device"]
    if config.LAYA_DEVICE != "auto":
        return config.LAYA_DEVICE
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 2**30 >= config.LAYA_MIN_FREE_VRAM_GB else "cpu"


_gpu16 = {"device": "cpu"}  # where the cuda16 mode actually ended up


def _move_agents(r, to_gpu):
    """Put every loaded Laya checkpoint on the GPU in 16-bit, or back on the CPU in fp32."""
    import torch

    for agent in getattr(r, "_agents", {}).values():
        if to_gpu:
            dt = torch.bfloat16 if "bf" in str(agent.cfg.get("amp_dtype", "fp16")) else torch.float16
            agent.model.to(dt).to("cuda").eval()
            agent.device, agent.dtype, agent.amp_enabled = torch.device("cuda"), dt, True
        else:
            agent.model.to("cpu").to(torch.float32).eval()  # CPU first: fp32 on the GPU may not fit
            agent.device, agent.dtype, agent.amp_enabled = torch.device("cpu"), torch.float32, False
    if not to_gpu:
        torch.cuda.empty_cache()


class _Laya:
    """Laya's Router with the cuda16 placement, falling back to the CPU when the GPU is full."""

    def __init__(self, r):
        self.r = r

    def _call(self, fn, *args, **kwargs):
        if _gpu16["device"] == "cpu" and any(a.device.type == "cuda" for a in self.r._agents.values()):
            _move_agents(self.r, False)
        try:
            out = fn(*args, **kwargs)
        except RuntimeError as e:  # CUDA OOM, or Laya's fp32 retry after one meeting 16-bit weights
            if _gpu16["device"] == "cpu":
                raise
            print(f"[laya-pipeline] Laya on the GPU failed ({str(e)[:80]}); moving it to the CPU.")
            _gpu16["device"] = "cpu"
            _move_agents(self.r, False)
            return fn(*args, **kwargs)
        # A checkpoint loaded by this call (e.g. the multilingual one) arrives on the CPU.
        if _gpu16["device"] == "cuda" and any(a.device.type == "cpu" for a in self.r._agents.values()):
            _move_agents(self.r, True)
        return out

    def predict(self, *args, **kwargs):
        return self._call(self.r.predict, *args, **kwargs)

    def predict_batch(self, items, batch_size=16, **kwargs):
        # Long states (passages, answers) need ~150 MB of activations per item in a batch, and the
        # GPU has ~0.6 GB of headroom next to the LLMs.
        if _gpu16["device"] == "cuda":
            batch_size = min(batch_size, 4)
        return self._call(self.r.predict_batch, items, batch_size=batch_size, **kwargs)


def router():
    global _router
    if _router is None:
        from laya import Router

        if config.LAYA_DEVICE == "cuda16":
            import torch

            r = Router(device="cpu", max_loaded=1)
            r.predict("warm up", questions.ROUTE)  # loads the English checkpoint
            if torch.cuda.is_available():
                try:
                    _move_agents(r, True)
                    _gpu16["device"] = "cuda"
                    r.predict("warm up", questions.ROUTE)
                except RuntimeError as e:
                    print(f"[laya-pipeline] Laya didn't fit on the GPU ({str(e)[:80]}); using the CPU.")
                    _gpu16["device"] = "cpu"
                    _move_agents(r, False)
            _router = _Laya(r)
        else:
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


def embed_sims(question, chunks):
    """Cosine similarity of each chunk to the question (one embeddings request)."""
    # nomic-embed expects these task prefixes.
    inputs = [f"search_query: {question}"] + [f"search_document: {c}" for c in chunks]
    vecs = [d.embedding for d in _lm.embeddings.create(model=config.EMBED_MODEL, input=inputs).data]
    return [_cosine(vecs[0], v) for v in vecs[1:]]


def compress(question, chunks, trace, emit=_noop, budget=None, sims=None):
    """Keep the `budget` chunks (default KEEP_CHUNKS) most relevant to `question`.

    Embeddings rank everything; the top part of the budget is kept outright, and Laya decides
    which of the next borderline chunks fill the remaining slots. `sims` can be passed in when
    the embeddings were computed already (in parallel with routing).
    """
    budget = budget or config.KEEP_CHUNKS
    trace["compress"] = {"chunks_in": len(chunks), "budget": budget,
                         "tokens_in": est_tokens("".join(chunks))}
    if len(chunks) <= budget:
        trace["compress"].update(chunks_out=len(chunks), tokens_out=trace["compress"]["tokens_in"])
        emit("compress", {"chunks": [{"i": i, "text": c, "status": "kept"} for i, c in enumerate(chunks)],
                          "note": f"{len(chunks)} chunks fit the budget of {budget}, nothing to compress"})
        return chunks

    sims = sims if sims is not None else embed_sims(question, chunks)
    order = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)

    # With the fixed budget, the original split (2 sure + 4 for Laya); with Laya's scope budgets,
    # the top two-thirds by similarity are kept outright: answers to lookups often rank 3rd-4th,
    # and Laya tends to reject long table-like chunks that hold them.
    n_borderline = (config.BORDERLINE_CHUNKS if budget == config.KEEP_CHUNKS and not config.ADAPTIVE_BUDGET
                    else max(1, budget // 3))
    sure = max(budget - n_borderline, 1)
    keep = order[:sure]
    borderline = order[sure : sure + 2 * (budget - sure)]

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
    rescued = rescued[: budget - len(keep)]
    keep += rescued

    trace["compress"].update(chunks_out=len(keep), rescued_by_laya=len(rescued),
                             tokens_out=est_tokens("".join(chunks[i] for i in keep)))
    status = {i: "kept" for i in keep[:sure]} | {i: "rescued" for i in rescued}
    emit("compress", {
        "chunks": [{"i": i, "text": c, "sim": round(sims[i], 4), "rank": order.index(i),
                    "status": status.get(i, "rejected" if i in laya else "dropped"), **laya.get(i, {})}
                   for i, c in enumerate(chunks)],
        "threshold": config.RELEVANCE_MIN_PROB,
        "note": f"{len(chunks)} chunks -> {len(keep)} of a budget of {budget} ({len(rescued)} rescued by Laya)",
    })
    return [chunks[i] for i in sorted(keep)]  # original order reads better than score order


# ---- 2. route ----------------------------------------------------------------------------

def route(question, trace, scope=False):
    """Ask Laya about the request (and, with a document attached, how much of it is needed).

    Only reads the question, so it can run while the context is being embedded. The model is
    picked afterwards by choose_model, once the prompt length is known.
    """
    schema = {**questions.ROUTE, **(questions.SCOPE if scope else {})}
    t = time.perf_counter()
    raw = router().predict(question, schema)["answers"]
    laya_ms = round((time.perf_counter() - t) * 1000)
    # Log Laya's raw probabilities (calibration is refitted on them); decide on calibrated ones.
    ids = {name: log_decision(name, question, raw[name], trace["id"]) for name in schema}
    ans = {name: lab.calibrated(name, raw[name]) for name in schema}

    task = ans["task"]["choice"]
    sure = _prob(ans["task"], task) >= config.TASK_MIN_PROB
    d = {"task": task, "sure": sure, "task_probs": ans["task"]["probabilities"],
         "p_hard": _prob(ans["hard"], "A"), "hard": _prob(ans["hard"], "A") >= config.HARD_MIN_PROB,
         "temperature": config.TEMPERATURE[task] if sure else config.UNSURE_TEMPERATURE,
         "laya_ms": laya_ms, "decision_ids": ids,
         "calibrated": {name: lab.temperature(name) != 1.0 for name in schema}}
    if scope:
        sc = ans["scope"]["choice"]
        d.update(scope=sc, p_scope=_prob(ans["scope"], sc), scope_probs=ans["scope"]["probabilities"],
                 scope_sure=_prob(ans["scope"], sc) >= config.SCOPE_MIN_PROB)
    return d


def context_budget(d):
    """How many chunks to keep: by Laya's scope answer when it's sure, else the middle budget."""
    if not config.ADAPTIVE_BUDGET:
        return config.KEEP_CHUNKS
    return config.SCOPE_CHUNKS[d["scope"] if d.get("scope_sure") else "B"]


def choose_model(d, new_tokens):
    """Pick the model from Laya's answers and the prompt length. Returns (model, think, reason).

    Hard requests (and code Laya is sure about) go to the big model, unless reading a long
    prompt there would add more than MAX_EXTRA_WAIT_S and Laya isn't very sure it's hard:
    then the small model reads it fast and thinks instead.
    """
    if not (d["hard"] or (d["sure"] and d["task"] == "code")):
        return config.SMALL_MODEL, False, "easy"
    extra = prompt_seconds(config.BIG_MODEL, new_tokens) - prompt_seconds(config.SMALL_MODEL, new_tokens)
    if config.LENGTH_ROUTING and extra > config.MAX_EXTRA_WAIT_S and d["p_hard"] < config.STRONG_HARD_PROB:
        return (config.SMALL_MODEL, True,
                f"long prompt: ~{new_tokens} tokens would take {extra:.0f} s longer to read on the big model")
    return config.BIG_MODEL, d["hard"], "hard" if d["hard"] else "code"


def _emit_route(d, model, think, reason, new_tokens, trace, emit):
    task = d["task"]
    trace["route"] = {"task": task if d["sure"] else f"unsure ({task}?)", "hard": d["hard"], "model": model,
                      "think": think, "reason": reason, "p_task": d["task_probs"][task],
                      "p_hard": d["p_hard"], "prompt_tokens_est": new_tokens}
    if "scope" in d:
        trace["route"].update(scope=d["scope"], p_scope=d["p_scope"])
    emit("route", {
        "task_probs": d["task_probs"], "task": task, "sure": d["sure"],
        "p_hard": d["p_hard"], "hard": d["hard"], "calibrated": d["calibrated"],
        "model": model, "temperature": d["temperature"], "think": think, "laya_ms": d["laya_ms"],
        "reason": reason, "prompt_tokens_est": new_tokens,
        "est_prompt_s": {m: round(prompt_seconds(m, new_tokens), 1) for m in (config.SMALL_MODEL, config.BIG_MODEL)},
        **({"scope": d["scope"], "scope_probs": d["scope_probs"], "scope_sure": d["scope_sure"]} if "scope" in d else {}),
        "thresholds": {"task": config.TASK_MIN_PROB, "hard": config.HARD_MIN_PROB},
        "decision_ids": d["decision_ids"],
    })


# ---- 3. generate + 4. check --------------------------------------------------------------

def generate(model, messages, temperature, think, emit=_noop):
    """Stream the answer, emitting tokens as they arrive. Returns (answer, stats)."""
    # Thinking costs hundreds of hidden tokens per answer, so only hard requests get it.
    if config.BACKEND == "llamacpp":
        extra = {"chat_template_kwargs": {"enable_thinking": bool(think)}}
    else:
        extra = {} if think else {"reasoning_effort": "none"}
    t0 = time.perf_counter()
    first = timings = None
    parts, n_answer, n_reasoning = [], 0, 0
    stream = _lm.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                         extra_body=extra, stream=True)
    for ev in stream:
        timings = (ev.model_extra or {}).get("timings") or timings  # llama-server, last chunk
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
    if timings:
        # prompt_n = prompt tokens actually read; cache_n = reused from the prompt cache.
        stats.update(prompt_tokens=timings.get("prompt_n"), cached_tokens=timings.get("cache_n"),
                     prompt_ms=round(timings.get("prompt_ms", 0)),
                     prompt_per_s=round(timings.get("prompt_per_second", 0)),
                     tokens_per_s=round(timings.get("predicted_per_second", 0), 1) or stats["tokens_per_s"])
        _note_speed(model, timings)
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
    chunks = chunk(context) if context.strip() else []
    min_budget = min(config.SCOPE_CHUNKS.values()) if config.ADAPTIVE_BUDGET else config.KEEP_CHUNKS
    ask_scope = config.ADAPTIVE_BUDGET and len(chunks) > min_budget

    def do_route():
        with _Stage("route", emit, trace, t0):
            return route(question, trace, scope=ask_scope)

    if not chunks:
        emit("stage", {"stage": "compress", "status": "skipped"})
        d, kept = do_route(), []
    elif config.OVERLAP_ROUTE:
        # Laya routes on the CPU while the GPU embeds the chunks; the budget needs both.
        with ThreadPoolExecutor(1) as pool, _Stage("compress", emit, trace, t0):
            routing = pool.submit(do_route)
            sims = embed_sims(question, chunks) if len(chunks) > min_budget else None
            d = routing.result()
            kept = compress(question, chunks, trace, emit, context_budget(d), sims)
    else:
        d = do_route()
        with _Stage("compress", emit, trace, t0):
            kept = compress(question, chunks, trace, emit, context_budget(d))

    system = "You are a helpful, precise assistant."
    user = question
    temperature = d["temperature"]
    if kept:
        temperature = min(temperature, config.CONTEXT_TEMPERATURE)
        rule = "Answer from the context. If it does not contain the answer, say so instead of guessing."
        ctx = "\n\n---\n\n".join(kept)
        if config.CONTEXT_IN_USER:
            # System prompt and history stay byte-identical from turn to turn, so the server's
            # prompt cache can reuse them; only this last message is new.
            user = f"Context:\n\n{ctx}\n\n---\n\n{rule}\n\nQuestion: {question}"
        else:
            system += f"\n\n{rule}\n\n{ctx}"
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]

    new_tokens = est_tokens(user)
    model, think, reason = choose_model(d, new_tokens)
    _emit_route(d, model, think, reason, new_tokens, trace, emit)

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
