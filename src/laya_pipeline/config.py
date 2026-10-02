"""Settings. Override any of these with environment variables of the same name.

The thresholds in THRESHOLDS can also be changed from the dashboard's Lab tab, which saves
them to logs/settings.json; an environment variable still wins over that.
"""

import json
import os
from pathlib import Path


def _env(name, default):
    return type(default)(os.environ.get(name, default))


LMSTUDIO_URL = _env("LMSTUDIO_URL", "http://localhost:1234/v1")
LLAMACPP_URL = _env("LLAMACPP_URL", "http://localhost:8080/v1")

# "llamacpp": one llama-server router serving all three models (see LLAMACPP_SERVE), with the
# big model a Mixture-of-Experts whose experts live in RAM. "lmstudio": the original setup.
BACKEND = _env("LAYA_BACKEND", "llamacpp")
LLM_URL = LLAMACPP_URL if BACKEND == "llamacpp" else LMSTUDIO_URL
# Script that starts the llama-server router; `laya up` runs it when nothing answers LLM_URL.
LLAMACPP_SERVE = _env("LLAMACPP_SERVE", str(Path(__file__).resolve().parents[2] / "serve" / "serve.sh"))

# Model IDs as the server lists them. LM Studio: as `lms ls` prints them; llama.cpp: the
# preset names in models.ini.
_MODELS = {
    "llamacpp": ("nemotron-3-nano-4b", "qwen3.6-35b-a3b", "nomic-embed-text-v1.5"),
    "lmstudio": ("nvidia/nemotron-3-nano-4b", "qwen/qwen3.5-9b", "text-embedding-nomic-embed-text-v1.5"),
}[BACKEND]
SMALL_MODEL = _env("SMALL_MODEL", _MODELS[0])
BIG_MODEL = _env("BIG_MODEL", _MODELS[1])
EMBED_MODEL = _env("EMBED_MODEL", _MODELS[2])

# What `laya-pipeline setup` downloads when a model is missing (`name@quant` picks a variant).
SMALL_MODEL_DOWNLOAD = _env("SMALL_MODEL_DOWNLOAD", "nvidia/nemotron-3-nano-4b")
BIG_MODEL_DOWNLOAD = _env("BIG_MODEL_DOWNLOAD", "qwen/qwen3.5-9b@q4_k_m")

# "cuda16": Laya's weights in 16-bit on the GPU (~0.9 GB of VRAM, 25-40x faster than the CPU:
# 18 ms per routing decision, 130 ms for six 350-token passages). It is loaded on the CPU and
# converted before moving, because the stock full-precision load needs ~1.7 GB. Falls back to
# the CPU if the GPU runs out of memory. "cpu" keeps all VRAM for the LLMs; "cuda" / "auto" are
# the stock full-precision placement.
LAYA_DEVICE = _env("LAYA_DEVICE", "cuda16")
LAYA_MIN_FREE_VRAM_GB = _env("LAYA_MIN_FREE_VRAM_GB", 2.0)

# Probability thresholds on Laya's answers.
HARD_MIN_PROB = _env("HARD_MIN_PROB", 0.6)  # route to the big model
BAD_ANSWER_MIN_PROB = _env("BAD_ANSWER_MIN_PROB", 0.7)  # retry on the big model

# Compression: chunk size in characters, how many chunks reach the LLM, and how many
# extra borderline chunks Laya gets to rescue.
CHUNK_CHARS = _env("CHUNK_CHARS", 1200)
KEEP_CHUNKS = _env("KEEP_CHUNKS", 6)
BORDERLINE_CHUNKS = _env("BORDERLINE_CHUNKS", 4)
RELEVANCE_MIN_PROB = _env("RELEVANCE_MIN_PROB", 0.6)

# Context budget by how much of the document Laya thinks the answer needs (the "scope"
# question): one passage / a few sections / most of it. Every chunk costs prompt-reading time,
# ~0.5 s per chunk on the big model, and the sizes keep the prompt just under one or two
# 2,048-token batches (a chunk is ~330 tokens). Unsure scope gets the middle budget.
SCOPE_CHUNKS = {"A": 3, "B": 5, "C": 11}
SCOPE_MIN_PROB = _env("SCOPE_MIN_PROB", 0.5)

# Prompt-reading time per model = BATCH_COST_S per batch of up to UBATCH tokens + tokens / SPEED.
# The MoE's batch cost is copying its RAM-resident experts to the GPU once per batch, so a
# 2,100-token prompt (2 batches) costs ~1.5 s more than a 2,000-token one. Measured 2026-10-02
# through llama-server on the RTX 3070 Laptop (bench/results/phase0); SPEED is refined
# from real requests.
PROMPT_SPEED = {SMALL_MODEL: 3300.0, BIG_MODEL: 800.0}  # big last: wins if both are one model
PROMPT_BATCH_COST_S = {SMALL_MODEL: 0.0, BIG_MODEL: 1.5}
UBATCH = {SMALL_MODEL: 512, BIG_MODEL: 2048}
CHARS_PER_TOKEN = 3.6  # rough estimate for English text and code
# A hard request still goes to the small model (with thinking) when the big model would make
# you wait this many extra seconds just to read the prompt, unless Laya is very sure it's hard.
MAX_EXTRA_WAIT_S = _env("MAX_EXTRA_WAIT_S", 8.0)
STRONG_HARD_PROB = _env("STRONG_HARD_PROB", 0.85)

# Switches for the speed-ups, so `laya-pipeline eval` can compare with and without them.
OVERLAP_ROUTE = _env("OVERLAP_ROUTE", 1)  # route while the context is being embedded
CONTEXT_IN_USER = _env("CONTEXT_IN_USER", 1)  # context in the last message: history stays cached
ADAPTIVE_BUDGET = _env("ADAPTIVE_BUDGET", 1)  # SCOPE_CHUNKS instead of a fixed KEEP_CHUNKS
LENGTH_ROUTING = _env("LENGTH_ROUTING", 1)  # MAX_EXTRA_WAIT_S rule

TEMPERATURE = {"code": 0.2, "factual": 0.2, "creative": 0.9, "chat": 0.7}
TASK_MIN_PROB = _env("TASK_MIN_PROB", 0.5)  # below this Laya's task guess is ignored...
UNSURE_TEMPERATURE = _env("UNSURE_TEMPERATURE", 0.4)  # ...and this neutral temperature is used
CONTEXT_TEMPERATURE = _env("CONTEXT_TEMPERATURE", 0.3)  # cap when answering from context files

LOG_DIR = Path(_env("LAYA_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs")))

THRESHOLDS = ["TASK_MIN_PROB", "HARD_MIN_PROB", "BAD_ANSWER_MIN_PROB", "RELEVANCE_MIN_PROB",
              "SCOPE_MIN_PROB"]
_DEFAULTS = {name: globals()[name] for name in THRESHOLDS}


def apply_saved_thresholds():
    """Re-read threshold overrides from logs/settings.json (called at import and after edits)."""
    try:
        saved = json.loads((LOG_DIR / "settings.json").read_text()).get("thresholds", {})
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    for name in THRESHOLDS:
        if name in os.environ:
            continue
        globals()[name] = float(saved.get(name, _DEFAULTS[name]))


apply_saved_thresholds()
