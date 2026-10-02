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

# Model IDs as `lms ls` prints them. Any LM Studio models work; these fit an 8 GB GPU.
SMALL_MODEL = _env("SMALL_MODEL", "nvidia/nemotron-3-nano-4b")
BIG_MODEL = _env("BIG_MODEL", "qwen/qwen3.5-9b")
EMBED_MODEL = _env("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")  # bundled with LM Studio

# What `laya-pipeline setup` downloads when a model is missing (`name@quant` picks a variant).
SMALL_MODEL_DOWNLOAD = _env("SMALL_MODEL_DOWNLOAD", "nvidia/nemotron-3-nano-4b")
BIG_MODEL_DOWNLOAD = _env("BIG_MODEL_DOWNLOAD", "qwen/qwen3.5-9b@q4_k_m")

# "cpu" keeps all 8 GB of VRAM for the LLM. "auto" uses the GPU when enough VRAM is free at
# startup, but a bigger LLM loaded later can then fail to fit next to it.
LAYA_DEVICE = _env("LAYA_DEVICE", "cpu")
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

TEMPERATURE = {"code": 0.2, "factual": 0.2, "creative": 0.9, "chat": 0.7}
TASK_MIN_PROB = _env("TASK_MIN_PROB", 0.5)  # below this Laya's task guess is ignored...
UNSURE_TEMPERATURE = _env("UNSURE_TEMPERATURE", 0.4)  # ...and this neutral temperature is used
CONTEXT_TEMPERATURE = _env("CONTEXT_TEMPERATURE", 0.3)  # cap when answering from context files

LOG_DIR = Path(_env("LAYA_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs")))

THRESHOLDS = ["TASK_MIN_PROB", "HARD_MIN_PROB", "BAD_ANSWER_MIN_PROB", "RELEVANCE_MIN_PROB"]
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
