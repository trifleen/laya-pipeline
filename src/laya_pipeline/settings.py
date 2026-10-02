"""Settings saved from the dashboard's Lab tab: logs/settings.json.

Holds threshold overrides, per-question calibration temperatures and edited Laya questions.
Precedence for thresholds: environment variable > settings.json > default in config.py.
"""

import json
import threading

from . import config

_lock = threading.Lock()


def _path():
    return config.LOG_DIR / "settings.json"


def load():
    try:
        return json.loads(_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def update(section, key, value):
    """Set settings[section][key] = value, or remove it when value is None."""
    with _lock:
        data = load()
        sec = data.setdefault(section, {})
        if value is None:
            sec.pop(key, None)
        else:
            sec[key] = value
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
