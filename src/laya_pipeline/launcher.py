"""`laya-pipeline setup` (one-time downloads) and `laya-pipeline up` (start everything)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import config
from .version import code_version


# ---- LM Studio -------------------------------------------------------------------------------

def find_lms():
    """The `lms` CLI: on PATH, or where LM Studio installs it."""
    found = shutil.which("lms")
    if found:
        return found
    for name in ("lms", "lms.exe"):
        candidate = Path.home() / ".lmstudio" / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


def _lms(lms, *args, timeout=30):
    return subprocess.run([lms, *args], capture_output=True, text=True, timeout=timeout)


def ensure_lmstudio_server():
    """Start LM Studio's API server if it isn't running. Returns False if that's impossible."""
    lms = find_lms()
    if not lms:
        print("✗ LM Studio's `lms` command not found. Install LM Studio (https://lmstudio.ai),\n"
              "  open it once, then run this again.")
        return False
    try:
        status = _lms(lms, "server", "status", timeout=15)
        if "is running" in status.stdout + status.stderr:
            return True
        print("Starting LM Studio server…")
        _lms(lms, "server", "start", timeout=60)
        return True
    except subprocess.TimeoutExpired:
        print("✗ LM Studio didn't respond. Open the LM Studio app, then run this again.")
        return False


# ---- setup -----------------------------------------------------------------------------------

def setup(args):
    """Download the LLMs and Laya's weights, then check everything."""
    if not ensure_lmstudio_server():
        sys.exit(1)
    lms = find_lms()

    installed = {m["modelKey"] for m in json.loads(_lms(lms, "ls", "--json").stdout or "[]")}
    wanted = [(config.SMALL_MODEL, config.SMALL_MODEL_DOWNLOAD), (config.BIG_MODEL, config.BIG_MODEL_DOWNLOAD),
              (config.EMBED_MODEL, config.EMBED_MODEL)]
    for key, download in wanted:
        if key in installed:
            print(f"✓ {key} already in LM Studio")
            continue
        print(f"↓ Downloading {download} with LM Studio (this can take a while)…")
        if subprocess.run([lms, "get", download, "-y"]).returncode != 0:
            print(f"✗ Download of {download} failed. Try it in the LM Studio app, then rerun setup.")
            sys.exit(1)

    print("↓ Loading Laya (downloads ~800 MB from Hugging Face the first time)…")
    from .pipeline import questions, router

    router().predict("warm up", questions.ROUTE)
    print("✓ Laya ready\n")

    from . import cmd_doctor
    cmd_doctor(args)
    print("\nAll set. Start the dashboard with:  ./laya   (or: uv run laya-pipeline up)")


# ---- up --------------------------------------------------------------------------------------

def _get(url, timeout=1.5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def open_window(url):
    """Omarchy opens it as an app window; everywhere else, the default browser."""
    if shutil.which("omarchy-launch-webapp"):
        subprocess.Popen(["omarchy-launch-webapp", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    else:
        webbrowser.open(url)


def _stop_running(url, pid):
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        try:
            _get(f"{url}/api/version", timeout=0.5)
        except Exception:
            return
        time.sleep(0.25)


def up(args):
    """Start LM Studio's server and the dashboard, open it, stay in the foreground."""
    url = f"http://localhost:{args.port}"
    ensure_lmstudio_server()  # the dashboard still starts without it, and says so in its header

    try:
        running = _get(f"{url}/api/version")
    except Exception:
        running = None
    if running:
        if running["version"] == code_version() and not args.restart:
            print(f"Dashboard already running at {url}")
            if not args.no_open:
                open_window(url)
            return
        print(f"Restarting the dashboard on port {args.port} "
              f"({'--restart' if args.restart else 'the code changed'})…")
        _stop_running(url, running["pid"])

    def open_when_ready():
        for _ in range(240):
            try:
                _get(f"{url}/api/status")
                break
            except Exception:
                time.sleep(0.25)
        if not args.no_open:
            open_window(url)
        print("Laya loads in the background (~10 s). Ctrl+C to stop.")

    threading.Thread(target=open_when_ready, daemon=True).start()
    from .server import serve

    try:
        serve(port=args.port)
    except OSError as e:
        print(f"✗ Couldn't use port {args.port} ({e}). Pick another with --port.")
        sys.exit(1)
    print("Dashboard stopped.")
