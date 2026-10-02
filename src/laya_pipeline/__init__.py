"""laya-pipeline: Laya routing and context compression in front of local models (llama.cpp or LM Studio)."""

import argparse
import json
import sys
from pathlib import Path


def _print_trace(trace):
    print(f"\n\033[2m[{json.dumps(trace)}]\033[0m", file=sys.stderr)


def cmd_ask(args):
    from .pipeline import ask

    context = "\n\n".join(Path(p).read_text() for p in args.context)
    answer, trace = ask(args.question, context)
    print(answer)
    _print_trace(trace)


def cmd_chat(args):
    from .pipeline import ask

    context = "\n\n".join(Path(p).read_text() for p in args.context)
    history = []
    print("Chat through the Laya pipeline. Ctrl+D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        answer, trace = ask(question, context, history[-6:])
        print(f"\n{answer}")
        _print_trace(trace)
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]


def cmd_doctor(args):
    from . import config
    from .pipeline import _lm, laya_device

    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU {torch.cuda.get_device_name()}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
    print(f"Laya will run on: {laya_device()}")
    try:
        ids = {m.id for m in _lm.models.list().data}
    except Exception as e:
        print(f"Model server ({config.BACKEND}) not reachable at {config.LLM_URL}: {e}")
        print("Start it with: ./laya up" if config.BACKEND == "llamacpp" else "Start it with: lms server start")
        return
    for name in (config.SMALL_MODEL, config.BIG_MODEL, config.EMBED_MODEL):
        print(f"{'ok' if name in ids else 'MISSING':8}{name}")


def cmd_dashboard(args):
    from .server import serve

    serve(port=args.port)


def cmd_up(args):
    from .launcher import up

    up(args)


def cmd_setup(args):
    from .launcher import setup

    setup(args)


def main():
    p = argparse.ArgumentParser(prog="laya-pipeline", description=__doc__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("setup", help="one-time: download the models and Laya, then check everything")
    s.set_defaults(fn=cmd_setup)

    u = sub.add_parser("up", help="start LM Studio's server + the dashboard and open it (Ctrl+C stops)")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-open", action="store_true", help="don't open a browser window")
    u.add_argument("--restart", action="store_true", help="replace a dashboard already running")
    u.set_defaults(fn=cmd_up)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("-c", "--context", action="append", default=[], help="text file to use as context")
    a.set_defaults(fn=cmd_ask)

    c = sub.add_parser("chat", help="interactive chat")
    c.add_argument("-c", "--context", action="append", default=[], help="text file to use as context")
    c.set_defaults(fn=cmd_chat)

    sub.add_parser("doctor", help="check GPU and LM Studio").set_defaults(fn=cmd_doctor)

    d = sub.add_parser("dashboard", help="live web dashboard at http://localhost:8765")
    d.add_argument("--port", type=int, default=8765)
    d.set_defaults(fn=cmd_dashboard)

    e = sub.add_parser("eval", help="score speed and quality on eval/cases.jsonl (see evaluate.py)")
    e.add_argument("--label", default="run", help="name for this run, e.g. baseline")
    e.add_argument("--only", nargs="*", help="case ids to run")
    e.add_argument("--compare", help="earlier eval JSON: show side by side and judge agreement")
    e.set_defaults(fn=lambda a: __import__("laya_pipeline.evaluate", fromlist=["run"]).run(a.label, a.only, a.compare))

    from .log import export, review
    sub.add_parser("review", help="label logged Laya decisions").set_defaults(fn=lambda _: review())
    sub.add_parser("export", help="write labelled decisions to logs/dataset.jsonl").set_defaults(
        fn=lambda _: export())

    args = p.parse_args()
    args.fn(args)
