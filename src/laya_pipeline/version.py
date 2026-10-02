"""Fingerprint of the code on disk, so `laya-pipeline up` can tell a stale dashboard apart."""

from pathlib import Path

PACKAGE = Path(__file__).parent


def code_version():
    """Newest source file's mtime in whole seconds (bytecode caches excluded)."""
    return str(int(max(f.stat().st_mtime for f in PACKAGE.rglob("*")
                       if f.is_file() and "__pycache__" not in f.parts)))
