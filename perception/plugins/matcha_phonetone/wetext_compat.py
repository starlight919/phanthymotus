"""Compatibility helpers for WeText on Python 3.8."""

import sys


def ensure_wetext_compat() -> None:
    if sys.version_info >= (3, 9):
        return
    import importlib.resources as resources
    from importlib_resources import files

    if not hasattr(resources, "files"):
        resources.files = files
