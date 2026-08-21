"""Locations of frontend assets within the verified VITS2 model release."""

from pathlib import Path


_release_root: Path | None = None


def configure_release_paths(root: Path) -> None:
    """Configure the verified model-release root before frontend import."""
    global _release_root
    _release_root = root.resolve()


def _root() -> Path:
    if _release_root is None:
        raise RuntimeError("VITS2 frontend release paths are not configured")
    return _release_root


def frontend_data_dir() -> Path:
    return _root() / "frontend_data"


def nltk_data_dir() -> Path:
    return _root() / "nltk_data"


def tn_cache_dir() -> Path:
    return _root() / "tn_cache"
