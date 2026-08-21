"""Run the release-controlled legacy Chinese TN FSTs without Pynini.

The graph files are produced by the existing TN release process and live in a
model release's ``tn_cache`` directory.  This adapter only executes them with
the lightweight ``kaldifst`` runtime; it does not build or modify TN rules.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import kaldifst
from wetext import token_parser


class LegacyFstReleaseError(ValueError):
    """Raised when a model TN release is incomplete or fails verification."""


class LegacyFstNormalizer:
    """Execute the deployed ``zh_tn`` tagger/verbalizer graph pair."""

    def __init__(self, release_dir: Union[str, Path]):
        root = Path(release_dir).resolve()
        manifest_path = root / "tn_manifest.json"
        if not manifest_path.is_file():
            raise LegacyFstReleaseError(f"missing TN manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LegacyFstReleaseError(f"invalid TN manifest: {exc}") from exc

        graphs = manifest.get("fst")
        if manifest.get("schema_version") != 1 or not isinstance(graphs, dict):
            raise LegacyFstReleaseError("unsupported TN manifest schema")
        self._tagger_path = self._verified_path(root, graphs, "zh_tn_tagger.fst")
        self._verbalizer_path = self._verified_path(root, graphs, "zh_tn_verbalizer.fst")

    @staticmethod
    def _verified_path(root: Path, graphs: dict, filename: str) -> str:
        metadata = graphs.get(filename)
        if not isinstance(metadata, dict):
            raise LegacyFstReleaseError(f"TN manifest does not declare {filename}")
        path = (root / filename).resolve()
        if root not in path.parents or not path.is_file():
            raise LegacyFstReleaseError(f"missing TN graph: {filename}")
        expected = metadata.get("sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not isinstance(expected, str) or expected != actual:
            raise LegacyFstReleaseError(f"TN graph checksum mismatch: {filename}")
        expected_size = metadata.get("bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise LegacyFstReleaseError(f"TN graph size mismatch: {filename}")
        return str(path)

    @property
    @lru_cache(maxsize=1)
    def _tagger(self):
        return kaldifst.TextNormalizer(self._tagger_path)

    @property
    @lru_cache(maxsize=1)
    def _verbalizer(self):
        return kaldifst.TextNormalizer(self._verbalizer_path)

    def normalize(self, text: str) -> str:
        """Return legacy-TN output without trimming or fallback rewriting."""
        if not text:
            return text
        tagged = self._tagger(text)
        # Legacy graphs expect raw token values.  The released wetext parser
        # escapes values for its bundled graphs, so disable that serialization
        # only for this graph pair.
        original_escape = token_parser.escape_value
        try:
            token_parser.escape_value = lambda value: value
            reordered = token_parser.TokenParser("zh", "tn").reorder(tagged)
        finally:
            token_parser.escape_value = original_escape
        return self._verbalizer(reordered)
