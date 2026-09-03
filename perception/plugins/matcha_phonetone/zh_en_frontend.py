"""Frozen ZH/EN inference frontend for the pre-tokenized Matcha models."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import runpy
import threading
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from unidecode import unidecode

from .token_vocabulary import TokenVocabulary
from .wetext_compat import ensure_wetext_compat

ensure_wetext_compat()

PUNCTUATION = set(";:,.!?-—…\"'()[] ")
PUNCTUATION_MAP = {"，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
                   "、": ",", "《": '"', "》": '"', "【": "[", "】": "]", "“": '"', "”": '"',
                   "‘": "'", "’": "'"}
SPAN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*(?:\s+[A-Za-z]+(?:['-][A-Za-z]+)*)*|[\u4e00-\u9fff]+|.")
TONE3_RE = re.compile(r"[a-züv]+[1-5]")


@dataclass(frozen=True)
class FrontendResult:
    normalized_text: str
    tokens: list[str]
    token_ids: list[int]
    fallbacks: tuple[tuple[str, str], ...] = ()


def _fallback_token(token: str, vocabulary: TokenVocabulary) -> str:
    training_fallbacks = {"da5": "da1"}
    if token in training_fallbacks and training_fallbacks[token] in vocabulary.token_to_id:
        return training_fallbacks[token]
    match = re.fullmatch(r"(.+)([1-5])", token)
    if not match:
        return token
    base, raw_tone = match.groups()
    candidates = [value for value in vocabulary.token_to_id if re.fullmatch(re.escape(base) + r"[1-5]", value)]
    if not candidates:
        return token
    tone = int(raw_tone)
    return min(candidates, key=lambda value: (abs(int(value[-1]) - tone), -int(value[-1])))


class _FstNormalizer:
    _lock = threading.RLock()

    def __init__(self, root: Path):
        import hashlib
        import kaldifst

        manifest = json.loads((root / "tn_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported TN manifest")
        paths = []
        for name in ("zh_tn_tagger.fst", "zh_tn_verbalizer.fst"):
            path, meta = root / name, manifest["fst"][name]
            if path.stat().st_size != meta["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"TN checksum mismatch: {name}")
            paths.append(str(path))
        self.tagger, self.verbalizer = map(kaldifst.TextNormalizer, paths)

    def __call__(self, text: str) -> str:
        from wetext import token_parser

        tagged = self.tagger(text)
        with self._lock:
            old = token_parser.escape_value
            try:
                token_parser.escape_value = lambda value: value
                reordered = token_parser.TokenParser("zh", "tn").reorder(tagged)
            finally:
                token_parser.escape_value = old
        normalized = self.verbalizer(reordered)
        # Product name is spoken as two English words, not one fused token.
        return re.sub(r"(?i)modelhub", "model hub", normalized)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("matcha_frontend_custom_dict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dictionary: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _resources():
    from pypinyin import load_phrases_dict
    from pypinyin.contrib.tone_convert import to_tone3

    release = Path(os.environ["MATCHA_FRONTEND_RELEASE"])
    data = release / "frontend_data"
    di = data / "phrase_pinyin_data" / "di.py"
    runpy.run_path(str(di))["load"]()
    custom_path = Path(__file__).with_name("heteronym.py")
    custom = _load_module(custom_path)
    load_phrases_dict(custom.custom_dict, style="tone2")
    overlay = {
        phrase: [to_tone3(item[0], neutral_tone_with_five=True) for item in values]
        for phrase, values in custom.custom_dict.items()
    }
    normalizer = _FstNormalizer(release / "tn_cache")
    return overlay, normalizer


def _apply_overlay(text: str, tokens: list[str], overlay: dict) -> list[str]:
    if len(tokens) != len(text):
        raise ValueError("Chinese pinyin alignment failed")
    phrases = sorted(overlay, key=len, reverse=True)
    for phrase in phrases:
        expected = overlay[phrase]
        if len(expected) != len(phrase) or not all(TONE3_RE.fullmatch(value) for value in expected):
            raise ValueError(f"invalid pinyin overlay: {phrase}")
    out, index = list(tokens), 0
    while index < len(text):
        phrase = next((value for value in phrases if text.startswith(value, index)), None)
        if phrase:
            out[index:index + len(phrase)] = overlay[phrase]
            index += len(phrase)
        else:
            index += 1
    return out


def _chinese(text: str, overlay: dict) -> list[str]:
    from pypinyin import Style, lazy_pinyin

    tokens = lazy_pinyin(text, style=Style.TONE3, tone_sandhi=False, neutral_tone_with_five=True)
    return _apply_overlay(text, tokens, overlay)


def _english(text: str) -> list[str]:
    return [char for char in _english_backend().phonemize([text], strip=True, njobs=1)[0] if not char.isspace()]


@lru_cache(maxsize=1)
def _english_backend():
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(language="en-us", preserve_punctuation=False, with_stress=True)


def _transliterate_non_cjk(text: str) -> str:
    return "".join(PUNCTUATION_MAP.get(char, char if "\u4e00" <= char <= "\u9fff" or char.isascii() else unidecode(char)) for char in text)


def prepare_text(text: str, vocab_path: str | Path | None = None) -> FrontendResult:
    overlay, normalizer = _resources()
    normalized = _transliterate_non_cjk(normalizer(text))
    tokens = []
    for match in SPAN_RE.finditer(normalized):
        value = match.group()
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.extend(_chinese(value, overlay))
        elif re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*(?:\s+[A-Za-z]+(?:['-][A-Za-z]+)*)*", value):
            tokens.extend(_english(value))
        elif value in PUNCTUATION and not value.isspace():
            tokens.append(value)
        elif not value.isspace():
            raise ValueError(f"unsupported normalized text: {value!r}")
    vocabulary = TokenVocabulary.load(vocab_path or Path(os.environ["MATCHA_FRONTEND_RELEASE"]) / "vocab.txt")
    resolved = [_fallback_token(token, vocabulary) if token not in vocabulary.token_to_id else token for token in tokens]
    fallbacks = tuple((token, replacement) for token, replacement in zip(tokens, resolved) if token != replacement)
    if fallbacks:
        warnings.warn(f"pinyin token fallback: {fallbacks}", RuntimeWarning)
    return FrontendResult(normalized, tokens, vocabulary.encode(" ".join(resolved)), fallbacks)
