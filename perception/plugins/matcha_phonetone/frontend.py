"""Standalone VITS-compatible ZH/EN PhoneTone frontend."""

from __future__ import annotations

import json
import os
import re
import runpy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from pypinyin import Style, lazy_pinyin, load_phrases_dict

from .heteronym import custom_dict
from .symbols import (
    language_id_map,
    language_tone_start_map,
    num_languages,
    num_tones,
    punctuation,
    symbols,
)
from .zh_en_frontend import _FstNormalizer, _apply_overlay, _transliterate_non_cjk

_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(symbols)}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:['-][A-Za-z]+)*|[!?…,.'-]")
_ARPA_RE = re.compile(r"^([A-Z]+)([012])?$")
_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")

@dataclass(frozen=True)
class PhoneToneResult:
    normalized_text: str
    phones: tuple[str, ...]
    phone_ids: tuple[int, ...]
    tone_ids: tuple[int, ...]
    language_ids: tuple[int, ...]

    def __post_init__(self):
        if not (len(self.phone_ids) == len(self.tone_ids) == len(self.language_ids)):
            raise ValueError("phone/tone/language sequences must be equal length")


def _release_root() -> Path:
    return Path(os.environ.get("MATCHA_FRONTEND_RELEASE", Path(__file__).parents[2] / "frontend_release"))


@lru_cache(maxsize=1)
def _assets():
    root = _release_root()
    di_path = root / "frontend_data/phrase_pinyin_data/di.py"
    if not di_path.is_file():
        raise FileNotFoundError(f"missing frozen pypinyin phrase dictionary: {di_path}")
    runpy.run_path(str(di_path))["load"]()
    # Reviewed entries are frozen release data and win legacy duplicate phrases.
    reviewed = _reviewed_dict(root / "reviewed_custom_dict.json")
    merged = {**custom_dict, **reviewed}
    load_phrases_dict(merged, style="tone2")
    mapping = {}
    for line in (root / "opencpop-strict.txt").read_text(encoding="utf-8").splitlines():
        pinyin, phones = line.split("\t", 1)
        mapping[pinyin] = phones.split()
    custom_en = json.loads((root / "custom_en_pronunciations.json").read_text(encoding="utf-8"))
    cmu = {}
    with (root / "cmudict.rep").open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("##") or "  " not in line:
                continue
            word, pronunciation = line.rstrip().split("  ", 1)
            cmu.setdefault(re.sub(r"\(\d+\)$", "", word), pronunciation.replace(" - ", " ").split())
    return mapping, custom_en, cmu, _FstNormalizer(root / "tn_cache")


def _zh_phones(text: str, lexical_pinyin: Sequence[str] | None = None) -> tuple[list[str], list[int]]:
    mapping, _, _, _ = _assets()
    phones, tones = [], []
    if lexical_pinyin is None:
        lexical = runtime_lexical_pinyin(text)
    else:
        lexical = list(lexical_pinyin)
        if len(lexical) != len(text):
            raise ValueError(f"Gold pinyin length mismatch: text={len(text)} gold={len(lexical)}")
    for tone3 in lexical:
        if not tone3 or tone3[-1] not in "12345":
            raise ValueError(f"invalid Chinese TONE3: {text!r} -> {tone3!r}")
        tone = int(tone3[-1])
        syllable = tone3[:-1]
        initial = next((value for value in _INITIALS if syllable.startswith(value)), "")
        raw_final = syllable[len(initial):]
        syllable = initial + {"uei": "ui", "iou": "iu", "uen": "un"}.get(raw_final, raw_final)
        if not initial:
            syllable = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}.get(syllable, syllable)
            if syllable and syllable[0] in "viu" and syllable not in mapping:
                syllable = {"v": "yu", "i": "y", "u": "w"}[syllable[0]] + syllable[1:]
        if syllable not in mapping:
            raise ValueError(f"Chinese pinyin OOV: {syllable!r} ({text!r})")
        unit = mapping[syllable]
        phones.extend(unit)
        tones.extend([tone] * len(unit))
    return phones, tones


def runtime_lexical_pinyin(text: str) -> list[str]:
    """Frozen pypinyin/di/custom baseline shared by training and inference."""
    _assets()
    lexical = lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True,
                          neutral_tone_with_five=True, errors=lambda value: list(value))
    lexical = _apply_overlay(text, lexical, _lexical_overlay())
    for index, char in enumerate(text):
        if char == "嗯":
            lexical[index] = "en1"
        elif char == "呣":
            lexical[index] = "mu3"
    return lexical


@lru_cache(maxsize=1)
def _lexical_overlay() -> dict[str, list[str]]:
    from pypinyin.contrib.tone_convert import to_tone3

    reviewed = _reviewed_dict(_release_root() / "reviewed_custom_dict.json")
    return {
        phrase: [to_tone3(item[0], neutral_tone_with_five=True) for item in values]
        for phrase, values in {**custom_dict, **reviewed}.items()
    }


def _reviewed_dict(path: Path) -> dict[str, list[list[str]]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for phrase, values in raw.items():
        if not isinstance(phrase, str) or len(phrase) < 2 or len(values) != len(phrase):
            raise ValueError(f"invalid reviewed polyphone phrase: {phrase!r}")
        if not all(isinstance(value, str) and re.fullmatch(r"[a-züv]+[1-5]", value) for value in values):
            raise ValueError(f"invalid reviewed polyphone pinyin: {phrase!r}")
        result[phrase] = [[value] for value in values]
    return result


def _arpa_to_phone(value: str) -> tuple[str, int]:
    match = _ARPA_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid ARPAbet phone: {value!r}")
    phone, stress = match.groups()
    symbol = "V" if phone == "V" else phone.lower()
    if symbol not in _SYMBOL_TO_ID:
        raise ValueError(f"English phone OOV: {symbol!r}")
    return symbol, int(stress) + 1 if stress is not None else 3


@lru_cache(maxsize=1)
def _g2p():
    import nltk
    from g2p_en import G2p

    local_nltk = str(_release_root() / "nltk_data")
    if local_nltk not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk)
    return G2p()


@lru_cache(maxsize=4096)
def _en_phones(word: str) -> tuple[list[str], list[int]]:
    _, custom_en, cmu, _ = _assets()
    pronunciation = custom_en.get(word.upper()) or cmu.get(word.upper())
    if pronunciation is None and word.isupper() and len(word) > 1:
        # Technical acronyms are spoken letter by letter; this also prevents
        # g2p_en from returning an empty sequence for values such as NWC.
        spelled = [cmu.get(char) for char in word if char.isalpha()]
        if spelled and all(spelled):
            pronunciation = [phone for item in spelled for phone in item]
    if pronunciation is None:
        pronunciation = [value for value in _g2p()(word) if value != " "]
    converted = [_arpa_to_phone(value) for value in pronunciation if _ARPA_RE.fullmatch(value)]
    if not converted:
        raise ValueError(f"English G2P produced no phones: {word!r}")
    return [x[0] for x in converted], [x[1] for x in converted]


def normalize_text(text: str) -> str:
    """Return the exact R15-normalized string used for token indexing."""
    _, _, _, normalizer = _assets()
    return _transliterate_non_cjk(normalizer(text)).replace("嗯", "恩").replace("呣", "母")


def prepare_phonetone(
    text: str,
    gold_lexical_pinyin: Sequence[str] | None = None,
    *,
    input_is_normalized: bool = False,
) -> PhoneToneResult:
    """Encode text; optional Gold is character-aligned to normalized text."""
    normalized = text if input_is_normalized else normalize_text(text)
    if gold_lexical_pinyin is not None and len(gold_lexical_pinyin) != len(normalized):
        raise ValueError(
            f"Gold pinyin must align with normalized text: text={len(normalized)} gold={len(gold_lexical_pinyin)}"
        )
    phones, raw_tones, langs = ["_"], [0], ["ZH"]
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group()
        if _CJK_RE.fullmatch(token):
            gold = gold_lexical_pinyin[match.start():match.end()] if gold_lexical_pinyin is not None else None
            token_phones, token_tones = _zh_phones(token, gold)
            language = "ZH"
        elif token in punctuation:
            token_phones, token_tones, language = [token], [0], "ZH"
        else:
            token_phones, token_tones = _en_phones(token)
            language = "EN"
        phones.extend(token_phones)
        raw_tones.extend(token_tones)
        langs.extend([language] * len(token_phones))
    phones.append("_")
    raw_tones.append(0)
    langs.append("ZH")
    phone_ids = tuple(_SYMBOL_TO_ID[phone] for phone in phones)
    tone_ids = tuple(tone + language_tone_start_map[lang] for tone, lang in zip(raw_tones, langs))
    language_ids = tuple(language_id_map[lang] for lang in langs)
    if any(tone >= num_tones for tone in tone_ids) or any(lang >= num_languages for lang in language_ids):
        raise ValueError("PhoneTone ID exceeds frozen VITS-compatible range")
    return PhoneToneResult(normalized, tuple(phones), phone_ids, tone_ids, language_ids)
