"""Stable lexical handling for technology names after Chinese TN."""

import re


_PRODUCT_FORMS = (
    (re.compile(r"\bchatgpt\b", re.IGNORECASE), "chat GPT"),
    (re.compile(r"\bopenai\b", re.IGNORECASE), "open AI"),
    (re.compile(r"\bchattts\b", re.IGNORECASE), "chat T T S"),
    (re.compile(r"\bstyletts2\b", re.IGNORECASE), "style T T S二"),
    (re.compile(r"\brise\s+vgpu\b", re.IGNORECASE), "rise V G P U"),
    (re.compile(r"\bphanthymotus\b", re.IGNORECASE), "Phanthy Motus"),
    (re.compile(r"\bhttps\b", re.IGNORECASE), "H T T P S"),
    (re.compile(r"\bhttp\b", re.IGNORECASE), "H T T P"),
)


def normalize_technical_lexemes(text: str) -> str:
    """Make known product names deterministic without touching TN semantics."""
    for pattern, replacement in _PRODUCT_FORMS:
        text = pattern.sub(replacement, text)
    return text
