"""Strict whitespace-token vocabulary used by the ZH/EN Matcha frontend."""

from pathlib import Path


class TokenVocabulary:
    def __init__(self, tokens):
        if not tokens or tokens[0] != "<pad>":
            raise ValueError("vocabulary must start with <pad>")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary contains duplicate tokens")
        self.tokens = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def load(cls, path):
        tokens = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            token, separator, raw_id = line.rpartition("\t")
            if not separator or int(raw_id) != len(tokens):
                raise ValueError(f"invalid vocabulary line {line_number}: {line!r}")
            tokens.append(token)
        return cls(tokens)

    def encode(self, text):
        tokens = text.split()
        unknown = sorted(set(tokens) - self.token_to_id.keys())
        if unknown:
            raise ValueError(f"unknown tokens: {unknown[:10]}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids):
        try:
            return [self.tokens[index] for index in ids]
        except IndexError as exc:
            raise ValueError("token id outside vocabulary") from exc
