from pathlib import Path

import pytest

from plugins.matcha_phonetone.token_vocabulary import TokenVocabulary


def test_vocabulary_round_trip(tmp_path: Path):
    path = tmp_path / "vocab.txt"
    path.write_text("<pad>\t0\nma1\t1\nma5\t2\nm\t3\n", encoding="utf-8")
    vocabulary = TokenVocabulary.load(path)
    ids = vocabulary.encode("ma1 ma5 m")
    assert ids == [1, 2, 3]
    assert vocabulary.decode(ids) == ["ma1", "ma5", "m"]


def test_unknown_token_fails_closed():
    vocabulary = TokenVocabulary(["<pad>", "ma1"])
    with pytest.raises(ValueError, match="unknown tokens"):
        vocabulary.encode("ma1 di5")
