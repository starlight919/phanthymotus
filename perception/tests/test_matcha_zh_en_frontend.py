import pytest

pytest.importorskip("unidecode")

from plugins.matcha_phonetone.heteronym import custom_dict
from plugins.matcha_phonetone.zh_en_frontend import _apply_overlay, _transliterate_non_cjk


def test_overlay_uses_shortest_confirmed_phrases():
    overlay = {"小心翼翼地": ["xiao3", "xin1", "yi4", "yi4", "de5"]}
    source = ["xiao3", "xin1", "yi4", "yi4", "di4", "fang4", "xia4"]
    assert _apply_overlay("小心翼翼地放下", source, overlay)[4] == "de5"


def test_invalid_overlay_fails_closed():
    with pytest.raises(ValueError, match="invalid pinyin overlay"):
        _apply_overlay("土地", ["tu3", "di4"], {"土地": ["tu3"]})


def test_shipped_overlay_is_minimal_and_valid():
    assert "地" not in custom_dict and "小心翼翼地放下" not in custom_dict


def test_transliteration_never_destroys_chinese():
    assert _transliterate_non_cjk("München妈妈") == "Munchen妈妈"


def test_web_semantic_pinyin_repairs_are_frozen():
    expected = {
        "呼吸着": ["hu1", "xi1", "zhe5"], "急得": ["ji2", "de5"],
        "变得": ["bian4", "de5"], "梳理得": ["shu1", "li3", "de5"],
        "佩服得很": ["pei4", "fu2", "de5", "hen3"],
        "放得开": ["fang4", "de5", "kai1"],
        "羞愧得要死": ["xiu1", "kui4", "de5", "yao4", "si3"],
        "抓空去": ["zhua1", "kong4", "qu4"], "我倒是": ["wo3", "dao4", "shi4"],
    }
    assert {phrase: [item[0] for item in custom_dict[phrase]] for phrase in expected} == expected


def test_normalize_once_then_encode_chunks(monkeypatch, tmp_path):
    from plugins.matcha_phonetone import zh_en_frontend as frontend

    calls = []
    def normalize(text):
        calls.append(text)
        return "hello,world!"

    monkeypatch.setattr(frontend, "_resources", lambda: ({}, normalize))
    monkeypatch.setattr(frontend, "_english", lambda text: [text])
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<pad>\t0\nhello\t1\n,\t2\nworld\t3\n!\t4\n")
    whole = frontend.prepare_text("raw input", vocab)
    assert calls == ["raw input"]
    assert whole.normalized_text == "hello,world!"
    assert whole.tokens == ["hello", ",", "world", "!"]
    assert whole.token_ids == [1, 2, 3, 4]
    assert whole.fallbacks == ()

    def unexpected(*args):
        pytest.fail("normalized-input G2P must not normalize or transliterate")

    monkeypatch.setattr(frontend, "normalize_text", unexpected)
    monkeypatch.setattr(frontend, "_transliterate_non_cjk", unexpected)
    chunks = [frontend.prepare_normalized_text(text, vocab)
              for text in ("hello,", "world!")]
    assert [i for chunk in chunks for i in chunk.token_ids] == whole.token_ids
    assert [chunk.normalized_text for chunk in chunks] == ["hello,", "world!"]
    assert calls == ["raw input"]


@pytest.mark.parametrize("output", ["ModelHub", "MODELHUB", "premodelhubpost", "model hub"])
def test_fst_output_has_no_modelhub_postreplacement(monkeypatch, output):
    import sys
    import types
    from plugins.matcha_phonetone.zh_en_frontend import _FstNormalizer

    escape = lambda value: value
    parser = types.SimpleNamespace(
        escape_value=escape,
        TokenParser=lambda *args: types.SimpleNamespace(reorder=lambda value: value),
    )
    monkeypatch.setitem(sys.modules, "wetext", types.SimpleNamespace(token_parser=parser))
    normalizer = object.__new__(_FstNormalizer)
    normalizer.tagger = lambda text: text
    normalizer.verbalizer = lambda text: output
    assert normalizer("input") == output
    assert parser.escape_value is escape
