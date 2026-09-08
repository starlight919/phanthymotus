import importlib.util

import pytest

pytest.importorskip("pypinyin")
pytest.importorskip("unidecode")


@pytest.fixture(autouse=True)
def release_available(request):
    if request.node.name in {
        "test_reviewed_dictionary_fails_closed",
        "test_normalized_phonetone_skips_normalization",
    }:
        return
    missing = [name for name in ("wetext", "kaldifst", "g2p_en", "inflect", "nltk")
               if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"frontend dependencies missing: {', '.join(missing)}")
    if not (phonetone_frontend._release_root() / "tn_cache/tn_manifest.json").is_file():
        pytest.skip("no frontend release; set MATCHA_FRONTEND_RELEASE")


from plugins.matcha_phonetone import frontend as phonetone_frontend
from plugins.matcha_phonetone.frontend import _en_phones, normalize_text, prepare_phonetone
from plugins.matcha_phonetone.symbols import language_tone_start_map, num_languages, num_tones


def test_equal_length_and_strict_bilingual_offsets():
    result = prepare_phonetone("妈妈不在 CUDA world!")
    assert len(result.phone_ids) == len(result.tone_ids) == len(result.language_ids)
    assert max(result.tone_ids) < num_tones == 10
    assert max(result.language_ids) < num_languages == 2
    assert language_tone_start_map == {"ZH": 0, "EN": 6}


def test_chinese_lexical_tone_and_neutral_tone():
    mama = prepare_phonetone("妈妈")
    assert mama.tone_ids[1:-1] == (1, 1, 5, 5)
    particles = prepare_phonetone("快乐地变得很好")
    assert 5 in particles.tone_ids
    for word in ("土地", "当地", "目的地"):
        _, tones = phonetone_frontend._zh_phones(word)
        assert tones[-1] == 4
    _, tones = phonetone_frontend._zh_phones("得了满分")
    assert tones[:len(phonetone_frontend._assets()[0]["de"])] == [2] * len(phonetone_frontend._assets()[0]["de"])
    assert phonetone_frontend._zh_phones("不在")[1][:2] == [2, 2]


def test_polyphone_overlay():
    result = prepare_phonetone("银行行长和华为")
    assert result.phone_ids and all(value < 6 for value in result.tone_ids)


def test_reviewed_dictionary_fails_closed(tmp_path):
    path = tmp_path / "reviewed.json"
    path.write_text('{"重庆": ["chong2", "qing4"]}', encoding="utf-8")
    assert phonetone_frontend._reviewed_dict(path)["重庆"] == [["chong2"], ["qing4"]]
    path.write_text('{"地": ["di4"]}', encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="invalid reviewed polyphone phrase"):
        phonetone_frontend._reviewed_dict(path)


def test_frozen_di_dictionary_is_loaded_before_custom_overlay():
    phonetone_frontend._assets()
    assert (phonetone_frontend._release_root() / "frontend_data/phrase_pinyin_data/di.py").is_file()
    # custom_dict remains the final authority even when di.py contains the phrase.
    assert phonetone_frontend._zh_phones("快乐地")[1][-1] == 5


def test_polyphone_overlay_is_the_only_context_override():
    _, tones = phonetone_frontend._zh_phones("得重新")
    assert tones[:2] == [3, 3]


def test_sample_gold_is_public_and_authoritative():
    plain = prepare_phonetone("不在")
    gold = prepare_phonetone("不在", ["bu4", "zai4"])
    assert plain.tone_ids[1:3] == (2, 2)
    assert gold.tone_ids[1:3] == (4, 4)


def test_sample_gold_must_align_with_normalized_text():
    import pytest

    with pytest.raises(ValueError, match="align with normalized text"):
        prepare_phonetone("妈妈", ["ma1"])


def test_public_normalization_is_the_gold_index_contract():
    normalized = normalize_text("温度是 20℃")
    assert prepare_phonetone(normalized, input_is_normalized=True).normalized_text == normalized


def test_yw_whole_syllables_are_not_corrupted():
    for text in ("为", "华为", "银行", "一个", "五", "云"):
        assert prepare_phonetone(text).phone_ids


def test_custom_then_vits_cmudict_then_g2p_fallback(monkeypatch):
    assert _en_phones("CUDA")[0] == ["k", "uw", "d", "ah"]
    assert _en_phones("IELTS")[0] == ["ay", "eh", "l", "t", "s"]
    assert _en_phones("world")[0]
    assert _en_phones("NWC")[0]
    monkeypatch.setattr(phonetone_frontend, "_g2p", lambda: lambda _: ["K", "OW1", "D", "EH2", "K", "S"])
    assert _en_phones("not-in-vits-cmudict")[0] == ["k", "ow", "d", "eh", "k", "s"]


def test_mixed_language_ids_and_english_stress():
    result = prepare_phonetone("你好 Apple Pay")
    assert 0 in result.language_ids and 1 in result.language_ids
    english_tones = [tone for tone, lang in zip(result.tone_ids, result.language_ids) if lang == 1]
    assert english_tones and set(english_tones) <= {7, 8, 9}


def test_interjections_do_not_become_pinyin_oov():
    result = prepare_phonetone("嗯，呣，我知道了")
    assert result.normalized_text.startswith("恩,母,")
    assert result.phone_ids


def test_normalized_phonetone_skips_normalization(monkeypatch):
    calls = []
    def normalize(text):
        calls.append(text)
        return "hello!"

    monkeypatch.setattr(phonetone_frontend, "normalize_text", normalize)
    monkeypatch.setattr(phonetone_frontend, "_en_phones", lambda word: (["hh", "ah"], [3, 1]))
    result = prepare_phonetone("raw input")
    assert calls == ["raw input"]
    assert prepare_phonetone(result.normalized_text, input_is_normalized=True) == result
    assert calls == ["raw input"]
    assert result.phones == ("_", "hh", "ah", "!", "_")
    assert result.tone_ids == (0, 9, 7, 0, 0)
    assert result.language_ids == (0, 1, 1, 0, 0)
