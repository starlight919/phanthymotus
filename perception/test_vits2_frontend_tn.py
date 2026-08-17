#!/usr/bin/env python3
"""Regression tests for the external WeText FST used by the VITS2 frontend."""

import os
import unittest
from pathlib import Path


TN_CACHE = Path(os.environ.get("TN_CACHE_DIR", ""))
FRONTEND_DATA = Path(os.environ.get("VITS2_FRONTEND_DATA_DIR", ""))
HAS_TN = all(
    (TN_CACHE / name).is_file()
    for name in ("zh_tn_tagger.fst", "zh_tn_verbalizer.fst")
)
HAS_PHRASE_DICT = (FRONTEND_DATA / "phrase_pinyin_data" / "di.py").is_file()


@unittest.skipUnless(HAS_TN, "set TN_CACHE_DIR to a verified TN release")
class TestVits2FrontendTN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix
        cls.normalize = staticmethod(normalize_text_mix)

    def test_p0_normalization(self):
        cases = {
            "20dB": "二十分贝",
            "09:05～11:30": "九点零五分到十一点三十分",
            "订单编号8492610": "订单编号八四九二六一零",
            "TTS-test-2026": "TTS test 二零二六",
        }
        for written, expected in cases.items():
            with self.subTest(written=written):
                self.assertEqual(self.normalize(written), expected)

    def test_conflicts_do_not_regress(self):
        cases = {
            "共有8492610人": "共有八百四十九万两千六百一十人",
            "2026-08-07": "二零二六年八月七日",
            "3-2": "三减二",
            "y=2x-1": "y等于二x减一",
        }
        for written, expected in cases.items():
            with self.subTest(written=written):
                self.assertEqual(self.normalize(written), expected)

    @unittest.skipUnless(
        HAS_PHRASE_DICT, "set VITS2_FRONTEND_DATA_DIR to release frontend data"
    )
    def test_kuaile_de_phrase_boundary(self):
        import jieba.posseg as psg

        from plugins.vits2_tts_trt.frontend.cleaner import clean_text_mix

        self.assertEqual(psg.lcut("快乐地生活")[0].word, "快乐地")
        normalized, phones, tones, _, word2ph = clean_text_mix("快乐地生活。")
        self.assertEqual(normalized, "快乐地生活.")
        token_index = 3  # leading blank + character index 2
        start = sum(word2ph[:token_index])
        end = start + word2ph[token_index]
        self.assertEqual(phones[start:end], ["d", "e"])
        self.assertEqual(tones[start:end], [5, 5])

if __name__ == "__main__":
    unittest.main()
