"""PhoneTone ONNX adapter contract without a real Jetson runtime."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def test_adapter_passes_all_phonetone_inputs_and_trims_wav(monkeypatch, tmp_path):
    frontend = types.ModuleType("plugins.matcha_phonetone.frontend")
    frontend.prepare_phonetone = lambda text: types.SimpleNamespace(
        phone_ids=(1, 2), tone_ids=(3, 4), language_ids=(0, 1)
    )
    monkeypatch.setitem(sys.modules, "plugins.matcha_phonetone.frontend", frontend)

    captured = {}
    class Session:
        def __init__(self, path, providers):
            captured["path"], captured["providers"] = path, providers
        def get_inputs(self):
            return [types.SimpleNamespace(name=name) for name in
                    ("x", "x_lengths", "tones", "languages", "scales")]
        def get_outputs(self):
            return [types.SimpleNamespace(name=name) for name in ("wav", "wav_lengths")]
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        def run(self, _outputs, inputs):
            captured["inputs"] = inputs
            return [np.array([[0.0, 1.0, -1.0]], dtype=np.float32), np.array([2])]

    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CUDAExecutionProvider"]
    ort.get_device = lambda: "GPU"
    ort.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    sys.modules.pop("plugins.matcha_phonetone.adapter", None)
    adapter = importlib.import_module("plugins.matcha_phonetone.adapter")
    (tmp_path / "model.onnx").write_bytes(b"model")

    pcm = adapter.MatchaPhoneToneORTAdapter(str(tmp_path)).synthesize("hello")
    assert captured["providers"][0] == "CUDAExecutionProvider"
    assert captured["inputs"]["x"].tolist() == [[0, 1, 0, 2, 0]]
    assert captured["inputs"]["tones"].tolist() == [[0, 3, 0, 4, 0]]
    assert len(pcm) == 4  # wav_lengths trims the padded third sample


def test_adapter_selects_step_and_independent_vocoder(monkeypatch, tmp_path):
    frontend = types.ModuleType("plugins.matcha_phonetone.frontend")
    frontend.prepare_phonetone = lambda text: types.SimpleNamespace(
        phone_ids=(1, 2), tone_ids=(3, 4), language_ids=(0, 1)
    )
    monkeypatch.setitem(sys.modules, "plugins.matcha_phonetone.frontend", frontend)

    class Session:
        def __init__(self, path, providers):
            self.path = str(path)
        def get_inputs(self):
            names = ("x", "x_lengths", "tones", "languages", "scales")
            if "bigvgan" in self.path:
                names = ("mels",)
            return [types.SimpleNamespace(name=name, shape=[1, 80, 64] if name == "mels" else None)
                    for name in names]
        def get_outputs(self):
            names = ("mel", "mel_lengths") if "bigvgan" not in self.path else ("wav",)
            return [types.SimpleNamespace(name=name) for name in names]
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        def run(self, _outputs, _inputs):
            if "bigvgan" in self.path:
                return [np.zeros((1, 768), dtype=np.float32)]
            return [np.zeros((1, 80, 4), dtype=np.float32), np.array([3])]

    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CUDAExecutionProvider"]
    ort.get_device = lambda: "GPU"
    ort.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    sys.modules.pop("plugins.matcha_phonetone.adapter", None)
    adapter = importlib.import_module("plugins.matcha_phonetone.adapter")
    (tmp_path / "model-steps-4.onnx").write_bytes(b"model")
    (tmp_path / "bigvgan.onnx").write_bytes(b"vocoder")
    monkeypatch.setenv("MATCHA_STEPS", "4")

    instance = adapter.MatchaPhoneToneORTAdapter(str(tmp_path), vocoder="bigvgan")
    pcm = instance.synthesize("hello")
    assert len(pcm) == 3 * adapter.HOP_LENGTH * 2
    assert instance.last_timings["vocoder"] == "bigvgan"
    assert instance.last_timings["vocoder_onnx_seconds"] >= 0
    assert instance.last_timings["istft_seconds"] == 0
