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
