from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import plugins.matcha_phonetone.adapter as adapter_module
from plugins.matcha_phonetone.adapter import MatchaTensorRTAdapter
from plugins.tts import CHUNK_BYTES


class _Session:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, inputs):
        self.calls.append(inputs)
        return self.result(inputs) if callable(self.result) else self.result


def _manifest(vocoder="hifigan", *, encoder_inputs=None):
    vocoder_contract = {"name": vocoder}
    if vocoder == "vocos":
        vocoder_contract["istft"] = {
            "n_fft": 1024, "hop_length": 256, "window": "hann_periodic", "padding": "same"
        }
    vocoder_outputs = ["mag", "x", "y"] if vocoder == "vocos" else ["audio"]
    return {
        "contract": {"solver_steps": 3, "vocoder": vocoder_contract},
        "engines": {
            "encoder": {
                "bindings": {
                    "inputs": encoder_inputs or ["x", "x_lengths", "tones", "languages"],
                    "outputs": ["mu_x", "logw", "x_mask"],
                },
            },
            "solver_steps_3": {
                "bindings": {
                    "inputs": ["noise", "mask", "mu"], "outputs": ["mel"],
                },
            },
            vocoder: {"bindings": {"inputs": ["mel"], "outputs": vocoder_outputs}},
        },
    }


def _runtime(vocoder="hifigan"):
    encoder = _Session({
        "mu_x": np.arange(240, dtype=np.float32).reshape(1, 80, 3),
        "logw": np.zeros((1, 1, 3), dtype=np.float32),
        "x_mask": np.ones((1, 1, 3), dtype=np.float32),
    })
    solver = _Session({"mel": np.ones((1, 80, 4), dtype=np.float32)})
    if vocoder == "vocos":
        spectral = np.ones((1, 513, 4), dtype=np.float32)
        vocoder_session = _Session({
            "mag": spectral, "x": np.ones_like(spectral), "y": np.zeros_like(spectral)
        })
    else:
        vocoder_session = _Session({
            "audio": np.linspace(-1.5, 1.5, 4 * 256, dtype=np.float32)[None, :]
        })
    return SimpleNamespace(
        manifest=_manifest(vocoder),
        encoder=encoder,
        solver=solver,
        vocoder=vocoder_session,
        closed=False,
        close=lambda: None,
    )


@pytest.fixture
def frontend(monkeypatch):
    result = SimpleNamespace(
        phone_ids=(3, 7, 11), tone_ids=(1, 2, 3), language_ids=(4, 5, 6)
    )
    monkeypatch.setattr(adapter_module.frontend, "normalize_text", lambda text: text)
    monkeypatch.setattr(adapter_module.frontend, "prepare_phonetone", lambda text, **_kwargs: result)
    return result


def test_synthesis_uses_named_bindings_and_matcha_blanks(frontend):
    runtime = _runtime()
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    pcm = adapter.synthesize("ignored")

    assert len(pcm) == 3 * 256 * 2
    assert len(runtime.encoder.calls) == 1
    encoded = runtime.encoder.calls[0]
    np.testing.assert_array_equal(encoded["x"], [[0, 3, 0, 7, 0, 11, 0]])
    np.testing.assert_array_equal(encoded["x_lengths"], [7])
    np.testing.assert_array_equal(encoded["tones"], [[0, 1, 0, 2, 0, 3, 0]])
    np.testing.assert_array_equal(encoded["languages"], [[0, 4, 0, 5, 0, 6, 0]])
    assert runtime.solver.calls[0]["noise"].shape == (1, 80, 4)
    assert np.all(runtime.solver.calls[0]["noise"] == 0)
    assert runtime.vocoder.calls[0]["mel"].shape == (1, 80, 3)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples[0] == -32767
    assert samples.max() <= 32767 and samples.min() >= -32767


def test_static_engines_receive_padded_inputs_with_real_text_length(frontend):
    runtime = _runtime()
    for engine, inputs in (("encoder", ("x", "tones", "languages")),
                           ("solver_steps_3", ("noise", "mask", "mu")),
                           ("hifigan", ("mel",))):
        runtime.manifest["engines"][engine]["bindings"]["inputs"] = {
            name: {"dtype": "float32", "shape": [1, 64] if name in ("x", "tones", "languages") else [1, 80, 64]}
            for name in inputs
        }
    runtime.manifest["engines"]["encoder"]["bindings"]["inputs"]["x_lengths"] = {
        "dtype": "int64", "shape": [1]
    }
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    adapter.synthesize("ignored")

    assert runtime.encoder.calls[0]["x"].shape == (1, 64)
    np.testing.assert_array_equal(runtime.encoder.calls[0]["x_lengths"], [7])
    assert runtime.solver.calls[0]["mu"].shape == (1, 80, 64)
    assert runtime.vocoder.calls[0]["mel"].shape == (1, 80, 64)


def test_stream_chunks_pcm_at_the_shared_transport_boundary(frontend):
    runtime = _runtime()
    runtime.vocoder.result = {"audio": np.zeros((1, 2 * 3200), dtype=np.float32)}
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    chunks = list(adapter.synthesize_stream("ignored"))

    assert [len(chunk) for chunk in chunks] == [3 * 256 * 2]


def test_synthesis_accepts_exported_solver_and_hifigan_aliases(frontend):
    runtime = _runtime()
    runtime.manifest["engines"]["solver_steps_3"]["bindings"]["outputs"] = ["mel_normalized"]
    runtime.manifest["engines"]["hifigan"]["bindings"] = {
        "inputs": ["mels"], "outputs": ["wav"],
    }
    runtime.solver.result = {"mel_normalized": np.ones((1, 80, 4), dtype=np.float32)}
    runtime.vocoder.result = {"wav": np.zeros((1, 4 * 256), dtype=np.float32)}
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    pcm = adapter.synthesize("ignored")

    assert len(pcm) == 3 * 256 * 2
    assert runtime.vocoder.calls[0]["mels"].shape == (1, 80, 3)


def test_bigvgan_uses_direct_waveform_contract(frontend):
    runtime = _runtime("bigvgan")
    runtime.vocoder.result = {"audio": np.zeros((1, 4 * 256), dtype=np.float32)}
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    pcm = adapter.synthesize("ignored")

    assert len(pcm) == 3 * 256 * 2
    assert runtime.vocoder.calls[0]["mel"].shape == (1, 80, 3)


def test_binding_contract_rejects_an_incomplete_encoder():
    runtime = _runtime()
    runtime.manifest = _manifest(encoder_inputs=["x", "x_lengths", "tones"])

    with pytest.raises(ValueError, match="languages"):
        MatchaTensorRTAdapter("unused", runtime=runtime)


def test_vocos_requires_its_explicit_spectral_contract():
    runtime = _runtime("vocos")
    runtime.manifest["contract"]["vocoder"].pop("istft")

    with pytest.raises(ValueError, match="spectral/ISTFT"):
        MatchaTensorRTAdapter("unused", runtime=runtime)


def test_vocos_decodes_spectral_output_with_cpu_istft(frontend):
    runtime = _runtime("vocos")
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    pcm = adapter.synthesize("ignored")

    assert runtime.vocoder.calls[0]["mel"].shape == (1, 80, 3)
    assert len(pcm) == 3 * 256 * 2
    assert np.isfinite(np.frombuffer(pcm, dtype="<i2")).all()


def test_vocos_rejects_non_same_padding():
    runtime = _runtime("vocos")
    runtime.manifest["contract"]["vocoder"]["istft"]["padding"] = "valid"

    with pytest.raises(ValueError, match="unsupported Vocos ISTFT"):
        MatchaTensorRTAdapter("unused", runtime=runtime)


def test_vocos_rejects_wrong_spectral_frequency_bins(frontend):
    runtime = _runtime("vocos")
    spectral = np.ones((1, 512, 4), dtype=np.float32)
    runtime.vocoder.result = {"mag": spectral, "x": spectral, "y": spectral}
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    with pytest.raises(ValueError, match="spectral bins"):
        adapter.synthesize("ignored")


def test_short_vocoder_output_is_rejected(frontend):
    runtime = _runtime()
    runtime.vocoder.result = {"audio": np.zeros((1, 10), dtype=np.float32)}
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    with pytest.raises(RuntimeError, match="shorter"):
        adapter.synthesize("ignored")


def test_dynamic_encoder_splits_only_after_profile_limit(monkeypatch):
    runtime = _runtime()
    encoder = runtime.manifest["engines"]["encoder"]
    encoder["bindings"]["inputs"] = {
        name: {"dtype": "int64", "shape": [1, -1] if name != "x_lengths" else [1]}
        for name in ("x", "x_lengths", "tones", "languages")
    }
    encoder["profiles"] = {
        name: {"min": [1, 1], "opt": [1, 5], "max": [1, 7]}
        for name in ("x", "tones", "languages")
    }
    encoder["profiles"]["x_lengths"] = {"min": [1], "opt": [1], "max": [1]}
    monkeypatch.setattr(adapter_module.frontend, "normalize_text", lambda text: text)
    monkeypatch.setattr(adapter_module.frontend, "prepare_phonetone", lambda text, **_kwargs:
                        SimpleNamespace(phone_ids=tuple(range(len(text))),
                                        tone_ids=(0,) * len(text), language_ids=(0,) * len(text)))
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    adapter.synthesize("abcdef")

    assert len(runtime.encoder.calls) == 2
    assert all(call["x"].shape[-1] <= 7 for call in runtime.encoder.calls)


def test_direct_vocoder_chunks_with_context_and_crops_exact_pcm(frontend):
    runtime = _runtime("bigvgan")
    runtime.manifest["contract"]["vocoder"].update(
        chunk_core_frames=64, context_frames=32
    )
    solver = runtime.manifest["engines"]["solver_steps_3"]
    solver["profiles"] = {
        name: {"min": [1, channels, 16], "opt": [1, channels, 128], "max": [1, channels, 512]}
        for name, channels in (("noise", 80), ("mask", 1), ("mu", 80))
    }
    vocoder = runtime.manifest["engines"]["bigvgan"]
    vocoder["profiles"] = {
        "mel": {"min": [1, 80, 16], "opt": [1, 80, 128], "max": [1, 80, 512]}
    }
    runtime.encoder.result["logw"][:] = np.log(100.0)
    runtime.solver.result = lambda inputs: {"mel": np.ones_like(inputs["mu"])}
    runtime.vocoder.result = lambda inputs: {
        "audio": np.zeros((1, inputs["mel"].shape[-1] * 256), dtype=np.float32)
    }
    adapter = MatchaTensorRTAdapter("unused", runtime=runtime)

    pcm = adapter.synthesize("ignored")

    assert len(runtime.vocoder.calls) == 5
    assert len(pcm) == 300 * 256 * 2
