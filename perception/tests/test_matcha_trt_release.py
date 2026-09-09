import hashlib
import json

import pytest

from plugins.matcha_phonetone.trt_release import load_runtime_release
from utils import model_downloader


def _release(tmp_path):
    files = {"encoder.plan": b"encoder", "solver.plan": b"solver", "vocos.plan": b"vocos"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    def entry(filename):
        return {
            "file": filename,
            "size": len(files[filename]),
            "sha256": hashlib.sha256(files[filename]).hexdigest(),
            "bindings": {
                "inputs": {"input": {"dtype": "float32", "shape": [1, 80, 64]}},
                "outputs": {"output": {"dtype": "float32", "shape": [1, 80, 64]}},
            },
        }

    return {
        "schema_version": 1,
        "release_status": "runtime-ready",
        "target": "jp511",
        "tensorrt_major": 8,
        "compute_capability": "8.7",
        "contract": {
            "sample_rate": 16000,
            "mel_channels": 80,
            "mel_normalization": {"kind": "matcha"},
            "frontend": {"sha256": "a" * 64},
            "solver_steps": 6,
            "vocoder": {
                "name": "vocos",
                "istft": {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "window": "hann_periodic",
                    "padding": "same",
                },
            },
        },
        "engines": {
            "encoder": entry("encoder.plan"),
            "solver_steps_6": entry("solver.plan"),
            "vocos": entry("vocos.plan"),
        },
    }


def _write(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_load_runtime_release_accepts_complete_verified_contract(tmp_path):
    manifest = _release(tmp_path)
    _write(tmp_path, manifest)
    assert load_runtime_release(tmp_path)["target"] == "jp511"


def test_load_runtime_release_accepts_plan_ready_contract(tmp_path):
    manifest = _release(tmp_path)
    manifest["release_status"] = "plan-ready"
    _write(tmp_path, manifest)

    assert load_runtime_release(tmp_path)["release_status"] == "plan-ready"


def test_matcha_downloader_has_no_plan_only_release(tmp_path, monkeypatch):
    monkeypatch.setattr(model_downloader, "require_models_subpath", lambda path: path)
    with pytest.raises(RuntimeError, match="No model bundle"):
        model_downloader.ensure_matcha_trt_model(str(tmp_path), family="jp511")


def test_matcha_build_release_requires_complete_pin(monkeypatch):
    monkeypatch.setenv("MATCHA_TRT_MODEL_URL", "https://models.example/runtime.tar.gz")
    monkeypatch.setenv("MATCHA_TRT_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MATCHA_TRT_MODEL_FAMILY", "jp511")
    monkeypatch.delenv("MATCHA_TRT_MODEL_SIZE", raising=False)

    with pytest.raises(RuntimeError, match="must be supplied together"):
        model_downloader._matcha_trt_archives()


def test_matcha_build_release_uses_complete_pin(monkeypatch):
    monkeypatch.setenv("MATCHA_TRT_MODEL_URL", "https://models.example/runtime.tar.gz")
    monkeypatch.setenv("MATCHA_TRT_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MATCHA_TRT_MODEL_SIZE", "123")
    monkeypatch.setenv("MATCHA_TRT_MODEL_FAMILY", "jp511")

    assert model_downloader._matcha_trt_archives() == {
        "jp511": {
            "url": "https://models.example/runtime.tar.gz",
            "sha256": "a" * 64,
            "size": 123,
        }
    }


@pytest.mark.parametrize("mutate", [
    lambda release: release.update(release_status="built-unvalidated"),
    lambda release: release["engines"]["vocos"].update(sha256="0" * 64),
    lambda release: release["engines"]["vocos"].update(file="../vocos.plan"),
    lambda release: release["engines"]["vocos"].pop("bindings"),
    lambda release: release["contract"].pop("frontend"),
    lambda release: release["contract"]["vocoder"].pop("istft"),
    lambda release: release["contract"]["vocoder"]["istft"].update(padding="valid"),
    lambda release: release.update(tensorrt_major=10),
])
def test_load_runtime_release_rejects_untrusted_or_incomplete_releases(tmp_path, mutate):
    release = _release(tmp_path)
    mutate(release)
    _write(tmp_path, release)
    with pytest.raises(ValueError):
        load_runtime_release(tmp_path)
