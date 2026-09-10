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


def test_schema_two_requires_ordered_profiles(tmp_path):
    manifest = _release(tmp_path)
    manifest["schema_version"] = 2
    manifest["contract"]["vocoder"].update({
        "kind": "spectral_cpu_istft", "chunk_core_frames": 128,
        "context_frames": 16, "output_crop_policy": "spectral_then_global_istft",
    })
    for engine in manifest["engines"].values():
        engine["profiles"] = {
            name: {"min": [1, 80, 16], "opt": [1, 80, 128], "max": [1, 80, 512]}
            for name, binding in engine["bindings"]["inputs"].items()
            if len(binding["shape"]) == 3
        }
        engine["profiles"].update({
            name: {"min": [1], "opt": [1], "max": [1]}
            for name, binding in engine["bindings"]["inputs"].items()
            if len(binding["shape"]) == 1
        })
    _write(tmp_path, manifest)
    assert load_runtime_release(tmp_path)["schema_version"] == 2

    next(iter(manifest["engines"].values()))["profiles"] = {}
    _write(tmp_path, manifest)
    with pytest.raises(ValueError, match="dynamic profiles"):
        load_runtime_release(tmp_path)


@pytest.fixture
def matcha_registry_env(monkeypatch):
    for key in ("URL", "SHA256", "SIZE", "FAMILY"):
        monkeypatch.delenv(f"MATCHA_TRT_MODEL_{key}", raising=False)
    monkeypatch.delenv("MATCHA_TRT_VOCODER", raising=False)


@pytest.mark.parametrize("family", ["jp511", "jp61"])
@pytest.mark.parametrize("vocoder", ["hifigan", "vocos", "bigvgan"])
def test_matcha_registry_selection(family, vocoder, monkeypatch, matcha_registry_env):
    monkeypatch.setenv("MATCHA_TRT_VOCODER", vocoder)
    entry = model_downloader._matcha_trt_archives()[family]
    assert entry["archive"] == f"matcha-trt-{family}-{vocoder}-plan-ready.tar.gz"
    assert entry["vocoder"] == vocoder
    assert entry["size"] > 0
    assert len(entry["sha256"]) == 64


def test_matcha_registry_defaults_and_invalid_selection(monkeypatch, matcha_registry_env):
    assert model_downloader._matcha_trt_archives()["jp61"]["vocoder"] == "hifigan"
    monkeypatch.setenv("MATCHA_TRT_VOCODER", "../vocos")
    with pytest.raises(ValueError, match="Unknown MATCHA_TRT_VOCODER"):
        model_downloader._matcha_trt_archives()


@pytest.mark.parametrize("mismatch", ["target", "vocoder"])
def test_matcha_registry_rejects_manifest_mismatch(tmp_path, monkeypatch, matcha_registry_env, mismatch):
    import plugins.matcha_phonetone.trt_release as release_module

    monkeypatch.setattr(model_downloader, "require_models_subpath", lambda path: path)
    monkeypatch.setattr(model_downloader, "ensure_verified_archive", lambda *args: None)
    manifest = {"target": "jp61", "contract": {"vocoder": {"name": "hifigan"}}}
    if mismatch == "target":
        manifest["target"] = "jp511"
    else:
        manifest["contract"]["vocoder"]["name"] = "vocos"
    monkeypatch.setattr(release_module, "load_runtime_release", lambda path: manifest)
    with pytest.raises(RuntimeError, match=f"archive {mismatch} mismatch"):
        model_downloader.ensure_matcha_trt_model(str(tmp_path), family="jp61")


def test_matcha_registry_cache_isolated_and_reused(tmp_path, monkeypatch, matcha_registry_env):
    import io
    import tarfile
    import plugins.matcha_phonetone.trt_release as release_module

    monkeypatch.setattr(model_downloader, "require_models_subpath", lambda path: path)
    downloads = []

    def fetch(name, url, destination, entry):
        downloads.append(url)
        payload = json.dumps({"target": "jp61", "contract": {
            "vocoder": {"name": entry["vocoder"]}
        }}).encode()
        with tarfile.open(destination, "w:gz") as archive:
            member = tarfile.TarInfo("engines/jp61/manifest.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    monkeypatch.setattr(model_downloader, "_fetch_pinned_file", fetch)
    monkeypatch.setattr(release_module, "load_runtime_release", lambda path: json.loads(
        (tmp_path / path / "manifest.json").read_text()
    ))
    paths = []
    for vocoder in ("hifigan", "vocos", "bigvgan", "hifigan"):
        monkeypatch.setenv("MATCHA_TRT_VOCODER", vocoder)
        paths.append(model_downloader.ensure_matcha_trt_model(str(tmp_path), family="jp61"))
        assert paths[-1] == str(tmp_path / "jp61" / vocoder / "engines/jp61")
    assert paths[0] == paths[-1]
    assert len(downloads) == 3
    assert downloads[0] == (
        f"{model_downloader.MATCHA_TRT_MODEL_BASE}/matcha-trt-jp61-hifigan-plan-ready.tar.gz"
    )


def test_matcha_explicit_override_preserves_path_and_vocoder(tmp_path, monkeypatch, matcha_registry_env):
    import plugins.matcha_phonetone.trt_release as release_module

    for key, value in {"URL": "https://models.example/custom.tar.gz", "SHA256": "a" * 64,
                       "SIZE": "123", "FAMILY": "jp511"}.items():
        monkeypatch.setenv(f"MATCHA_TRT_MODEL_{key}", value)
    monkeypatch.setattr(model_downloader, "require_models_subpath", lambda path: path)
    monkeypatch.setattr(model_downloader, "ensure_verified_archive", lambda *args: None)
    monkeypatch.setattr(release_module, "load_runtime_release", lambda path: {
        "target": "jp511", "contract": {"vocoder": {"name": "vocos"}}
    })
    assert model_downloader.ensure_matcha_trt_model(str(tmp_path), family="jp511") == str(
        tmp_path / "engines/jp511"
    )


def test_load_runtime_release_accepts_bigvgan_contract(tmp_path):
    manifest = _release(tmp_path)
    manifest["contract"]["vocoder"] = {"name": "bigvgan"}
    manifest["engines"]["bigvgan"] = manifest["engines"].pop("vocos")
    manifest["engines"]["bigvgan"]["file"] = "bigvgan.plan"
    (tmp_path / "bigvgan.plan").write_bytes((tmp_path / "vocos.plan").read_bytes())
    manifest["engines"]["bigvgan"]["size"] = (tmp_path / "bigvgan.plan").stat().st_size
    manifest["engines"]["bigvgan"]["sha256"] = hashlib.sha256((tmp_path / "bigvgan.plan").read_bytes()).hexdigest()
    manifest["engines"]["bigvgan"]["bindings"] = {
        "inputs": {"mels": {"dtype": "float32", "shape": [1, 80, 64]}},
        "outputs": {"wav": {"dtype": "float32", "shape": [1, 1, 16384]}},
    }
    _write(tmp_path, manifest)

    assert load_runtime_release(tmp_path)["contract"]["vocoder"]["name"] == "bigvgan"


@pytest.mark.parametrize("fields, succeeds, installs", [
    ({}, True, False),
    ({"URL": "https://example/model.tar.gz"}, False, False),
    ({"SHA256": "a" * 64, "SIZE": "123"}, False, False),
    ({"URL": "https://example/model.tar.gz", "SHA256": "a" * 64, "SIZE": "123"}, True, True),
])
def test_docker_matcha_preload_is_optional_and_requires_full_pin(fields, succeeds, installs):
    import os
    from pathlib import Path
    import subprocess

    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile.jetson").read_text()
    assert "ENV MATCHA_TRT_VOCODER=hifigan" in dockerfile
    block = dockerfile.split("# Optional build-time preload requires all archive integrity fields.\n", 1)[1]
    block = block.split("\n\n", 1)[0].removeprefix("RUN ").replace("\\\n", "\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("MATCHA_TRT_")}
    env.update({f"MATCHA_TRT_MODEL_{key}": fields.get(key, "") for key in ("URL", "SHA256", "SIZE")})
    env["JP_VERSION"] = "61"
    result = subprocess.run(
        ["/bin/sh", "-c", 'python3() { printf "preload-called"; };\n' + block],
        env=env, capture_output=True, text=True,
    )
    assert (result.returncode == 0) is succeeds, result.stderr
    assert ("preload-called" in result.stdout) is installs


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
