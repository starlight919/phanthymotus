"""Fail-closed validation for target-specific Matcha TensorRT releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


RUNTIME_SCHEMA_VERSION = 1
RUNTIME_READY_STATUSES = {
    "plan-ready",
    "runtime-ready",
    "framework-validated",
    "approved",
    "deployed",
}
TARGET_TRT_MAJORS = {"jp511": 8, "jp61": 10}
REQUIRED_CONTRACT_FIELDS = {
    "sample_rate",
    "mel_channels",
    "mel_normalization",
    "frontend",
    "solver_steps",
    "vocoder",
}
VOCOS_ISTFT_FIELDS = {"n_fft", "hop_length", "window", "padding"}


def load_runtime_release(engine_dir: str | Path) -> dict:
    """Load and validate the release manifest next to target-specific plans."""
    root = Path(engine_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Matcha TensorRT manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Matcha TensorRT manifest: {manifest_path}") from error
    _validate_manifest(root, manifest)
    return manifest


def _validate_manifest(root: Path, manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Matcha TensorRT manifest must be an object")
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("Unsupported Matcha TensorRT manifest schema_version")
    if manifest.get("release_status") not in RUNTIME_READY_STATUSES:
        raise ValueError("Matcha TensorRT release is neither plan-ready nor runtime-ready")
    target = manifest.get("target")
    if target not in TARGET_TRT_MAJORS:
        raise ValueError("Matcha TensorRT manifest has an unsupported target")
    if manifest.get("tensorrt_major") != TARGET_TRT_MAJORS[target]:
        raise ValueError("Matcha TensorRT target and TensorRT major disagree")
    if not isinstance(manifest.get("compute_capability"), str):
        raise ValueError("Matcha TensorRT manifest requires compute_capability")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or REQUIRED_CONTRACT_FIELDS - contract.keys():
        raise ValueError("Matcha TensorRT manifest has an incomplete runtime contract")
    if contract["sample_rate"] != 16000 or contract["mel_channels"] != 80:
        raise ValueError("Unsupported Matcha audio contract")
    if not isinstance(contract["solver_steps"], int) or contract["solver_steps"] < 2:
        raise ValueError("Matcha TensorRT contract requires solver_steps >= 2")
    if not isinstance(contract["frontend"], dict) or not contract["frontend"].get("sha256"):
        raise ValueError("Matcha TensorRT contract requires a pinned frontend release")
    if not isinstance(contract["vocoder"], dict) or not contract["vocoder"].get("name"):
        raise ValueError("Matcha TensorRT contract requires a selected vocoder")
    _validate_vocoder_contract(contract["vocoder"])

    engines = manifest.get("engines")
    if not isinstance(engines, dict):
        raise ValueError("Matcha TensorRT manifest requires engines")
    required = {"encoder", f"solver_steps_{contract['solver_steps']}", contract["vocoder"]["name"]}
    if set(engines) != required:
        raise ValueError("Matcha TensorRT manifest engine set does not match its contract")
    for name, entry in engines.items():
        _validate_engine(root, name, entry)


def _validate_vocoder_contract(vocoder: dict) -> None:
    if vocoder["name"] != "vocos":
        return
    istft = vocoder.get("istft")
    if not isinstance(istft, dict) or VOCOS_ISTFT_FIELDS - istft.keys():
        raise ValueError("Vocos runtime requires a declared spectral/ISTFT contract")
    if (istft["n_fft"], istft["hop_length"], istft["window"], istft["padding"]) != (
        1024, 256, "hann_periodic", "same",
    ):
        raise ValueError("Unsupported Vocos ISTFT contract")


def _validate_engine(root: Path, name: str, entry: object) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid engine entry: {name}")
    file_name = entry.get("file")
    sha256 = entry.get("sha256")
    size = entry.get("size")
    bindings = entry.get("bindings")
    if not isinstance(file_name, str) or not _safe_relpath(file_name):
        raise ValueError(f"Invalid engine path: {name}")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"Invalid engine SHA256: {name}")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Invalid engine size: {name}")
    if not isinstance(bindings, dict) or not bindings.get("inputs") or not bindings.get("outputs"):
        raise ValueError(f"Incomplete named I/O contract: {name}")
    for direction in ("inputs", "outputs"):
        values = bindings[direction]
        if not isinstance(values, dict) or not all(_valid_binding(v) for v in values.values()):
            raise ValueError(f"Invalid {direction} contract: {name}")
    path = root / file_name
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Engine file size mismatch: {name}")
    if _sha256(path) != sha256.lower():
        raise ValueError(f"Engine file SHA256 mismatch: {name}")


def _valid_binding(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("dtype"), str) and isinstance(value.get("shape"), list)


def _safe_relpath(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value and value not in ("", ".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
