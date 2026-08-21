"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_vits2_tn(model_dir: str) -> None:
    """Verify the compiled TN assets shipped with a VITS2 release."""
    tn_dir = os.path.join(model_dir, "tn_cache")
    sums_path = os.path.join(tn_dir, "SHA256SUMS")
    required = ("zh_tn_tagger.fst", "zh_tn_verbalizer.fst")
    if not os.path.isfile(sums_path):
        raise RuntimeError(f"VITS2 TN checksum file is missing: {sums_path}")

    expected = {}
    with open(sums_path, encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) == 2:
                expected[fields[1].lstrip("*")] = fields[0].lower()
    for name in required:
        path = os.path.join(tn_dir, name)
        if name not in expected or not os.path.isfile(path):
            raise RuntimeError(f"VITS2 TN asset is incomplete: {name}")
        actual = _sha256(path)
        if actual != expected[name]:
            raise RuntimeError(
                f"VITS2 TN SHA256 mismatch for {name}: "
                f"expected {expected[name]}, got {actual}"
            )

    # New releases carry source/build provenance.  Keep old, checksum-valid
    # releases loadable while validating the stronger contract when present.
    manifest_path = os.path.join(tn_dir, "tn_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest_fst = manifest.get("fst", {})
        for name in required:
            recorded = manifest_fst.get(name, {}).get("sha256", "").lower()
            if recorded != expected[name]:
                raise RuntimeError(
                    f"VITS2 TN manifest disagrees with SHA256SUMS for {name}"
                )


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
    return hook

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "asr_en": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.zip",
        "check_file": "tokens.txt",
    },
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{COS_BASE}/matcha-icefall-zh-en.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_vocoder": {
        "url": f"{COS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "vits2": {
        "url": os.environ.get("VITS2_MODEL_URL", ""),
        "check_file": "config.json",
        "check_files": (
            "config.json",
            "engines/manifest.json",
            "tn_cache/zh_tn_tagger.fst",
            "tn_cache/zh_tn_verbalizer.fst",
            "frontend_data/phrase_pinyin_data/di.py",
            "nltk_data/corpora/cmudict.zip",
        ),
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,  # Not an archive, just a single file download
    },
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_files = info.get("check_files", (info["check_file"],))
    missing = [
        relative for relative in check_files
        if not os.path.isfile(os.path.join(model_dir, relative))
    ]
    if not missing:
        if name == "vits2":
            _verify_vits2_tn(model_dir)
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    if not url:
        raise RuntimeError(
            f"No download URL configured for {name}; set VITS2_MODEL_URL or "
            f"mount a complete model at {model_dir}; missing: {', '.join(missing)}"
        )
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive)
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    # Determine suffix from URL
    if url.endswith(".zip"):
        suffix = ".zip"
    elif url.endswith(".tar.gz") or url.endswith(".tgz"):
        suffix = ".tar.gz"
    else:
        suffix = ".tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        expected_sha256 = (
            os.environ.get("VITS2_MODEL_SHA256", "").strip().lower()
            if name == "vits2" else ""
        )
        if expected_sha256:
            actual_sha256 = _sha256(tmp_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"[model_downloader] {name}: SHA256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")

        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(
                tmp_path,
                model_dir,
                "r:gz" if suffix == ".tar.gz" else "r:bz2",
            )

        if name == "vits2":
            _verify_vits2_tn(model_dir)

        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Verify
    missing = [
        relative for relative in check_files
        if not os.path.isfile(os.path.join(model_dir, relative))
    ]
    if missing:
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but required files "
            f"are missing in {model_dir}: {', '.join(missing)}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Filter out __MACOSX and directory entries
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str, mode: str = "r:bz2") -> None:
    """Extract an archive, stripping its common top-level directory prefix."""
    with tarfile.open(tar_path, mode) as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
