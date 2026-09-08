"""Deterministic Matcha TensorRT runtime primitives.

The execution adapter is intentionally contract-driven: TensorRT plans are only
loaded after ``trt_release.load_runtime_release`` has verified their target,
checksums, and named I/O declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .trt_release import load_runtime_release


HOP_LENGTH = 256
SAMPLE_RATE = 16000


def intersperse(values: tuple[int, ...] | list[int]) -> np.ndarray:
    """Insert Matcha's blank token before, between, and after phone tokens."""
    result = np.zeros(len(values) * 2 + 1, dtype=np.int64)
    result[1::2] = values
    return result


def fix_len_compatibility(length: int, num_downsamplings_in_unet: int = 2) -> int:
    """Mirror Matcha ``fix_len_compatibility`` for the decoder U-Net."""
    if length <= 0:
        raise ValueError("length must be positive")
    factor = 2 ** num_downsamplings_in_unet
    return ((length + factor - 1) // factor) * factor


def sequence_mask(lengths: np.ndarray, max_length: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    return np.arange(max_length, dtype=np.int64)[None, :] < lengths[:, None]


def generate_path(durations: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reproduce Matcha's cumulative duration alignment map."""
    durations = np.asarray(durations, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if durations.ndim != 2 or mask.ndim != 3:
        raise ValueError("durations must be [B,T_text] and mask must be [B,T_text,T_mel]")
    if durations.shape[:2] != mask.shape[:2]:
        raise ValueError("duration and alignment mask text dimensions differ")
    cumulative = np.cumsum(durations, axis=1, dtype=np.float32)
    path = np.arange(mask.shape[2], dtype=np.float32)[None, None, :] < cumulative[:, :, None]
    previous = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    return (path.astype(np.float32) - previous.astype(np.float32)) * mask


@dataclass(frozen=True)
class RegulatedEncoder:
    mu: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    valid_frames: int


def regulate_encoder(
    mu_x: np.ndarray,
    logw: np.ndarray,
    x_mask: np.ndarray,
    length_scale: float,
) -> RegulatedEncoder:
    """Apply Matcha's duration rule and align encoder states to mel frames."""
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    mu_x = np.asarray(mu_x, dtype=np.float32)
    logw = np.asarray(logw, dtype=np.float32)
    x_mask = np.asarray(x_mask, dtype=np.float32)
    if mu_x.ndim != 3 or logw.ndim != 3 or x_mask.ndim != 3:
        raise ValueError("encoder tensors must be rank 3")
    if logw.shape != x_mask.shape or mu_x.shape[0] != logw.shape[0] or mu_x.shape[2] != logw.shape[2]:
        raise ValueError("encoder tensor dimensions do not match")

    durations = np.ceil(np.exp(logw) * x_mask) * float(length_scale)
    lengths = np.maximum(durations.sum(axis=(1, 2)), 1).astype(np.int64)
    valid_frames = int(lengths.max())
    compatible_frames = fix_len_compatibility(valid_frames)
    y_mask = sequence_mask(lengths, compatible_frames).astype(np.float32)[:, None, :]
    alignment_mask = x_mask[:, :, :, None] * y_mask[:, :, None, :]
    alignment = generate_path(durations[:, 0], alignment_mask[:, 0])
    mu_y = np.matmul(alignment.transpose(0, 2, 1), mu_x.transpose(0, 2, 1)).transpose(0, 2, 1)
    return RegulatedEncoder(mu=mu_y, mask=y_mask, lengths=lengths, valid_frames=valid_frames)


class MatchaTensorRTRuntime:
    """Own one encoder, one fixed-step solver, and one selected vocoder."""

    def __init__(self, engine_dir: str | Path, cuda=None, session_type=None):
        self.engine_dir = Path(engine_dir)
        self.manifest = load_runtime_release(self.engine_dir)
        self._closed = False
        if cuda is None or session_type is None:
            from plugins.vits2_tts_trt.runtime.backends.trt_cuda_session import (
                CudaRuntime,
                TensorRTCudaSession,
            )
            cuda = cuda or CudaRuntime()
            session_type = session_type or TensorRTCudaSession
        self.cuda = cuda
        engines = self.manifest["engines"]
        self.encoder = self._load(session_type, engines["encoder"])
        solver_name = f"solver_steps_{self.manifest['contract']['solver_steps']}"
        self.solver = self._load(session_type, engines[solver_name])
        vocoder_name = self.manifest["contract"]["vocoder"]["name"]
        self.vocoder = self._load(session_type, engines[vocoder_name])

    def _load(self, session_type, entry):
        return session_type(self.engine_dir / entry["file"], self.cuda, entry["sha256"])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for session in (self.encoder, self.solver, self.vocoder):
            close = getattr(session, "close", None)
            if close is not None:
                close()
        close = getattr(self.cuda, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
