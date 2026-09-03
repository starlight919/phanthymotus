"""ONNX Runtime adapter for the PhoneTone Matcha graph."""

from __future__ import annotations

import os
import threading
import logging
from pathlib import Path

import numpy as np

from .frontend import prepare_phonetone

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200
log = logging.getLogger(__name__)


def _intersperse(values: list[int]) -> list[int]:
    result = [0] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


class MatchaPhoneToneORTAdapter:
    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "cuda"):
        if speaker_id != 0:
            raise ValueError("PhoneTone model supports speaker_id=0 only")
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        root = Path(model_dir)
        os.environ.setdefault("MATCHA_FRONTEND_RELEASE", str(root / "frontend_release"))
        import onnxruntime as ort

        graph = root / "model.onnx"
        if not graph.is_file():
            raise FileNotFoundError(graph)
        if device not in ("cuda", "gpu"):
            raise ValueError("PhoneTone leaderboard requires device=cuda")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(graph), providers=providers)
        active_providers = self._session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(f"CUDAExecutionProvider is not active: {active_providers}")
        log.info("[tts] Matcha ORT device=%s providers=%s", ort.get_device(), active_providers)
        self._inputs = {item.name for item in self._session.get_inputs()}
        required = {"x", "x_lengths", "tones", "languages", "scales"}
        missing = required - self._inputs
        if missing:
            raise ValueError(f"PhoneTone graph missing inputs: {sorted(missing)}")
        self._speaker_id = speaker_id
        self._speed = float(speed)
        self._lock = threading.Lock()

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        self._speed = float(speed)

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        with self._lock:
            result = prepare_phonetone(text)
            inputs = {
                "x": np.asarray([_intersperse(list(result.phone_ids))], dtype=np.int64),
                "tones": np.asarray([_intersperse(list(result.tone_ids))], dtype=np.int64),
                "languages": np.asarray([_intersperse(list(result.language_ids))], dtype=np.int64),
                "x_lengths": np.asarray([len(result.phone_ids) * 2 + 1], dtype=np.int64),
                "scales": np.asarray([0.667, 1.0 / self._speed], dtype=np.float32),
            }
            if "spks" in self._inputs:
                inputs["spks"] = np.asarray([self._speaker_id], dtype=np.int64)
            output = self._session.run(None, inputs)
            names = [item.name for item in self._session.get_outputs()]
            if "wav" not in names or "wav_lengths" not in names:
                raise ValueError("PhoneTone graph must output wav and wav_lengths")
            wav = np.asarray(output[names.index("wav")]).reshape(-1)
            length = int(np.asarray(output[names.index("wav_lengths")]).reshape(-1)[0])
        samples = np.clip(wav[:length] * 32767, -32768, 32767).astype("<i2", copy=False)
        return samples.tobytes()

    def synthesize_stream(self, text: str):
        pcm = self.synthesize(text)
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]

    def warmup(self) -> int:
        return sum(len(chunk) for chunk in self.synthesize_stream("你好。"))
