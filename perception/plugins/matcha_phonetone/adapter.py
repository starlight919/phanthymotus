"""ORT Matcha acoustic model with independently selectable vocoders."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from .frontend import prepare_phonetone

SAMPLE_RATE = 16000
HOP_LENGTH = 256
CHUNK_BYTES = 3200
log = logging.getLogger(__name__)


def _intersperse(values: list[int]) -> list[int]:
    result = [0] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def _session(ort, path: Path, providers):
    options_factory = getattr(ort, "SessionOptions", None)
    if options_factory is None:
        return ort.InferenceSession(str(path), providers=providers)
    options = options_factory()
    level = getattr(getattr(ort, "GraphOptimizationLevel", None), "ORT_ENABLE_ALL", None)
    if level is not None:
        options.graph_optimization_level = level
    options.intra_op_num_threads = int(os.environ.get("MATCHA_ORT_INTRA_OP_THREADS", "2"))
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except TypeError:
        return ort.InferenceSession(str(path), providers=providers)


def _numpy_istft_same(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Match Vocos' ``padding=same`` ISTFT without scipy or torch."""
    mag, x, y = (np.asarray(value, dtype=np.float32) for value in (mag, x, y))
    spec = mag * (x + 1j * y)
    n_fft, hop, win = 1024, HOP_LENGTH, 1024
    frames = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32, copy=False)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float32)
    frames *= window[None, :, None]
    frame_count = frames.shape[2]
    output_size = (frame_count - 1) * hop + win
    audio = np.zeros((frames.shape[0], output_size), dtype=np.float32)
    envelope = np.zeros(output_size, dtype=np.float32)
    window_sq = window * window
    for index in range(frame_count):
        start = index * hop
        audio[:, start:start + win] += frames[:, :, index]
        envelope[start:start + win] += window_sq
    pad = (win - hop) // 2
    return audio[:, pad:-pad] / np.maximum(envelope[pad:-pad], 1e-11)


class _VocoderORT:
    def __init__(self, path: Path, device: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = _session(ort, path, providers)
        active = self._session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"vocoder CUDAExecutionProvider is not active: {active}")
        self._input = self._session.get_inputs()[0].name
        input_shape = getattr(self._session.get_inputs()[0], "shape", ())
        self._fixed_mel_frames = input_shape[2] if len(input_shape) > 2 and isinstance(input_shape[2], int) else None
        self._outputs = [item.name for item in self._session.get_outputs()]
        self._kind = "vocos" if {"mag", "x", "y"}.issubset(self._outputs) else "direct"
        log.info("[tts] Matcha vocoder=%s device=%s providers=%s path=%s",
                 self._kind, getattr(ort, "get_device", lambda: device)(), active, path)

    def decode(self, mel: np.ndarray, expected_samples: int) -> tuple[np.ndarray, float, float]:
        mel = np.asarray(mel, dtype=np.float32)
        started = time.perf_counter()
        if self._kind == "direct" and self._fixed_mel_frames:
            frame_count = mel.shape[2]
            values = []
            for offset in range(0, frame_count, self._fixed_mel_frames):
                chunk = mel[:, :, offset:offset + self._fixed_mel_frames]
                valid = chunk.shape[2]
                if valid < self._fixed_mel_frames:
                    chunk = np.pad(chunk, ((0, 0), (0, 0), (0, self._fixed_mel_frames - valid)))
                chunk_values = self._session.run(None, {self._input: chunk})
                wav_chunk = np.asarray(chunk_values[0], dtype=np.float32)
                if wav_chunk.ndim == 3:
                    wav_chunk = wav_chunk[:, 0]
                values.append(wav_chunk[:, :valid * HOP_LENGTH])
            wav = np.concatenate(values, axis=1)
        else:
            values = self._session.run(None, {self._input: mel})
            wav = None
        onnx_seconds = time.perf_counter() - started
        istft_seconds = 0.0
        if self._kind == "vocos":
            outputs = dict(zip(self._outputs, values))
            istft_started = time.perf_counter()
            wav = _numpy_istft_same(outputs["mag"], outputs["x"], outputs["y"])
            istft_seconds = time.perf_counter() - istft_started
        else:
            if wav is None:
                wav = np.asarray(values[0], dtype=np.float32)
                if wav.ndim == 3:
                    wav = wav[:, 0]
                if wav.ndim != 2:
                    raise ValueError(f"vocoder output must be [B,T] or [B,1,T], got {wav.shape}")
        return (np.asarray(wav[0, :expected_samples], dtype=np.float32),
                onnx_seconds, istft_seconds)


class MatchaPhoneToneORTAdapter:
    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0,
                 device: str = "cuda", vocoder: str = "vocos",
                 acoustic_model: str = "", vocoder_model: str = ""):
        if speaker_id != 0:
            raise ValueError("PhoneTone model supports speaker_id=0 only")
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        if device not in ("cuda", "gpu"):
            raise ValueError("PhoneTone leaderboard requires device=cuda")

        root = Path(model_dir)
        os.environ.setdefault("MATCHA_FRONTEND_RELEASE", str(root / "frontend_release"))
        import onnxruntime as ort

        requested_steps = os.environ.get("MATCHA_STEPS", "")
        if acoustic_model:
            acoustic_path = Path(acoustic_model)
        elif requested_steps:
            acoustic_path = root / f"model-steps-{requested_steps}.onnx"
        else:
            acoustic_path = next(
                (root / name for name in ("acoustic.onnx", "model-steps-5.onnx",
                                          "model-steps-3.onnx", "model.onnx")
                 if (root / name).is_file()), root / "acoustic.onnx"
            )
        if not acoustic_path.is_file():
            raise FileNotFoundError(acoustic_path)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = _session(ort, acoustic_path, providers)
        active = self._session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"acoustic CUDAExecutionProvider is not active: {active}")
        log.info("[tts] Matcha acoustic device=%s providers=%s path=%s",
                 getattr(ort, "get_device", lambda: device)(), active, acoustic_path)
        self._inputs = {item.name for item in self._session.get_inputs()}
        self._outputs = [item.name for item in self._session.get_outputs()]
        required = {"x", "x_lengths", "tones", "languages", "scales"}
        missing = required - self._inputs
        if missing:
            raise ValueError(f"PhoneTone acoustic graph missing inputs: {sorted(missing)}")

        self._vocoder_name = os.environ.get("MATCHA_VOCODER", vocoder).lower()
        self._vocoder = None
        legacy = {"wav", "wav_lengths"}.issubset(self._outputs)
        if not legacy:
            if not vocoder_model:
                names = {"vocos": ("vocos.onnx", "vocos-16khz-univ.onnx"),
                         "bigvgan": ("bigvgan.onnx",)}
                candidates = names.get(self._vocoder_name, (f"{self._vocoder_name}.onnx",))
                vocoder_path = next((root / name for name in candidates if (root / name).is_file()), None)
            else:
                vocoder_path = Path(vocoder_model)
            if vocoder_path is None or not vocoder_path.is_file():
                raise FileNotFoundError(
                    f"missing independent {self._vocoder_name} vocoder under {root}; "
                    f"set vocoder_model or add {self._vocoder_name}.onnx"
                )
            self._vocoder = _VocoderORT(vocoder_path, device)
        elif vocoder_model:
            log.warning("[tts] independent vocoder is ignored by legacy waveform graph %s", acoustic_path)

        self._speaker_id = speaker_id
        self._speed = float(speed)
        self._lock = threading.Lock()
        self.last_timings: dict[str, float | int | str] = {}

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        self._speed = float(speed)

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        with self._lock:
            started = time.perf_counter()
            frontend_started = time.perf_counter()
            result = prepare_phonetone(text)
            frontend_seconds = time.perf_counter() - frontend_started
            inputs = {
                "x": np.asarray([_intersperse(list(result.phone_ids))], dtype=np.int64),
                "tones": np.asarray([_intersperse(list(result.tone_ids))], dtype=np.int64),
                "languages": np.asarray([_intersperse(list(result.language_ids))], dtype=np.int64),
                "x_lengths": np.asarray([len(result.phone_ids) * 2 + 1], dtype=np.int64),
                "scales": np.asarray([0.667, 1.0 / self._speed], dtype=np.float32),
            }
            if "spks" in self._inputs:
                inputs["spks"] = np.asarray([self._speaker_id], dtype=np.int64)

            acoustic_started = time.perf_counter()
            values = self._session.run(None, inputs)
            acoustic_seconds = time.perf_counter() - acoustic_started
            outputs = dict(zip(self._outputs, values))
            vocoder_onnx_seconds = 0.0
            istft_seconds = 0.0
            if "wav" in outputs:
                wav = np.asarray(outputs["wav"], dtype=np.float32).reshape(-1)
                length = int(np.asarray(outputs.get("wav_lengths", [len(wav)])).reshape(-1)[0])
                vocoder_seconds = 0.0
                vocoder_name = "embedded"
                mel_frames = 0
            else:
                if "mel" not in outputs:
                    raise ValueError(f"acoustic graph must output mel or wav, got {self._outputs}")
                mel = np.asarray(outputs["mel"], dtype=np.float32)
                mel_length = int(np.asarray(outputs.get("mel_lengths", [mel.shape[-1]])).reshape(-1)[0])
                mel = mel[:, :, :mel_length]
                mel_frames = mel.shape[-1]
                expected_samples = mel_frames * HOP_LENGTH
                wav, vocoder_onnx_seconds, istft_seconds = self._vocoder.decode(mel, expected_samples)
                vocoder_seconds = vocoder_onnx_seconds + istft_seconds
                length = len(wav)
                vocoder_name = self._vocoder_name

            post_started = time.perf_counter()
            samples = np.clip(wav[:length] * 32767, -32768, 32767).astype("<i2", copy=False)
            postprocess_seconds = time.perf_counter() - post_started
            total_seconds = time.perf_counter() - started
            self.last_timings = {
                "frontend_seconds": frontend_seconds,
                "acoustic_seconds": acoustic_seconds,
                "vocoder_seconds": vocoder_seconds,
                "vocoder_onnx_seconds": vocoder_onnx_seconds,
                "istft_seconds": istft_seconds,
                "postprocess_seconds": postprocess_seconds,
                "total_seconds": total_seconds,
                "vocoder": vocoder_name,
                "mel_frames": mel_frames,
                "audio_seconds": len(samples) / (SAMPLE_RATE * 2),
            }
            return samples.tobytes()

    def synthesize_stream(self, text: str):
        pcm = self.synthesize(text)
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]

    def warmup(self) -> int:
        return sum(len(chunk) for chunk in self.synthesize_stream("你好。"))
