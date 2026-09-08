"""Contract-driven Matcha PhoneTone TensorRT synthesis adapter."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from plugins.tts import CHUNK_BYTES, TTSAdapter

from .frontend import prepare_phonetone
from .trt_runtime import HOP_LENGTH, SAMPLE_RATE, MatchaTensorRTRuntime, intersperse, regulate_encoder


_VOCOS_REQUIRED_CONTRACT = {"n_fft", "hop_length", "window", "padding"}


def _numpy_istft_same(mag: np.ndarray, x: np.ndarray, y: np.ndarray, *,
                      n_fft: int, hop_length: int, window: str) -> np.ndarray:
    """Reconstruct Vocos ``padding=same`` audio without a torch dependency."""
    if n_fft <= 0 or hop_length <= 0 or window != "hann_periodic":
        raise ValueError("unsupported Vocos ISTFT contract")
    mag, x, y = (np.asarray(value, dtype=np.float32) for value in (mag, x, y))
    if mag.shape != x.shape or mag.shape != y.shape or mag.ndim != 3:
        raise ValueError("Vocos spectral outputs must have matching [B,F,T] shapes")
    if mag.shape[1] != n_fft // 2 + 1:
        raise ValueError("Vocos spectral bins do not match the declared FFT size")
    spectrum = mag * (x + 1j * y)
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1).astype(np.float32, copy=False)
    analysis_window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    frames *= analysis_window[None, :, None]
    frame_count = frames.shape[2]
    output_size = (frame_count - 1) * hop_length + n_fft
    audio = np.zeros((frames.shape[0], output_size), dtype=np.float32)
    envelope = np.zeros(output_size, dtype=np.float32)
    window_sq = analysis_window * analysis_window
    for index in range(frame_count):
        start = index * hop_length
        audio[:, start:start + n_fft] += frames[:, :, index]
        envelope[start:start + n_fft] += window_sq
    padding = (n_fft - hop_length) // 2
    return audio[:, padding:-padding] / np.maximum(envelope[padding:-padding], 1e-11)


class MatchaTensorRTAdapter(TTSAdapter):
    """Run the three-engine Matcha graph with only manifest-declared bindings.

    The release must use the canonical named TensorRT contract. In particular, a
    Vocos release requires a runtime implementation of its explicitly-declared
    spectral/ISTFT contract and is rejected until that contract is available.
    """

    def __init__(self, engine_dir: str | Path, speed: float = 1.0, runtime=None):
        self._lock = threading.Lock()
        self._runtime = runtime or MatchaTensorRTRuntime(engine_dir)
        self._manifest = self._runtime.manifest
        self._contract = self._manifest["contract"]
        self.set_speed(speed)
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        engines = self._manifest["engines"]
        solver_name = f"solver_steps_{self._contract['solver_steps']}"
        self._require_bindings(engines["encoder"], {"x", "x_lengths", "tones", "languages"},
                               {"mu_x", "logw", "x_mask"})
        self._require_bindings(engines[solver_name], {"noise", "mask", "mu"}, {"mel"})
        vocoder = self._contract["vocoder"]["name"]
        if vocoder == "vocos":
            istft = self._contract["vocoder"].get("istft")
            if not isinstance(istft, dict) or _VOCOS_REQUIRED_CONTRACT - istft.keys():
                raise ValueError("Vocos TensorRT runtime requires a declared spectral/ISTFT contract")
            if istft["padding"] != "same":
                raise ValueError("unsupported Vocos ISTFT contract")
            self._require_bindings(engines[vocoder], {"mel"}, {"mag", "x", "y"})
            return
        self._require_bindings(engines[vocoder], {"mel"}, {"audio"})

    @staticmethod
    def _require_bindings(entry: dict, inputs: set[str], outputs: set[str]) -> None:
        bindings = entry["bindings"]
        missing_inputs = inputs - set(bindings["inputs"])
        missing_outputs = outputs - set(bindings["outputs"])
        if missing_inputs or missing_outputs:
            raise ValueError(
                "Matcha TensorRT engine lacks canonical bindings: "
                f"inputs={sorted(missing_inputs)} outputs={sorted(missing_outputs)}"
            )

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if not 0 < speed <= 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        self._speed = speed

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        with self._lock:
            frontend = prepare_phonetone(text)
            x = intersperse(frontend.phone_ids)[None, :]
            tones = intersperse(frontend.tone_ids)[None, :]
            languages = intersperse(frontend.language_ids)[None, :]
            x_lengths = np.asarray([x.shape[1]], dtype=np.int64)
            encoded = self._runtime.encoder.run({
                "x": x,
                "x_lengths": x_lengths,
                "tones": tones,
                "languages": languages,
            })
            regulated = regulate_encoder(
                encoded["mu_x"], encoded["logw"], encoded["x_mask"], 1.0 / self._speed
            )
            noise = np.zeros_like(regulated.mu, dtype=np.float32)
            mel = self._runtime.solver.run({
                "noise": noise,
                "mask": regulated.mask,
                "mu": regulated.mu,
            })["mel"]
            valid_mel = mel[:, :, :regulated.valid_frames]
            vocoder_outputs = self._runtime.vocoder.run({"mel": valid_mel})
            if self._contract["vocoder"]["name"] == "vocos":
                istft = self._contract["vocoder"]["istft"]
                audio = _numpy_istft_same(
                    vocoder_outputs["mag"], vocoder_outputs["x"], vocoder_outputs["y"],
                    n_fft=int(istft["n_fft"]), hop_length=int(istft["hop_length"]),
                    window=istft["window"],
                )
            else:
                audio = vocoder_outputs["audio"]
            pcm = self._pcm16(audio, regulated.valid_frames)
        for offset in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[offset:offset + CHUNK_BYTES]

    @staticmethod
    def _pcm16(audio: np.ndarray, valid_frames: int) -> bytes:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        expected_samples = valid_frames * HOP_LENGTH
        if samples.size < expected_samples:
            raise RuntimeError(
                f"vocoder output is shorter than the declared mel duration: {samples.size} < {expected_samples}"
            )
        samples = samples[:expected_samples]
        return np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def close(self) -> None:
        self._runtime.close()
