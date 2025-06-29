import collections
import math
import time
from typing import Deque, Optional

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

RATE = 16_000
PCM_SEC = RATE


class OwwDetector:
    """wake word detector using .onnx
    trained from openwakeword"""

    def __init__(
        self,
        melspec_path: str,
        embed_path: str,
        keyword_path: str,
        *,
        window: int = 16,
        threshold: float = 0.015,
        cooldown_ms: int = 1500,
        device: str = "cpu",
    ):
        prov = (
            ["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        )

        self.mel = ort.InferenceSession(melspec_path, providers=prov)
        self.emb = ort.InferenceSession(embed_path, providers=prov)
        self.cls = ort.InferenceSession(keyword_path, providers=prov)
        self.m_in = self.mel.get_inputs()[0].name
        self.m_out = self.mel.get_outputs()[0].name
        self.e_in = self.emb.get_inputs()[0].name
        self.e_out = self.emb.get_outputs()[0].name
        self.c_in = self.cls.get_inputs()[0].name
        self.c_out = self.cls.get_outputs()[0].name
        self.window = window or (self.cls.get_inputs()[0].shape[1] or 16)
        self.threshold = threshold
        # prevent consecutive ww detections on sinusoid
        self.cooldown = cooldown_ms / 1000.0
        self._last_fire = 0.0
        self._was_above = False
        self._pcm_ring: Deque[float] = collections.deque(maxlen=PCM_SEC)
        self._emb_ring: Deque[np.ndarray] = collections.deque(maxlen=120)
        self.last_score_raw = 0.0
        self.last_score_div = 0.0
        self.last_score = 0.0

    @staticmethod
    def _to_16k_mono(block: np.ndarray, src_rate: int) -> np.ndarray:
        if block.ndim == 2:
            block = block.mean(axis=1).astype(np.int16)
        if src_rate == RATE:
            return block
        up, down = RATE, src_rate
        g = math.gcd(up, down)
        up //= g
        down //= g
        res = resample_poly(block.astype(np.float32), up, down)
        return np.clip(res, -32768, 32767).astype(np.int16)

    def _pcm_to_embedding(self, pcm16: np.ndarray) -> Optional[np.ndarray]:
        self._pcm_ring.extend(pcm16.astype(np.float32))
        if len(self._pcm_ring) < PCM_SEC:
            return None
        window = np.array(self._pcm_ring, dtype=np.float32)
        mel = self.mel.run([self.m_out], {self.m_in: window[None, :]})[0].squeeze()
        mel = mel / 10.0 + 2.0
        patch = mel[-76:][None, :, :, None]
        return self.emb.run([self.e_out], {self.e_in: patch})[0].squeeze()

    def __call__(self, pcm_int16: np.ndarray, *, src_rate: int = 16_000) -> bool:
        pcm16 = self._to_16k_mono(pcm_int16, src_rate)
        emb = self._pcm_to_embedding(pcm16)
        if emb is None:
            return False
        self._emb_ring.append(emb)
        if len(self._emb_ring) < self.window:
            return False
        stack = np.stack(self._emb_ring)[-self.window :]

        self.last_score_raw = float(
            self.cls.run([self.c_out], {self.c_in: stack[None, :, :]})[0]
        )
        self.last_score_div = float(
            self.cls.run([self.c_out], {self.c_in: (stack / 32768.0)[None, :, :]})[0]
        )
        self.last_score = max(self.last_score_raw, self.last_score_div)

        now = time.time()
        above = self.last_score >= self.threshold
        fired = (
            above and not self._was_above and (now - self._last_fire) >= self.cooldown
        )
        self._was_above = above
        if fired:
            self._last_fire = now

        return fired
