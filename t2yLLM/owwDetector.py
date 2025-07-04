import numpy as np
import onnxruntime as ort
import time
import math
from scipy.signal import resample_poly
from line_profiler import profile
from typing import Optional

RATE = 16_000
PCM_SEC = RATE


class OwwDetector:
    def __init__(
        self,
        melspec_path: str,
        embed_path: str,
        keyword_path: str,
        *,
        threads: int = 1,
        window: int = 16,
        threshold: float = 0.015,
        cooldown_ms: int = 1500,
        device: str = "cpu",
    ):
        prov = (
            ["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.mel = ort.InferenceSession(
            melspec_path, sess_options=sess_options, providers=prov
        )
        self.emb = ort.InferenceSession(
            embed_path, sess_options=sess_options, providers=prov
        )
        self.cls = ort.InferenceSession(
            keyword_path, sess_options=sess_options, providers=prov
        )

        self.m_in = self.mel.get_inputs()[0].name
        self.m_out = self.mel.get_outputs()[0].name
        self.e_in = self.emb.get_inputs()[0].name
        self.e_out = self.emb.get_outputs()[0].name
        self.c_in = self.cls.get_inputs()[0].name
        self.c_out = self.cls.get_outputs()[0].name

        self.window = window
        self.threshold = threshold
        self.cooldown = cooldown_ms / 1000.0

        self.emb_size = self.emb.get_outputs()[0].shape[-1]
        self._pcm_buffer = np.zeros(PCM_SEC, dtype=np.float32)
        self._emb_buffer = np.zeros((self.window, self.emb_size), dtype=np.float32)

        self._pcm_samples_processed = 0
        self._emb_frames_collected = 0

        self._last_fire = 0.0
        self._was_above = False

        self.last_score = 0.0
        self.last_score_raw = 0.0
        self.last_score_div = 0.0

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

    @profile
    def _pcm_to_embedding(self, pcm16: np.ndarray) -> Optional[np.ndarray]:
        pcm_float = pcm16.astype(np.float32) / 32768.0

        self._pcm_buffer = np.roll(self._pcm_buffer, -len(pcm_float))
        self._pcm_buffer[-len(pcm_float) :] = pcm_float

        if self._pcm_samples_processed < PCM_SEC:
            self._pcm_samples_processed += len(pcm16)
            if self._pcm_samples_processed < PCM_SEC:
                return None

        mel = self.mel.run([self.m_out], {self.m_in: self._pcm_buffer[None, :]})[
            0
        ].squeeze()
        mel = mel / 10.0 + 2.0

        patch = mel[-76:][None, :, :, None]
        return self.emb.run([self.e_out], {self.e_in: patch})[0].squeeze()

    @profile
    def __call__(self, pcm_int16: np.ndarray, *, src_rate: int = 16_000) -> bool:
        pcm16 = self._to_16k_mono(pcm_int16, src_rate)
        emb = self._pcm_to_embedding(pcm16)

        if emb is None:
            return False

        self._emb_buffer = np.roll(self._emb_buffer, shift=-1, axis=0)
        self._emb_buffer[-1, :] = emb

        if self._emb_frames_collected < self.window:
            self._emb_frames_collected += 1
            return False

        stack = self._emb_buffer[None, :, :]

        self.last_score_raw = float(self.cls.run([self.c_out], {self.c_in: stack})[0])
        self.last_score_div = float(
            self.cls.run([self.c_out], {self.c_in: stack / 32768.0})[0]
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
