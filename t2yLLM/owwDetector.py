"""class responsible for getting openwakeword classifier results"""

import numpy as np
from typing import Union
import onnxruntime as ort
import time
import math
from scipy.signal import resample_poly
from line_profiler import profile
from typing import Optional

RATE = 16_000
PCM_SEC = RATE


class MDCBuffer:
    """circular buffer accepting multi dim inputs
    instead of np.roll() copies"""

    def __init__(self, shape: Union[int, tuple], dtype: np.dtype = np.float32):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.dtype = dtype
        self.buffer = np.zeros(shape, dtype=dtype)
        self.index = 0
        self.size = shape[0]
        self.processed = False

    def ins(self, data: np.ndarray):
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 0:
            self.buffer[self.index] = data
            self.index = (self.index + 1) % self.size
            if self.index == 0:
                self.processed = True
        if data.ndim == 1:
            if self.buffer.ndim == 1:
                for val in data:
                    self.buffer[self.index] = val
                    self.index = (self.index + 1) % self.size
                    if self.index == 0:
                        self.processed = True
            else:
                self.buffer[self.index] = data
                self.index = (self.index + 1) % self.size
                if self.index == 0:
                    self.processed = True
        else:
            for row in data:
                self.buffer[self.index] = row
                self.index = (self.index + 1) % self.size
                if self.index == 0:
                    self.processed = True

    def get(self):
        """
        gets buffer data in chronological order
        """
        if not self.processed:
            return self.buffer[: self.index].copy()
        else:
            if self.buffer.ndim == 1:
                return np.concatenate(
                    [self.buffer[self.index :], self.buffer[: self.index]]
                )
            else:
                return np.concatenate(
                    [self.buffer[self.index :], self.buffer[: self.index]], axis=0
                )

    def __getitem__(self, key):
        """
        gets buffer data data in storage order
        """
        return self.buffer[key]

    def getpart(self, k: int):
        """
        gets k last parts of data (chronological)
        """
        if k > self.size:
            k = self.size
        if not self.processed and k > self.index:
            k = self.index

        if k == 1:
            return self.buffer[(self.index - 1) % self.size]
        else:
            seg = [(self.index - i - 1) % self.size for i in range(k)]
            return self.buffer[seg[::-1]]

    def isproc(self):
        return self.processed

    def clear(self):
        self.buffer.fill(0)
        self.index = 0
        self.processed = False

    def __len__(self):
        return self.size if self.processed else self.index


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

        self._pcm_buffer = MDCBuffer(PCM_SEC, dtype=np.float32)
        self._emb_buffer = MDCBuffer((self.window, self.emb_size), dtype=np.float32)

        self._pcm_samples_processed = 0
        self._emb_frames_collected = 0

        self._last_fire = 0.0
        self._was_above = False

        self.last_score = 0.0
        self.last_score_raw = 0.0
        self.last_score_div = 0.0

    @staticmethod
    def _to_16k_mono(block: np.ndarray, src_rate: int) -> np.ndarray:
        """resamples input signal converting it 16kHz mono"""
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
        self._pcm_buffer.ins(pcm_float)

        if self._pcm_samples_processed < PCM_SEC:
            self._pcm_samples_processed += len(pcm16)
            if self._pcm_samples_processed < PCM_SEC:
                return None
        pcm_data = self._pcm_buffer.get()
        mel = self.mel.run([self.m_out], {self.m_in: pcm_data[None, :]})[0].squeeze()
        mel = mel / 10.0 + 2.0

        patch = mel[-76:][None, :, :, None]
        return self.emb.run([self.e_out], {self.e_in: patch})[0].squeeze()

    @profile
    def __call__(self, pcm_int16: np.ndarray, *, src_rate: int = 16_000) -> bool:
        pcm16 = self._to_16k_mono(pcm_int16, src_rate)
        emb = self._pcm_to_embedding(pcm16)

        if emb is None:
            return False

        self._emb_buffer.ins(emb)

        if self._emb_frames_collected < self.window:
            self._emb_frames_collected += 1
            return False

        emb_data = self._emb_buffer.get()
        stack = emb_data[None, :, :]

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
