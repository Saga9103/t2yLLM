from .config.yamlConfigLoader import Loader
import os
from pathlib import Path
import socket
import threading
import queue
import subprocess
import gc
import time
import logging
import re
import sys
from collections import deque
from typing import Union
import uuid
from functools import wraps

from datetime import datetime

# MATH:
import numpy as np
import math
import torch

# SOUND
import pyaudio

# sound resample
import scipy.io.wavfile as wav
from scipy.signal import resample, resample_poly

# near realtime STT with WhisperLive
from faster_whisper import WhisperModel, BatchedInferencePipeline

# wake word detection
import pvporcupine

# piper
CURRENTDIR = Path(__file__).resolve().parent
PIPERROOT = CURRENTDIR / "config" / "piper"

os.environ["LD_LIBRARY_PATH"] = f"{PIPERROOT}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["ESPEAKNG_DATA_PATH"] = str(PIPERROOT / "espeak-ng-data")

# add parent DIR for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def not_implemented(obj):
    if isinstance(obj, type):
        original_init = obj.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            raise NotImplementedError(f"{obj.__name__} is not implemented")

        obj.__init__ = new_init

        return obj
    else:

        @wraps(obj)
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"{obj.__name__}() is not implemented")

        return wrapper


class LocalAudio:
    def __init__(self):
        # audio params
        self.sample_rate = 16000
        self.jabra_rate = 48000
        self.chunk_size = 512
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.jabra_channels = 2
        self.volume = 1.0
        self.buffer_ms_size = 30
        # setup
        self.audio = pyaudio.PyAudio()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.audio_buffer = bytearray()
        self.fragments = {}
        self.msg_id = None  # later with pydantic
        self.running = False
        """
        class variables controlling that
        the devices stays open during the full
        stream and not fragment based. Else it induces 
        latency and cracking sounds because of closing and 
        opening intermitently
        """
        self.last_fragment_time = 0.0
        self.fragment_timeout = 5.0  # in seconds
        self.device_open = False
        self.stream_on = False
        """
        padding with silence still doesnt solves the
        craking sound of audio closing and opening at the end 
        and begining of audio. Annoying
        """
        self.padding = b"\x00" * 2 * self.chunk_size
        # connect to jabra or respeaker
        self.input_thread = None
        self.output_thread = None
        self.input_stream = None
        self.output_stream = None

        self.indev_idx = None
        self.outdev_idx = None

        self.threads = []

        self.is_jabra = False
        self.is_respeaker = False

        self.up_factor, self.down_factor = self.ratio(self.jabra_rate, self.sample_rate)

        self.logger = logging.getLogger("LocalAudio")

        self.get_device()
        self.start()

    def get_device(self):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            device_name = info["name"].lower()

            if any(dev in device_name for dev in ("jabra", "respeaker", "usb")):
                if "jabra" in device_name:
                    self.is_jabra = True
                if info["maxInputChannels"] > 0:
                    self.indev_idx = i
                    self.logger.info(f"Found input device: {info['name']} (index {i})")
                if info["maxOutputChannels"] > 0:
                    self.outdev_idx = i
                    self.logger.info(f"Found output device: {info['name']} (index {i})")

    def start(self):
        if self.running:
            return
        self.running = True
        input_thread = threading.Thread(target=self.audio_input, daemon=True)
        input_thread.start()
        self.threads.append(input_thread)
        output_thread = threading.Thread(target=self.audio_output, daemon=True)
        output_thread.start()
        self.threads.append(output_thread)

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2.0)
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

    def audio_input(self):
        try:
            self.input_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.indev_idx,
                frames_per_buffer=self.chunk_size,
            )
            self.logger.info("Audio input started")

            while self.running:
                try:
                    data = self.input_stream.read(
                        self.chunk_size, exception_on_overflow=False
                    )
                    self.input_queue.put(data)
                except Exception as e:
                    self.logger.error(f"Error reading audio: {e}")
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Failed to start audio input: {e}")

    def ratio(self, target, source):
        gcd = math.gcd(target, source)
        return target // gcd, source // gcd

    def convert16_48(self, pcm_data: bytes):
        mono16 = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        mono48 = resample_poly(mono16, up=self.up_factor, down=self.down_factor, axis=0)
        stereo48 = np.repeat(mono48[:, None], 2, axis=1).ravel()
        int16_data = np.clip(stereo48 * 32768.0, -32768, 32767).astype(np.int16)
        return int16_data.tobytes()

    def audio_output(self):
        try:
            if self.is_jabra:
                self.output_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.jabra_channels,
                    rate=self.jabra_rate,
                    output=True,
                    output_device_index=self.outdev_idx,
                    frames_per_buffer=self.chunk_size,
                )
            else:
                self.output_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.outdev_idx,
                    frames_per_buffer=self.chunk_size,
                )
            self.logger.info("Voice function started")

            while self.running:
                try:
                    audio_data = self.output_queue.get(timeout=0.1)

                    if audio_data == b"__END_OF_AUDIO__":
                        self.stream_on = False
                        continue

                    if not audio_data or len(audio_data) == 0:
                        continue

                    self.stream_on = True

                    if len(audio_data) % 2 != 0:
                        audio_data = audio_data[:-1]
                    if self.volume != 1.0:
                        pcm_data = np.frombuffer(audio_data, dtype=np.int16).astype(
                            np.float32
                        )
                        pcm_data = np.clip(
                            pcm_data * self.volume, -32768, 32767
                        ).astype(np.int16)
                        audio_data = pcm_data.tobytes()

                    if self.is_jabra:
                        audio_data = self.convert16_48(audio_data)
                        self.output_stream.write(audio_data)

                    else:
                        self.output_stream.write(audio_data)

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error playing audio: {e}")
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Failed to start audio output: {e}")

    def get_chunk(self):
        try:
            return self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def play(self, audio_data):
        if audio_data:
            chunk_size = self.chunk_size * 2
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                self.output_queue.put(chunk)

    def send_completion(self):
        self.output_queue.put(b"__END_OF_AUDIO__")


@not_implemented
class WLanAudio:
    def __init__(self):
        pass


class LocalDispatcher:
    def __init__(self, audio_handler=None):
        # from config
        self.voice_config = Loader().loadWhispConfig()  # load the .yaml
        porcupine_dir = Path("./porcupine")
        porcupine_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("LocalDispatcher")

        # faster-whisper
        self.fast_whisper_model = WhisperModel(
            "quantized", device="cuda", compute_type="int8_float16"
        )
        self.batched_model = BatchedInferencePipeline(model=self.fast_whisper_model)

        self.running = False
        self.is_recording = False
        self.wakeword_detected = False
        self.detection_time = 0

        self.audio_buffer = bytearray()
        self.pre_recording_buffer = deque(
            maxlen=int(
                self.voice_config.audio.sample_rate * 0.2
            )  # 0.2 seconds of 16kHz audio
        )

        self.audio_receiving_buffer = bytearray()
        self.last_buffer_rcv_activity = 0.0

        self.client_address = "127.0.0.1"  # Local mode

        self.last_detection_time = 0
        # comm handlers
        # command_queue to send to the LLM
        self.command_queue = queue.Queue()
        # whisper_queue for processing whisper transcriptions
        self.whisper_queue = queue.Queue()
        # response_queue for sending audio to the client
        self.response_queue = queue.Queue()

        self.recent_transcriptions = []  # Keeping track of recent transcriptions
        self.last_audio_sent_time = 0  # Time when audio was last sent
        self.waiting_for_completion = False
        self.completion_message_id = None
        # if no activity, the server will reset its state
        self.reset_server_delay = int(
            self.voice_config.network.server_reset_delay
        )  # in seconds
        self.msg_id = None
        # had a problem with loopbacks
        self.ignore_own_audio_seconds = self.voice_config.audio.ignore_own
        # in anycase, it cant be longer than keyword silence threshold
        self.threads = []
        self.lock = threading.Lock()

        # silero params
        self.speech_end_silence_start = 0
        self.is_silero_speech_active = False
        self.silero_sensitivity = 0.5
        self.post_speech_silence_duration = self.voice_config.audio.EOS  # secondes
        self.wake_word_timeout = 4.0  # seconds

        # pvporcupine params
        try:
            self.porcupine = pvporcupine.create(
                access_key=os.environ.get("PORCUPINE_KEY"),
                keyword_paths=[
                    str(porcupine_dir / self.voice_config.model.porcupine_keyword_path)
                ],
                # that, you have to download from them, it doesnt come
                # with the .ppn file
                model_path=str(porcupine_dir / self.voice_config.model.porcupine_path),
                sensitivities=[0.6],  # defaults to 0.5 if not
            )
        except Exception as e:
            self.logger.error(
                f"\033[91mError loading porcupine, create a custom model and ensure you both have .pv and .ppn files : {e}\033[0m"
            )
            self.porcupine = None
        self.porcupine_buffer = bytearray()
        self.porcupine_frame_length = 512
        if self.porcupine:
            self.porcupine_frame_length = self.porcupine.frame_length
        self.keyword_detected = False
        self.keyword_hang = 4.0  # in seconds
        self.detection_time = 0.0  # in seconds
        self.count = 0

        self._init_silero_vad()

        if audio_handler is None:  # to instanciate differently
            audio_handler = LocalAudio()
        self.audio_handler = audio_handler

    def start(self):
        if self.running:
            return

        self.running = True
        self.threads = []
        self.client_address = "127.0.0.1"  # Local mode
        self.active_stream = False
        self.last_detection_time = time.time()
        self.last_audio_sent_time = time.time() - 180
        self.waiting_for_completion = False
        self.completion_message_id = None

        os.makedirs(self.voice_config.model.tmpfs_dir, exist_ok=True)

        audio_thread = threading.Thread(
            target=self.audio_processing_thread, daemon=True, name="AudioProcessor"
        )
        audio_thread.start()
        self.threads.append(audio_thread)

        whisper_thread = threading.Thread(
            target=self.whisper_handler, daemon=True, name="WhisperProcessor"
        )
        whisper_thread.start()
        self.threads.append(whisper_thread)

        command_thread = threading.Thread(
            target=self.command_handler, daemon=True, name="CommandProcessor"
        )
        command_thread.start()
        self.threads.append(command_thread)

        response_thread = threading.Thread(
            target=self.tts_response_handler, daemon=True, name="ResponseProcessor"
        )
        response_thread.start()
        self.threads.append(response_thread)

        llm_thread = threading.Thread(
            target=self.llm_response_handler, daemon=True, name="LLMListener"
        )
        llm_thread.start()
        self.threads.append(llm_thread)

        completion_thread = threading.Thread(
            target=self.completion_signal_listener,
            daemon=True,
            name="CompletionListener",
        )
        completion_thread.start()
        self.threads.append(completion_thread)

        self.logger.info("All threads are UP")

    def stop(self):
        self.logger.info("Stopping Voice Assist")
        self.running = False

        if hasattr(self, "threads"):
            for thread in self.threads:
                thread.join(timeout=2.0)

        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Voice Assist stopped")

    def audio_processing_thread(self):
        self.logger.info("Local audio processing thread started")

        while self.running:
            try:
                audio_chunk = self.audio_handler.get_chunk()

                if audio_chunk and not self.audio_handler.stream_on:
                    self.process_audio_chunk(audio_chunk)

            except Exception:
                time.sleep(0.1)

    def completion_signal_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)

        try:
            sock.bind(("127.0.0.1", self.voice_config.network.RCV_END_SIGNAL))

            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)

                    if not self.running:
                        break

                    if data:
                        try:
                            signal = data.decode("utf-8")
                            # self.logger.info(f"Received completion signal: {signal}")

                            if signal.startswith("__DONE__"):
                                # we extract the message ID
                                if "[" in signal and "]" in signal:
                                    self.msg_id = signal[
                                        signal.find("[") + 1 : signal.find("]")
                                    ]
                                """
                                self.logger.info(
                                    f"msg_id={self.msg_id}, waiting for={
                                        self.completion_message_id
                                    }"
                                )
                                """
                                # if message id is a completion id message then we safely set
                                # completion indicators
                                # and resume streaming
                                if (
                                    not self.completion_message_id
                                    or self.msg_id == self.completion_message_id
                                ):
                                    self.waiting_for_completion = False
                                    self.completion_message_id = None

                                    time.sleep(0.5)

                        except Exception as e:
                            self.logger.error(
                                f"Error processing completion signal: {e}"
                            )

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in the completion signal listener: {e}")
                    if self.running:
                        time.sleep(1)
        finally:
            sock.close()
            self.logger.info("Completion signal listener thread stopped")

    def force_exit_handler(self):
        self.logger.info("\nTimeout reached, force exiting now")
        os._exit(1)

    def signal_handler(self, signal, frame):
        self.logger.info(f"Stop signal {signal} received, shutting down")
        self.running = False

        force_timer = threading.Timer(10.0, self.force_exit_handler)
        force_timer.daemon = True
        force_timer.start()

    # we generate unique id for temp files in RAMDISK
    def get_unique_filename(self, prefix="streamer_", suffix=".wav"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(
            self.voice_config.model.tmpfs_dir, f"{prefix}{timestamp}{suffix}"
        )

    def run_whisper(self, raw_audio: bytes):
        """
        runs faster-Whisper on raw pcm. audio must be 16kHz mono
        """
        try:
            text = ""
            segments, _ = self.batched_model.transcribe(
                raw_audio,
                beam_size=5,  # from Whisper_streaming recommandations
                language=self.voice_config.general.lang,
                batch_size=1,  # default 2 or 8, 1 for lower latency an limit vram usage
            )
            for segment in segments:
                text += " " + segment.text

            if not text:
                pass

            # self.logger.info(f"\033[92mWhisper transcription: {text}\033[0m")

            return text

        except Exception:
            # self.logger.error(f"Error in Whisper translation : {e}")
            return None

    def extract_command(self, text) -> str:
        if not text:
            return ""
        text = text.lower()
        keyword = self.voice_config.model.keyword.lower()
        pos = text.rfind(keyword)
        if pos == -1:
            return text
        command = text[pos + len(keyword) :].strip()
        command = re.sub(r"^[,\.;:!?]+\s*", "", command)

        return command

    def send_command_to_llm(self, command):
        """Send the extracted user text -after keyword- to the LLM via UDP"""
        if not command:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message_id = str(uuid.uuid4())
            self.completion_message_id = message_id
            self.waiting_for_completion = True
            message = f"[{message_id}]{command}".encode("utf-8")
            sock.sendto(
                message, ("127.0.0.1", self.voice_config.network.SEND_CHAT_PORT)
            )
            # max awaiting time of self.server_reset_delay seconds for an answer
            # after that we reset
            wait_time_start = time.time()
            while self.waiting_for_completion:
                if time.time() - wait_time_start > self.reset_server_delay:
                    self.logger.warning(
                        f"Timeout: no response from LLM after {
                            self.reset_server_delay
                        } seconds (ID {message_id})"
                    )
                    self.waiting_for_completion = False
                    break
                time.sleep(0.05)

            return False

        except Exception as e:
            self.logger.error(f"Error sending command : {e}")
            self.waiting_for_completion = False
            return False

        finally:
            sock.close()

    def text_to_speech(self, text) -> Union[bytes, None]:
        if not text:
            return None
        try:
            audio_file = self.get_unique_filename(prefix="tts_", suffix=".flac")
            if self.voice_config.general.lang == "fr":
                voice_folder = CURRENTDIR / self.voice_config.model.piper_fr_voice_path
            else:
                voice_folder = CURRENTDIR / self.voice_config.model.piper_en_voice_path

            cmd = [
                f"{self.voice_config.model.piper_path}",
                "--model",
                f"{voice_folder}",
                "--speaker",
                "1",
                "--cuda",
                "--output_file",
                audio_file,
            ]
            subprocess.run(
                cmd, input=text.encode(), capture_output=True, env=os.environ
            )
            # Resample to 16kHz because it is 22kHz
            rate, data = wav.read(audio_file)
            audio_data = resample(
                data, int(len(data) * self.voice_config.audio.sample_rate / rate)
            ).astype(np.int16)
            os.remove(audio_file)

            return audio_data.tobytes()

        except Exception as e:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            self.logger.error(f"Error in TTS engine : {e}")
            return None

    def send_audio_done(self, await_time):
        try:
            time.sleep(await_time)
            done_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            msg_id = self.completion_message_id if self.completion_message_id else "0"
            done_sock.sendto(
                f"__AUDIO_DONE__[{msg_id}]".encode("utf-8"),
                ("127.0.0.1", self.voice_config.network.SEND_CHAT_COMPLETION),
            )
            done_sock.close()
        except Exception as e:
            self.logger.error(f"Error sending audio completion signal: {e}")

        return True

    def command_handler(self):
        self.logger.info("Command processing thread started")

        while self.running:
            try:
                try:
                    command = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if not self.running:
                    break
                self.send_command_to_llm(command)
                self.command_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in command thread: {e}")
                if self.running:
                    time.sleep(1)
        self.logger.info("Command thread stopped")

    def clean_markdown(self, text: str) -> str:
        text = re.sub(r"BBMATHBB.*?BBMATHBB", " ", text, flags=re.DOTALL)
        text = re.sub(r"(\*\*|\*|_|`)(.+?)\1", r"\2", text)
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tts_response_handler(self):
        self.logger.info("Answer thread started")

        full_response = ""
        last_sent_index = 0

        while self.running:
            try:
                try:
                    response = self.response_queue.get(timeout=1.0)
                except queue.Empty:
                    if full_response and last_sent_index < len(full_response):
                        remaining = full_response[last_sent_index:]
                        if remaining.strip() and re.search(
                            r"[.!?]\s*$", remaining.strip()
                        ):
                            clean_text = self.clean_markdown(remaining)
                            audio_data = self.text_to_speech(clean_text)
                            if audio_data:
                                self.audio_handler.play(audio_data)
                                self.audio_handler.send_completion()
                            last_sent_index = len(full_response)
                    continue

                if not self.running:
                    break

                if response == "__END__":
                    if full_response and last_sent_index < len(full_response):
                        remaining = full_response[last_sent_index:]
                        if remaining.strip():
                            clean_text = self.clean_markdown(remaining)
                            audio_data = self.text_to_speech(clean_text)
                            if audio_data:
                                self.audio_handler.play(audio_data)
                                self.audio_handler.send_completion()
                    full_response = ""
                    last_sent_index = 0
                    self.send_audio_done(0.5)
                    self.response_queue.task_done()
                    continue

                full_response += response
                complete_segment = ""
                text_to_check = full_response[last_sent_index:]
                sentence_boundaries = list(re.finditer(r"[.!?]\s+", text_to_check))
                if sentence_boundaries:
                    last_boundary = sentence_boundaries[-1]
                    end_pos = last_boundary.end() + last_sent_index
                    complete_segment = full_response[last_sent_index:end_pos].strip()
                    last_sent_index = end_pos
                if complete_segment:
                    clean_segment = self.clean_markdown(complete_segment)
                    audio_data = self.text_to_speech(clean_segment)
                    if audio_data:
                        self.audio_handler.play(audio_data)

                self.response_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in answering thread : {e}")
                if self.running:
                    time.sleep(1)

        self.logger.info("Answer thread stopped")

    def llm_response_handler(self):
        self.logger.info("LLM listener thread started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)

        try:
            sock.bind(("127.0.0.1", self.voice_config.network.RCV_CHAT_CMD_PORT))

            while self.running:
                try:
                    data, addr = sock.recvfrom(
                        self.voice_config.network.rcv_buffer_size
                    )

                    if not self.running:
                        break

                    if data:
                        try:
                            response = data.decode("utf-8")
                            self.response_queue.put(response)
                        except UnicodeDecodeError:
                            self.logger.warning("Received invalid data from LLM")

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in LLM listener: {e}")
                    if self.running:
                        time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"ERROR: Failed to bind to port {
                    self.voice_config.network.RCV_CHAT_CMD_PORT
                }: {e}"
            )
        finally:
            sock.close()
            self.logger.info("LLM listener thread stopped")

    def _init_silero_vad(self):
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                onnx=False,
            )
            self.logger.info("Silero-vad initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Silero VAD: {e}")
            raise

    def _is_silero_speech(self, audio_data):
        try:
            if isinstance(audio_data, bytes):
                audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_chunk = audio_data.astype(np.float32) / 32768.0

            with torch.no_grad():
                vad_prob = (
                    self.silero_vad_model(
                        torch.from_numpy(audio_chunk),
                        16000,
                    )
                    .detach()
                    .item()
                )

            is_speech = vad_prob > (1 - self.silero_sensitivity)

            # if is_speech and not self.is_silero_speech_active:
            # self.logger.info("Silero VAD detected speech")
            # elif not is_speech and self.is_silero_speech_active:
            # self.logger.info("Silero VAD detected silence")

            self.is_silero_speech_active = is_speech
            return is_speech

        except Exception as e:
            self.logger.warn(f"Error in Silero VAD processing: {e}")
            return False

    def detect_wakeword(self, audio_data):
        """Detects wake word in audio data with
        pvporcupine"""
        try:
            pcm_data = np.frombuffer(audio_data, dtype=np.int16)
            result = self.porcupine.process(pcm_data)

            if result >= 0:
                self.logger.info("\033[91mWake word detected\033[0m")
                return True
            return False

        except Exception as e:
            self.logger.warn(f"Error in wake word detection: {e}")
            return False

    def start_recording(self):
        # self.logger.info("Starting recording")
        self.is_recording = True
        self.wakeword_detected = True
        self.detection_time = time.time()

        with self.lock:
            self.audio_buffer = bytearray()
            for chunk in self.pre_recording_buffer:
                self.audio_buffer.extend(chunk)

    def stop_recording(self):
        if not self.is_recording:
            return

        # self.logger.info("Stopping recording")
        self.is_recording = False

        if len(self.audio_buffer) > 0:
            audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16)
            self.whisper_queue.put(audio_data)
            self.audio_buffer = bytearray()
            self.audio_receiving_buffer = bytearray()
            self.pre_recording_buffer.clear()

        self.wakeword_detected = False
        self.speech_end_silence_start = 0
        self.is_silero_speech_active = False

    def reset_state(self):
        self.wakeword_detected = False
        self.is_recording = False
        self.detection_time = 0
        self.audio_buffer = bytearray()
        self.audio_receiving_buffer = bytearray()
        self.pre_recording_buffer.clear()
        self.speech_end_silence_start = 0
        self.is_silero_speech_active = False

    def process_audio_chunk(self, audio_chunk):
        if not self.is_recording:
            self.pre_recording_buffer.append(audio_chunk)

        if not self.wakeword_detected and not self.is_recording:
            if self.detect_wakeword(audio_chunk):
                self.audio_buffer = bytearray()
                self.start_recording()
                return

        if self.wakeword_detected and not self.is_recording:
            if time.time() - self.detection_time > self.wake_word_timeout:
                self.logger.warn("Wake word timeout, resetting state")
                self.reset_state()
                return

        if self.is_recording:
            with self.lock:
                self.audio_buffer.extend(audio_chunk)

            is_speech = self._is_silero_speech(audio_chunk)

            if not is_speech:
                if self.speech_end_silence_start == 0:
                    self.speech_end_silence_start = time.time()

                elif (
                    time.time() - self.speech_end_silence_start
                    >= self.post_speech_silence_duration
                ):
                    self.stop_recording()
            else:
                self.speech_end_silence_start = 0

    def whisper_handler(self):
        self.logger.info("Whisper thread started")

        while self.running:
            try:
                try:
                    audio_data = self.whisper_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if not self.running:
                    break
                current_time = time.time()
                time_since_last_audio = current_time - self.last_audio_sent_time

                if time_since_last_audio < self.ignore_own_audio_seconds:
                    # self.logger.info("Skipping audio processing, likely a loopback")
                    self.whisper_queue.task_done()
                    continue

                # we normalize audio for Whisper
                normalized_data = audio_data.astype(np.float32) / 32768.0
                # self.logger.info("Transcribing audio with Whisper")
                transcription = self.run_whisper(normalized_data)

                if (
                    transcription
                    and len(transcription.split())
                    >= self.voice_config.audio.min_cmd_length
                ):
                    command = self.extract_command(transcription)
                    # self.logger.info(f"Command detected: {command}")
                    self.command_queue.put(command)
                    self.last_audio_sent_time = current_time
                    self.recent_transcriptions = [transcription]

                self.whisper_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in Whisper thread: {e}")
                if self.running:
                    time.sleep(0.5)

        self.logger.info("Whisper thread stopped")

    def send_test_command(self):
        if self.voice_config.general.lang == "fr":
            test_command = "Bonjour. Comment ça va aujourd'hui?"
            self.logger.info(f"\nSending test command to LLM : '{test_command}'")
            self.command_queue.put(test_command)
        else:
            test_command = "Hello. How are you today ?"
            self.logger.info(f"\nSending test command to LLM : '{test_command}'")
            self.command_queue.put(test_command)

    def get_status(self):
        return {
            "running": self.running,
            "recording": self.is_recording,
            "command_queue": self.command_queue.qsize(),
            "response_queue": self.response_queue.qsize(),
            "whisper_queue": self.whisper_queue.qsize(),
        }


class VoiceEngine:
    def __init__(self):
        self.server = None
        self.running = False
        self.audio_handler = None

    def start(self):
        if self.running:
            return
        self.audio_handler = LocalAudio()
        time.sleep(2.0)
        self.server = LocalDispatcher(audio_handler=self.audio_handler)
        self.server.start()
        self.running = True

    def stop(self):
        if self.server:
            self.server.stop()
        if self.audio_handler:
            self.audio_handler.stop()
        self.running = False

    def send_message(self, text):
        if self.server:
            self.server.command_queue.put(text)

    def get_status(self):
        if self.server:
            return {
                "running": self.server.running,
                "recording": self.server.is_recording,
                "command_queue": self.server.command_queue.qsize(),
                "response_queue": self.server.response_queue.qsize(),
                "whisper_queue": self.server.whisper_queue.qsize(),
                "audio_playing": self.audio_handler.stream_on
                if self.audio_handler
                else False,
            }
        return {"running": False}
