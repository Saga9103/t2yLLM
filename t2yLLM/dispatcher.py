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

# that import is only really useful if using mms-tts
from num2words import num2words
from datetime import datetime

# MATH:
import numpy as np
import torch

# SOUND
import soundfile as sf
import wave

# sound resample
import scipy.io.wavfile as wav
from scipy.signal import resample

# near realtime STT with WhisperLive
from faster_whisper import WhisperModel, BatchedInferencePipeline

# wake word detection
import pvporcupine

# UDP
from hmacauth import HMACAuth

# piper
CURRENTDIR = Path(__file__).resolve().parent
PIPERROOT = CURRENTDIR / "config" / "piper"

os.environ["LD_LIBRARY_PATH"] = f"{PIPERROOT}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["ESPEAKNG_DATA_PATH"] = str(PIPERROOT / "espeak-ng-data")

# add parent DIR for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class VoiceServer:
    def __init__(self):
        # from config
        self.voice_config = Loader().loadWhispConfig()  # load the .yaml
        porcupine_dir = Path("./porcupine")
        porcupine_dir.mkdir(exist_ok=True)
        self.logger = self.set_logger()

        # faster-whisper
        self.fast_whisper_model = WhisperModel(
            "quantized", device="cuda", compute_type="int8_float16"
        )  # Faster-Whisper is incredibly fast with low latency but on a 16GB GPU
        # it becomes a hard limit to handle with a LLM on top
        # model here "quantized" which is custom should be in config or at least a fallback"
        self.batched_model = BatchedInferencePipeline(model=self.fast_whisper_model)

        self.hmac_auth = HMACAuth()

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

        self.client_address = self.voice_config.network.RPI_IP

        self.last_detection_time = 0
        # comm handlers
        # command_queue to send to the LLM
        self.command_queue = queue.Queue()
        # whisper_queue for processing whisper transcriptions
        self.whisper_queue = queue.Queue()
        # response_queue for sending audio to the client (here Raspberry+respeaker lite)
        self.response_queue = queue.Queue()

        self.recent_transcriptions = []  # Keeping track of recent transcriptions
        self.last_audio_sent_time = 0  # Time when audio was last sent to client
        self.waiting_for_completion = False
        self.completion_message_id = None
        # if no activity, the server will reset its state
        # i had problems with it hanging indefinitly
        # when I reset the LLM backend server and becoming
        # unresponsive
        self.reset_server_delay = int(
            self.voice_config.network.server_reset_delay
        )  # in seconds
        self.msg_id = None
        # had a problem with loopbacks
        self.ignore_own_audio_seconds = self.voice_config.audio.ignore_own
        # in anycase, it cant be longer than keyword silence threashold
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
        self.porcupine_frame_length = self.porcupine.frame_length
        self.keyword_detected = False
        self.keyword_hang = 4.0  # in seconds
        self.detection_time = 0.0  # in seconds
        self.count = 0

        self._init_silero_vad()

    def set_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger("WhisperServer")

    def start(self):
        if self.running:
            return

        self.running = True
        self.threads = []
        self.client_address = self.voice_config.network.RPI_IP
        self.active_stream = False
        self.last_detection_time = time.time()
        self.last_audio_sent_time = time.time() - 180
        self.waiting_for_completion = False
        self.completion_message_id = None

        os.makedirs(self.voice_config.model.tmpfs_dir, exist_ok=True)

        audio_thread = threading.Thread(
            target=self.audio_server, daemon=True, name="AudioServer"
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

        tts_status_thread = threading.Thread(
            target=self.tts_status_listener, daemon=True, name="TtsStatusListener"
        )
        tts_status_thread.start()
        self.threads.append(tts_status_thread)

        self.logger.info("All threads are UP")
        self.logger.info(
            f"Listening for audio on port {self.voice_config.network.LISTEN_RPI_PORT}"
        )
        self.logger.info(
            f"\033[32mKeyword : '{self.voice_config.model.keyword}'\033[0m"
        )
        self.logger.info(
            f"\033[32mTTS engine: {self.voice_config.model.tts_engine}\033[0m"
        )
        self.logger.info(
            f"Listening for LLM on port {self.voice_config.network.RCV_CHAT_CMD_PORT}"
        )
        self.logger.info(
            "\033[33mEnter: \n'exit' to stop \n'status' to see the queue status \n'test' to send test command\033[0m"
        )

    def stop(self):
        self.logger.info("Stopping VoiceServer")
        self.running = False

        if hasattr(self, "threads"):
            for thread in self.threads:
                thread.join(timeout=2.0)

        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("VoiceServer stopped")

    def completion_signal_listener(self):
        self.logger.info("Completion signal listener thread started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)

        try:
            sock.bind(
                (
                    self.voice_config.network.LISTEN_IP,
                    self.voice_config.network.RCV_END_SIGNAL,
                )
            )  # LLM is sending end signal through this port
            self.logger.info(
                f"Listening for completion signal on port {
                    self.voice_config.network.RCV_END_SIGNAL
                }"
            )

            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)

                    if not self.running:
                        break

                    if data:
                        message_data = self.hmac_auth.unpack_message(data)
                        if message_data is None:
                            self.logger.error("HMAC verification failed")
                            continue
                        try:
                            # signal = data.decode("utf-8")
                            if message_data.get("type") == "completion":
                                self.msg_id = message_data.get("message_id")
                                # self.logger.info(f"Received completion signal: {signal}")

                                # if signal.startswith("__DONE__"):
                                # we extract the message ID
                                # if "[" in signal and "]" in signal:
                                #    self.msg_id = signal[
                                #        signal.find("[") + 1 : signal.find("]")
                                #   ]

                                # self.logger.info(
                                #    f"msg_id={self.msg_id}, waiting for={
                                #        self.completion_message_id
                                #    }"
                                # )
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

                                    self.logger.info(
                                        "Completion received, resuming active streaming"
                                    )
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

    # useless with piper-tts was mandatory with mms-tts
    def num_2_text(self, text):  # slow
        #  matching integers or floats
        def convert(match):
            num = match.group()
            try:
                # replacing comas by a . for float
                num = num.replace(",", ".")
                return num2words(float(num), lang=self.voice_config.general.lang)
            except Exception:
                return num  # else nothing

        return re.sub(r"\d+([.,]\d+)?", convert, text)

    def convert_raw_to_wav(self, raw_data, output_file):
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)  # Mono -> to add to config
            wf.setsampwidth(2)  # 16-bit ->to add to config
            wf.setframerate(self.voice_config.audio.sample_rate)
            wf.writeframes(raw_data)
        return output_file

    # this is now preferred to send audio over UDP because it is
    # probably as fast and the size is reduced
    def convert_raw_to_flac(self, raw_data, output_file):
        sample_rate = self.voice_config.audio.sample_rate
        audio_array = np.frombuffer(raw_data, dtype=np.int16)
        sf.write(output_file, audio_array, samplerate=sample_rate, format="FLAC")
        return output_file

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
                # word_timestamps=True,
                # hallucination_silence_threshold=0.5, #it needs wordtimestamps enabled
                # When word_timestamps is True, skip silent periods longer than this threshold
                # (in seconds) when a possible hallucination is detected. set as None.
                # vad_filter=True,
                # vad_parameters=dict(min_silence_duration_ms=800),
                # since we use silero on the buffer, it prevents low noise and silence from
                # reaching whisper and we dont have that hallucination problems anymore
                batch_size=1,  # default 2 or 8, 1 for lower latency an limit vram usage
            )
            for segment in segments:
                text += " " + segment.text

            if not text:
                pass

            self.logger.info(f"\033[92mWhisper transcription: {text}\033[0m")

            return text

        except Exception as e:
            self.logger.error(f"Error in Whisper translation : {e}")
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
            data = {
                "message_id": message_id,
                "command": command,
                "text": f"[{message_id}]{command}",
            }
            message = self.hmac_auth.pack_message(data)
            # message = f"[{message_id}]{command}".encode("utf-8")
            sock.sendto(
                message,
                (
                    self.voice_config.network.CHAT_ADDR,
                    self.voice_config.network.SEND_CHAT_PORT,
                ),
            )
            self.logger.info(
                f"\033[92m\n<> Sent to LLM <> ({message_id}): {command}\n on port {
                    self.voice_config.network.SEND_CHAT_PORT
                }\n\033[0m"
            )
            # max awaiting time of self.server_reset_delay seconds for an answer
            # after that we reset
            wait_time_start = time.time()
            while self.waiting_for_completion:
                if time.time() - wait_time_start > self.reset_server_delay:
                    self.logger.warning(
                        f"Timeout: no response from LLM after {
                            self.reset_server_delay
                        } seconds (ID {message_id})"  # Et ici
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
            # text = self.num_2_text(text)  # not really useful with piper but with mms-tts yes
            # Generate a unique temporary file name, stored in ramdisk
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
            # and all fails
            rate, data = wav.read(audio_file)
            audio_data = resample(
                data, int(len(data) * self.voice_config.audio.sample_rate / rate)
            ).astype(np.int16)

            # Clean up
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
            data = {
                "type": "audio_done",
                "message_id": msg_id,
                "text": f"__AUDIO_DONE__[{msg_id}]",
            }
            message = self.hmac_auth.pack_message(data)
            done_sock.sendto(
                message,
                ("127.0.0.1", self.voice_config.network.SEND_CHAT_COMPLETION),
            )
            """
            done_sock.sendto(
                f"__AUDIO_DONE__[{msg_id}]".encode("utf-8"),
                (
                    self.voice_config.network.CHAT_ADDR,
                    self.voice_config.network.SEND_CHAT_COMPLETION,
                ),
            )
            self.logger.info(
                f"Sent completion signal after waiting for {
                    await_time:.2f
                }s ,message ID {msg_id}"
            )
            """
            done_sock.close()
        except Exception as e:
            self.logger.error(f"Error sending audio completion signal: {e}")

        return True

    def send_audio_to_client(self, audio_data):
        """Send audio data to the client (Raspberry Pi)"""
        if not audio_data:
            self.logger.info("No audio data to send")
            return False
        if not self.client_address:
            self.logger.error("Client address wasnt properly defined")
            return False
        else:
            client_addr = (
                self.client_address,
                self.voice_config.network.SEND_RPI_PORT,
            )

        try:
            self.last_audio_sent_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)
            max_pack = self.voice_config.network.MAX_UDP_SIZE - 100
            self.logger.info(
                f"Sending audio answer ({len(audio_data)} bytes) to {client_addr[0]}:{
                    client_addr[1]
                } .."
            )
            self.logger.info(
                f"Sending {len(audio_data)} bytes of audio to {client_addr[0]}:{
                    client_addr[1]
                }"
            )

            seq = 0
            total_packets = (len(audio_data) + max_pack - 1) // max_pack

            for i in range(0, len(audio_data), max_pack):
                if not self.running:
                    self.logger.info("Stopping audio")
                    break

                seg = audio_data[i : i + max_pack]
                packet_data = {
                    "type": "audio_packet",
                    "seq": seq,
                    "total": total_packets,
                    "audio": seg,
                }
                message = self.hmac_auth.pack_message(packet_data)
                sock.sendto(message, client_addr)
                seq += 1

                time.sleep(0.002)

            if self.running:
                end_data = {"type": "end_of_audio"}
                message = self.hmac_auth.pack_message(end_data)
                sock.sendto(message, client_addr)
            self.logger.info(
                f"Audio sent to client {client_addr[0]}:{client_addr[1]} ({
                    len(audio_data)
                } bytes in {total_packets} packets)"
            )

            # We approximate audio duration with a margin (16-bit mono at 16kHz)
            audio_duration = len(audio_data) / (2 * self.voice_config.audio.sample_rate)

            # wait until audio has been played before sending end signal
            # plus some margin
            await_time = audio_duration + 0.5
            self.send_audio_done(await_time)

        except Exception as e:
            self.logger.error(f"Error sending audio to client: {e}")
            return False
        finally:
            sock.close()

    def tts_status_listener(self):
        self.logger.info("TTS Processing started")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind(
                (
                    self.voice_config.network.LISTEN_IP,
                    self.voice_config.network.STATUS_REQUEST_PORT,
                )
            )
            self.logger.info("Listening for TTS status requests on port 5005")

            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)

                    if not self.running:
                        break

                    if data:
                        message_data = self.hmac_auth.unpack_message(data)
                        if message_data is None:
                            self.logger.error("HMAC verification failed for TTS")
                            continue
                        try:
                            if message_data.get("type") == "tts_status_request":
                                duration = getattr(
                                    sys.modules[__name__], "actual_audio_duration", 0
                                )
                                response_data = {
                                    "type": "tts_duration",
                                    "duration": duration,
                                }
                                response = self.hmac_auth.pack_message(response_data)
                                sock.sendto(response, addr)
                                self.logger.info(
                                    f"Sent TTS duration {duration:.2f}s to {addr[0]}"
                                )
                        except Exception as e:
                            self.logger.error(
                                f"Error processing TTS status request: {e}"
                            )

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in TTS status listener: {e}")
                    if self.running:
                        time.sleep(1)
        finally:
            sock.close()
            self.logger.info("TTS status thread stopped")

    def filter_audio(self, audio_data):
        # Convert bytearray to numpy array of 16-bit samples
        audio_array = np.frombuffer(
            audio_data, dtype=np.int16
        ).copy()  # Add .copy() to make a writeable array

        threshold = 2000  # silence thresh
        audio_array[np.abs(audio_array) < threshold] = 0
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        if rms < threshold:  # self.voice_config.audio.min_audio_level:
            return None  # No significant audio, don't process
        else:
            return audio_array.tobytes()

    def calculate_audio_stats(self, audio_data):
        if not audio_data or len(audio_data) < 100:
            return {"rms": 0, "peak": 0, "zero_crossings": 0}
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        peak = np.max(np.abs(audio_array))
        # Calculate zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array)))) / len(
            audio_array
        )

        return {"rms": rms, "peak": peak, "zero_crossings": zero_crossings}

    def command_handler(self):
        self.logger.info("Command processing thread started")

        while self.running:
            try:
                try:
                    command = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    continue  # No items in queue, just continue

                if not self.running:
                    break

                self.send_command_to_llm(command)
                self.command_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in command thread: {e}")
                if self.running:
                    time.sleep(1)

        self.logger.info("Command thread stopped")

    def tts_response_handler(self):
        self.logger.info("Answer thread started")

        full_response = ""
        last_sent_index = 0

        def clean_markdown(text):
            text = re.sub(r"BBMATHBB.*?BBMATHBB", " ", text, flags=re.DOTALL)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"\*(.+?)\*", r"\1", text)
            text = re.sub(r"\_(.+?)\_", r"\1", text)
            text = re.sub(r"\`(.+?)\`", r"\1", text)
            text = re.sub(r"\#+ ", "", text)
            text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text

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
                            clean_text = clean_markdown(remaining)
                            audio_data = self.text_to_speech(clean_text)
                            if audio_data:
                                self.send_audio_to_client(audio_data)
                            last_sent_index = len(full_response)
                    continue

                if not self.running:
                    break

                if response == "__END__":
                    if full_response and last_sent_index < len(full_response):
                        remaining = full_response[last_sent_index:]
                        if remaining.strip():
                            clean_text = clean_markdown(remaining)
                            audio_data = self.text_to_speech(clean_text)
                            if audio_data:
                                self.send_audio_to_client(audio_data)

                    full_response = ""
                    last_sent_index = 0

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
                    clean_segment = clean_markdown(complete_segment)
                    audio_data = self.text_to_speech(clean_segment)
                    if audio_data:
                        self.send_audio_to_client(audio_data)

                self.response_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in answering thread : {e}")
                if self.running:
                    time.sleep(1)

        self.logger.info("Answer thread stopped")

    def split_text_into_segments(self, text):
        if not text:
            return []

        segments = re.split(r"(?<=[.!?])\s+", text)
        result = []
        current = ""

        for segment in segments:
            if not segment.strip():
                continue
            if len(segment) > 150:
                subsegments = re.split(r"(?<=[,;:])\s+", segment)
                for subsegment in subsegments:
                    subsegment = subsegment.strip()
                    if not subsegment:
                        continue
                    if len(current) + len(subsegment) + 1 < 150:
                        current += " " + subsegment if current else subsegment
                    else:
                        if current:
                            result.append(current.strip())
                        current = subsegment
            elif len(current) + len(segment) + 1 < 150:
                current += " " + segment if current else segment
            else:
                if current:
                    result.append(current.strip())
                current = segment
        if current:
            result.append(current.strip())

        return result

    def compress_to_flac(self, audio_data):
        if not audio_data:
            return None

        try:
            output_file = self.get_unique_filename(prefix="stream_", suffix=".flac")
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return audio_data

            sf.write(
                output_file,
                audio_array,
                samplerate=self.voice_config.audio.sample_rate,
                format="FLAC",
                subtype="PCM_16",  # Forcer en 16-bits
            )
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                self.logger.error(f"Error creating .flac {output_file}")
                return audio_data

            with open(output_file, "rb") as f:
                flac_data = f.read()
            if not flac_data.startswith(b"fLaC"):
                self.logger.error("invalid flac file")
                os.remove(output_file)
                return audio_data

            os.remove(output_file)

            return flac_data

        except Exception:
            return audio_data

    def send_audio_segment(self, audio_data, segment_idx, total_segments, is_last):
        if not audio_data or not self.client_address:
            self.logger.error("Error in data or client address")
            return False

        client_addr = (self.client_address, self.voice_config.network.SEND_RPI_PORT)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)
            max_pack = self.voice_config.network.MAX_UDP_SIZE
            if isinstance(audio_data, bytearray):
                audio_data = bytes(audio_data)

            seq = 0
            total_packets = (len(audio_data) + max_pack - 1) // max_pack

            for i in range(0, len(audio_data), max_pack):
                if not self.running:
                    break

                seg = audio_data[i : i + max_pack]
                segment_id = f"{int(time.time())}-{segment_idx}"
                header = f"{segment_id}/{segment_idx}/{total_segments}/{seq}/{total_packets}:".encode(
                    "ascii"
                )
                packet = header + seg
                sock.sendto(packet, client_addr)
                seq += 1
                # Très petit délai pour éviter la congestion réseau
                time.sleep(0.001)

            time.sleep(0.02)
            sock.sendto(f"__SEGMENT_COMPLETE__{segment_id}".encode(), client_addr)

            self.logger.info(
                f"audio segment {segment_idx + 1}/{total_segments} sent ({
                    len(audio_data)
                } octets in {total_packets} packets)"
            )
            if is_last:
                time.sleep(0.05)
                sock.sendto(b"__END_OF_AUDIO__", client_addr)

            return True

        except Exception as e:
            self.logger.error(f"Error sending audio segment : {e}")
            return False
        finally:
            sock.close()

    def send_end_marker(self):
        if not self.client_address:
            return False

        client_addr = (
            self.client_address,
            self.voice_config.network.SEND_RPI_PORT,
        )

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"__END_OF_AUDIO__", client_addr)
            sock.close()
        except Exception as e:
            self.logger.error(f"Error sending end marker: {e}")

    def llm_response_handler(self):
        self.logger.info("LLM listener thread started")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)

        try:
            sock.bind(
                (
                    self.voice_config.network.LISTEN_IP,
                    self.voice_config.network.RCV_CHAT_CMD_PORT,
                )
            )
            self.logger.info(
                f"Listening for LLM on port {
                    self.voice_config.network.RCV_CHAT_CMD_PORT
                }"
            )
            self.logger.info(
                f"Listening for LLM on port {
                    self.voice_config.network.RCV_CHAT_CMD_PORT
                }"
            )

            while self.running:
                try:
                    data, addr = sock.recvfrom(
                        self.voice_config.network.rcv_buffer_size
                    )

                    if not self.running:
                        break

                    if data:
                        response_data = self.hmac_auth.unpack_message(data)
                        if response_data is None:
                            self.logger.error("HMAC verification failed")
                            continue
                        try:
                            # maybe we should have some
                            # response = data.decode("utf-8")
                            response = response_data.get("text", "")
                            # other checks
                            self.logger.info(f"Receiving from LLM : {response[:50]}...")
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
            self.logger.info(
                f"ERROR: Failed to bind to port {
                    self.voice_config.network.RCV_CHAT_CMD_PORT
                }: {e}"
            )
        finally:
            sock.close()
            self.logger.info("LLM listener thread stopped")

    def _init_silero_vad(self):
        """Initializes the Silero VAD model"""
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

    # thanks to realtimeSTT because their solution is incredible
    # and after a lot of fails I found their perfectly working code
    # for processing a buffer with silero in a straming fashion
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

            if is_speech and not self.is_silero_speech_active:
                self.logger.info("Silero VAD detected speech")
            elif not is_speech and self.is_silero_speech_active:
                self.logger.info("Silero VAD detected silence")

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
                self.logger.info("Wake word detected")
                return True
            return False

        except Exception as e:
            self.logger.warn(f"Error in wake word detection: {e}")
            return False

    def start_recording(self):
        """Start recording audio after wake word detection"""
        self.logger.info("Starting recording")
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

        self.logger.info("Stopping recording")
        self.is_recording = False

        if len(self.audio_buffer) > 0:
            audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16)
            self.whisper_queue.put(audio_data)
            self.audio_buffer = bytearray()
            self.audio_receiving_buffer = bytearray()  # Add this line
            self.pre_recording_buffer.clear()  # Add this line

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
        """Process a chunk of audio data"""
        if not self.is_recording:
            self.pre_recording_buffer.append(audio_chunk)

        if not self.wakeword_detected and not self.is_recording:
            if self.detect_wakeword(audio_chunk):
                self.audio_buffer = bytearray()
                self.start_recording()
                return

        if self.wakeword_detected and not self.is_recording:
            if time.time() - self.detection_time > self.wake_word_timeout:
                self.logger.info("Wake word timeout, resetting state")
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
                    self.logger.info(
                        f"Silence detected for {
                            self.post_speech_silence_duration
                        }s, stopping recording"
                    )
                    self.stop_recording()
            else:
                self.speech_end_silence_start = 0

    def audio_server(self):
        self.logger.info("Starting audio server")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)

        try:
            sock.bind(
                (
                    self.voice_config.network.LISTEN_IP,
                    self.voice_config.network.LISTEN_RPI_PORT,
                )
            )
            self.logger.info(
                f"Listening for audio on {self.voice_config.network.LISTEN_IP}:{
                    self.voice_config.network.LISTEN_RPI_PORT
                }"
            )

            while self.running:
                try:
                    data, addr = sock.recvfrom(4096)

                    if not self.running:
                        break

                    if addr[0] not in self.voice_config.network.AUTHORIZED_IPS:
                        self.logger.warn(
                            f"Rejected audio from unauthorized IP: {addr[0]}"
                        )
                        continue

                    message_data = self.hmac_auth.unpack_message(data)
                    if message_data is None:
                        self.logger.error("HMAC verification failed for audio")
                        continue

                    audio_chunk = message_data.get("audio")
                    if not audio_chunk:
                        if message_data.get("type") == "end_command":
                            self.logger.info("Received end of command marker")
                            if self.is_recording:
                                self.stop_recording()
                            continue
                        continue

                    self.process_audio_chunk(data)

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in audio server: {e}")
                    if self.running:
                        time.sleep(1)

        except Exception as e:
            self.logger.error(
                f"Error binding to port {self.voice_config.network.LISTEN_RPI_PORT}: {
                    e
                }"
            )
        finally:
            sock.close()
            self.logger.info("Audio server stopped")

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
                    self.logger.info("Skipping audio processing, likely a loopback")
                    self.whisper_queue.task_done()
                    continue

                # we normalize audio for Whisper
                normalized_data = audio_data.astype(np.float32) / 32768.0
                self.logger.info("Transcribing audio with Whisper")
                transcription = self.run_whisper(normalized_data)

                if len(transcription.split()) >= self.voice_config.audio.min_cmd_length:
                    command = self.extract_command(transcription)
                    self.logger.info(f"Command detected: {command}")
                    self.command_queue.put(command)
                    self.last_audio_sent_time = current_time
                    self.recent_transcriptions = [transcription]
                else:
                    self.logger.warn(
                        f"Transcription too short, ignoring: {transcription}"
                    )

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
            self.send_command_to_llm(test_command)
        else:
            test_command = "Hello. How are you today ?"
            self.logger.info(f"\nSending test command to LLM : '{test_command}'")
            self.send_command_to_llm(test_command)


class VoiceEngine:
    def __init__(self):
        self.server = None
        self.running = False

    def start(self):
        if self.running:
            return

        self.server = VoiceServer()
        self.server.start()
        self.running = True

    def stop(self):
        if self.server:
            self.server.stop()
        self.running = False

    def send_message(self, text):
        if self.server:
            self.server.send_command_to_llm(text)

    def get_status(self):
        if self.server:
            return {
                "running": self.server.running,
                "recording": self.server.is_recording,
                "command_queue": self.server.command_queue.qsize(),
                "response_queue": self.server.response_queue.qsize(),
                "whisper_queue": self.server.whisper_queue.qsize(),
            }
        return {"running": False}
