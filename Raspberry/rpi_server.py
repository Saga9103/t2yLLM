import io
import socket
import time
import threading
import queue
import pyaudio
import audioop
import signal
import sys
import wave
import numpy as np
import soundfile as sf
from piConfigLoader import Loader
from pathlib import Path
import secrets
import hmac
import hashlib
from typing import Dict
import json
import os

CONFIG = Loader().load_pi_config()


class HMACAuth:
    def __init__(self, secret_key: bytes = None):
        env_key = os.environ.get("T2YLLM_HMAC_KEY")
        if env_key:
            try:
                self.secret_key = bytes.fromhex(env_key)
            except ValueError:
                print("Warning: Invalid HMAC key in environment variable")
                self.secret_key = secret_key or self.load_or_create_key()
        else:
            self.secret_key = secret_key or self.load_or_create_key()

    @staticmethod
    def load_or_create_key() -> bytes:
        key_file = Path.home() / ".t2yllm" / "hmac.key"
        key_file.parent.mkdir(exist_ok=True, mode=0o700)

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = secrets.token_bytes(32)
            with open(key_file, "wb") as f:
                f.write(key)
            key_file.chmod(0o600)
            return key

    def create_signature(self, message: bytes) -> bytes:
        h = hmac.new(self.secret_key, message, hashlib.sha256)
        return h.digest()

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        expected = self.create_signature(message)
        return hmac.compare_digest(expected, signature)

    def pack_message(self, data: Dict) -> bytes:
        data["timestamp"] = time.time()
        message = json.dumps(data).encode("utf-8")
        signature = self.create_signature(message)
        return signature + b"||" + message

    def unpack_message(self, packed_msg: bytes, max_age: int = 300):
        try:
            parts = packed_msg.split(b"||", 1)
            if len(parts) != 2:
                return None
            signature, message = parts

            if not self.verify_signature(message, signature):
                return None

            data = json.loads(message.decode("utf-8"))

            if "timestamp" in data:
                age = time.time() - data["timestamp"]
                if age > max_age:
                    return None

            return data
        except Exception:
            return None


class AudioStreamer:
    def __init__(self):
        self.config = {
            "server_ip": CONFIG.network.server_ip,
            "server_port": CONFIG.network.server_port,
            "receive_port": CONFIG.network.receive_port,
            "sample_rate": CONFIG.network.sample_rate,
            "chunk_size": 512,
            "format": pyaudio.paInt16,
            "channels": CONFIG.network.channels,
            "output_volume": CONFIG.network.output_volume,
            "buffer_ms": CONFIG.network.buffer_ms,
            "output_device_index": None,
            "input_device_index": None,
            "max_udp_packet": CONFIG.network.max_udp_packet,
        }

        self.running = True
        self.audio = pyaudio.PyAudio()
        self.hmac_auth = HMACAuth()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.input_stream = None
        self.output_stream = None
        self.client_socket = None
        self.server_socket = None
        self.received_audio_buffer = bytearray()
        self.fragments = {}
        self.current_message_id = None
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
        self.padding = b"\x00" * 2 * self.config["chunk_size"]

        self.threads = []

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.stop()
        sys.exit(0)

    def setup_audio_socket(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True
        except Exception:
            return False

    def setup_server_socket(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind(("0.0.0.0", self.config["receive_port"]))
            self.server_socket.settimeout(0.5)
            return True
        except Exception:
            return False

    def audio_sender(self):
        if not self.setup_audio_socket():
            return

        buffer_chunks = max(
            1,
            int(
                (self.config["buffer_ms"] / 1000)
                * self.config["sample_rate"]
                / self.config["chunk_size"]
            ),
        )

        header = {
            "type": "audio_header",
            "sample_rate": self.config["sample_rate"],
            "channels": self.config["channels"],
            "format": "int16",
            "chunk_size": self.config["chunk_size"],
        }

        try:
            message = self.hmac_auth.pack_message(header)
            # header_str = str(header).encode("utf-8")
            self.client_socket.sendto(
                message, (self.config["server_ip"], self.config["server_port"])
            )
        except Exception as e:
            print(f"Error sending header: {e}")

        audio_buffer = []

        silent_chunks = 0
        is_speaking = False
        end_threshold = int(
            1.5 * self.config["sample_rate"] / self.config["chunk_size"]
        )

        while self.running:
            try:
                try:
                    audio_data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                audio_buffer.append(audio_data)
                rms = audioop.rms(audio_data, 2)
                if rms < 400:  # Seuil pour le silence
                    silent_chunks += 1
                    if silent_chunks > end_threshold and is_speaking:
                        is_speaking = False
                        audio_buffer = []
                        end_data = {"type": "end_command"}
                        message = self.hmac_auth.pack_message(end_data)
                        self.client_socket.sendto(
                            message,
                            (self.config["server_ip"], self.config["server_port"]),
                        )
                else:
                    silent_chunks = 0
                    is_speaking = True

                if len(audio_buffer) >= buffer_chunks:
                    combined_data = b"".join(audio_buffer)
                    audio_buffer = []
                    packet_data = {"type": "audio_data", "audio": combined_data}
                    message = self.hmac_auth.pack_message(packet_data)

                    for i in range(
                        0, len(combined_data), self.config["max_udp_packet"]
                    ):
                        packet = combined_data[i : i + self.config["max_udp_packet"]]
                        try:
                            self.client_socket.sendto(
                                packet,
                                (self.config["server_ip"], self.config["server_port"]),
                            )
                        except Exception:
                            break
                        time.sleep(0.002)

            except Exception:
                time.sleep(0.5)

    def audio_receiver(self):
        if not self.setup_server_socket():
            return

        self.last_fragment_time = time.time()

        while self.running:
            try:
                try:
                    data, addr = self.server_socket.recvfrom(8192)

                    if data:
                        self.last_fragment_time = time.time()
                        message_data = self.hmac_auth.unpack_message(data)
                        if message_data is None:
                            print("HMAC verification failed")
                            continue
                        msg_type = message_data.get("type")

                    if msg_type == "end_of_audio":
                        if (
                            self.current_message_id
                            and self.current_message_id in self.fragments
                        ):
                            self.process_frags()

                            if self.received_audio_buffer:
                                self.output_queue.put(bytes(self.received_audio_buffer))
                                self.received_audio_buffer = bytearray()

                            self.output_queue.put(
                                b"__END_OF_AUDIO__"
                            )  # now we need to forward it
                            self.fragments = {}
                            self.current_message_id = None
                            continue

                    elif msg_type == "audio_packet":
                        seq = message_data.get("seq")
                        total = message_data.get("total")
                        audio_data = message_data.get("audio")

                        if audio_data:
                            if self.current_message_id is None:
                                self.current_message_id = f"msg_{time.time()}"

                            if self.current_message_id not in self.fragments:
                                self.fragments[self.current_message_id] = {}

                            self.fragments[self.current_message_id][seq] = audio_data

                            if len(self.fragments[self.current_message_id]) == total:
                                self.process_frags()
                                del self.fragments[self.current_message_id]
                                self.current_message_id = None

                    current_time = time.time()  # now we check if a timeout was reached
                    # since the last received fragment (in case __END_OF_AUDIO__ was not
                    # received or sent properly)
                    if len(self.received_audio_buffer) > 0:
                        if (
                            current_time - self.last_fragment_time
                        ) > self.fragment_timeout:
                            self.output_queue.put(bytes(self.received_audio_buffer))
                            self.received_audio_buffer = bytearray()
                            self.output_queue.put(b"__TIMEOUT__")
                            self.last_fragment_time = current_time

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error in receiving thread : {e}")
                    time.sleep(0.5)

            except Exception as e:
                print(f"Error in receiving thread : {e}")
                time.sleep(0.5)

    def process_packets(self, data):
        header_end = data.find(b":")
        if header_end > 0 and header_end < 30:
            try:
                header = data[:header_end].decode("ascii")
                content = data[header_end + 1 :]

                if header.count("/") == 3:
                    seg_idx, total_segments, seq_num, total_chunks = map(
                        int, header.split("/")
                    )

                    segment_msg_id = f"seg_{seg_idx}"

                    if segment_msg_id not in self.fragments:
                        self.fragments[segment_msg_id] = {}

                    self.fragments[segment_msg_id][int(seq_num)] = content

                    if len(self.fragments[segment_msg_id]) == int(total_chunks):
                        reassembled = bytearray()
                        for i in range(int(total_chunks)):
                            if i in self.fragments[segment_msg_id]:
                                reassembled.extend(self.fragments[segment_msg_id][i])

                        segment_data = bytes(reassembled)

                        try:
                            if segment_data.startswith(b"fLaC"):
                                debug_file = f"/tmp/segment_{seg_idx}.flac"
                                with open(debug_file, "wb") as f:
                                    f.write(segment_data)

                                self.process_flac_fst(segment_data)
                            elif (
                                segment_data.startswith(b"RIFF")
                                and b"WAVE" in segment_data[:12]
                            ):
                                self.wav_data(segment_data)
                            else:
                                self.output_queue.put(segment_data)
                        except Exception as e:
                            print(f"Error processing audio segment: {e}")
                        del self.fragments[segment_msg_id]

                    return True

                elif "/" in header:
                    seq_num, total_chunks = map(int, header.split("/"))
                    if self.current_message_id is None:
                        self.current_message_id = f"msg_{time.time()}"
                    if self.current_message_id not in self.fragments:
                        self.fragments[self.current_message_id] = {}
                    self.fragments[self.current_message_id][seq_num] = content

                    if len(self.fragments[self.current_message_id]) == total_chunks:
                        self.process_frags()
                        del self.fragments[self.current_message_id]
                        self.current_message_id = None

                    return True
            except Exception as e:
                print(f"Error processing audio segment : {e}")

        return False

    def process_flac_fst(self, data):
        try:
            with io.BytesIO(data) as flac_buffer:
                audio_array, sample_rate = sf.read(flac_buffer, dtype="int16")

                if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                    audio_array = audio_array[:, 0]

                if len(audio_array) == 0:
                    return

                pcm_data = audio_array.astype(np.int16).tobytes()
                self.output_queue.put(pcm_data)

        except Exception as e:
            print(f"Error processing flac: {e}")

    def process_frags(self):
        try:
            if len(self.fragments[self.current_message_id]) > 0:
                sorted_keys = sorted(self.fragments[self.current_message_id].keys())
                assembled_chunks = [
                    self.fragments[self.current_message_id][k] for k in sorted_keys
                ]
                assembled_data = b"".join(assembled_chunks)

                if (
                    assembled_data.startswith(b"RIFF")
                    and b"WAVE" in assembled_data[:12]
                ):
                    self.wav_data(assembled_data)
                elif assembled_data.startswith(b"fLaC"):
                    self.flac_data(assembled_data)
                else:
                    self.output_queue.put(assembled_data)
        except Exception as e:
            print(f"Error processing data : {e}")

    def wav_data(self, data):
        try:
            wav_file = wave.open(io.BytesIO(data), "rb")
            audio_data = wav_file.readframes(wav_file.getnframes())
            self.output_queue.put(audio_data)
        except Exception:
            self.output_queue.put(data)

    def flac_data(self, data):
        try:
            with io.BytesIO(data) as flac_file:
                audio_data, sample_rate = sf.read(flac_file, dtype="int16")

                if sample_rate != self.config["sample_rate"]:
                    return

                if isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.tobytes()

                self.output_queue.put(audio_data)

        except Exception:
            self.output_queue.put(data)

    def audio_output(self):
        try:
            self.output_stream = self.audio.open(
                format=self.config["format"],
                channels=self.config["channels"],
                rate=self.config["sample_rate"],
                output=True,
                output_device_index=self.config["output_device_index"],
                frames_per_buffer=self.config["chunk_size"],
            )
            self.device_open = True
        except Exception:
            return

        total_played = 0
        self.stream_on = False

        while self.running:
            try:
                try:
                    audio_data = self.output_queue.get(timeout=0.1)

                    if (
                        audio_data == b"__END_OF_AUDIO__"
                        or audio_data == b"__TIMEOUT__"
                    ):
                        self.stream_on = False
                        continue

                    if not audio_data or len(audio_data) == 0:
                        if self.stream_on:
                            self.output_stream.write(self.padding)
                            continue
                        continue

                    self.stream_on = True

                    if len(audio_data) % 2 != 0:  # 16kHz mono
                        audio_data = (
                            audio_data[:-1] if len(audio_data) % 2 else audio_data
                        )

                    if self.config["output_volume"] != 1.0:
                        try:
                            audio_data = audioop.mul(
                                audio_data, 2, self.config["output_volume"]
                            )
                        except Exception as e:
                            print(f"Error setting volume: {e}")

                    chunk_size = self.config["chunk_size"] * 4
                    for i in range(0, len(audio_data), chunk_size):
                        if not self.running:
                            break
                        chunk = audio_data[i : i + chunk_size]
                        try:
                            self.output_stream.write(chunk)
                            total_played += len(chunk)
                        except Exception:
                            break

                except queue.Empty:
                    if self.output_stream:
                        self.output_stream.write(self.padding)
                        continue
                    continue
                except Exception:
                    time.sleep(0.1)

            except Exception:
                time.sleep(0.5)

        try:
            if self.output_stream and self.device_open:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.device_open = False
        except Exception as e:
            print(f"Error closing audio stream : {e}")

    def audio_input(self):
        try:
            self.input_stream = self.audio.open(
                format=self.config["format"],
                channels=self.config["channels"],
                rate=self.config["sample_rate"],
                input=True,
                input_device_index=self.config["input_device_index"],
                frames_per_buffer=self.config["chunk_size"],
            )
        except Exception:
            return

        audio_levels = []
        level_check_counter = 0

        while self.running:
            try:
                data = self.input_stream.read(
                    self.config["chunk_size"], exception_on_overflow=False
                )

                level_check_counter += 1
                if level_check_counter % 10 == 0:
                    rms = audioop.rms(data, 2)
                    audio_levels.append(rms)

                    if len(audio_levels) > 100:
                        audio_levels.pop(0)

                    level_check_counter = 0

                self.input_queue.put(data)

            except IOError:
                time.sleep(0.1)

            except Exception:
                time.sleep(0.5)

        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
        except Exception:
            pass

    def play_beep(self):
        duration = 0.15
        pause = 0.07
        frequency = 1109
        sample_rate = 16000

        fade_duration = 0.0
        fade_length = int(sample_rate * fade_duration)

        t1 = np.linspace(0, duration, int(sample_rate * duration), False)
        tone1 = 0.5 * np.sin(frequency * 2 * np.pi * t1)
        t_pause = np.zeros(int(sample_rate * pause))
        t2 = np.linspace(0, duration, int(sample_rate * duration), False)
        tone2 = 0.5 * np.sin(frequency * 2 * np.pi * t2)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)

        if len(tone1) > 2 * fade_length:
            tone1[:fade_length] *= fade_in
            tone1[-fade_length:] *= fade_out

        if len(tone2) > 2 * fade_length:
            tone2[:fade_length] *= fade_in
            tone2[-fade_length:] *= fade_out

        full_tone = np.concatenate((tone1, t_pause, tone2))

        audio_data = (full_tone * 32768.0).astype(np.int16)

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
        )

        stream.write(audio_data.tobytes())

        stream.stop_stream()
        stream.close()
        p.terminate()

        time.sleep(0.1)

    def start(self):
        self.threads = []

        input_thread = threading.Thread(target=self.audio_input, daemon=True)
        input_thread.start()
        self.threads.append(input_thread)

        sender_thread = threading.Thread(target=self.audio_sender, daemon=True)
        sender_thread.start()
        self.threads.append(sender_thread)

        receiver_thread = threading.Thread(target=self.audio_receiver, daemon=True)
        receiver_thread.start()
        self.threads.append(receiver_thread)

        output_thread = threading.Thread(target=self.audio_output, daemon=True)
        output_thread.start()
        self.threads.append(output_thread)

        try:
            while self.running:
                time.sleep(0.1)
        finally:
            self.running = False
            time.sleep(5.0)

            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)

    def stop(self):
        self.running = False

        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass

        if self.output_stream and self.device_open:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.device_open = False
            except Exception:
                pass

        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        try:
            self.audio.terminate()
        except Exception:
            pass


def main():
    streamer = AudioStreamer()
    streamer.start()


if __name__ == "__main__":
    main()
