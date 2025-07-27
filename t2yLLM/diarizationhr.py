import redis
import io
import numpy as np
import hashlib
import secrets
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn.functional as F
import faiss
import logging
import threading
import os
import time
import shutil
import tempfile
import subprocess
from speechbrain.inference.speaker import EncoderClassifier
from t2yLLM.config.yamlConfigLoader import Loader


class SpeakerManager:
    """
    High performance distributed Speaker identification module to handle
    many simultaneous users :

    - Uses Faiss (faiss-cpu pip maintained version) as in memory vector similarity search.
    Since in theory users are grouped in sessions of 1 to 15 people,
    faiss.IndexFlatIP is used.
    It is not too expansive and does not relies on training or graph
    structure/construct so embeddings can be removed anytime if 1 or more were
    allocated to the wrong user. Uses locks per user groups (multiple sessions)

    - Uses redis for more complex session and user data retriaval. Redis server
    is created temporarly and as soon as the session is closed, the data is
    erased and files destroyed. Each new launch creates new password for access.
    Could maybe use SCAN instead of KEYS but advantage is not really clear.

    - Speaker Diarization relies on SpeechBrain spkrec-ecapa-voxceleb model
    from huggingface.
    It gets speech from the dominant speaker and prevents mixing
    different portions of text from different user.
    Ignores other users.

    - Not used here, just overkill and unecessary architecture
      and dependencies for HA. test for an other project.
      If you want to use it anyway :
      sudo apt install redis-server
      pip install faiss-cpu (or set up a conda env for the orginal faiss)
      then set Distributed to True in faster_whisper.general in server_config.yaml
    """

    def __init__(
        self,
        device="cpu",
        similarity_threshold=0.25,  # dissimilar users ofter show < 0.1
        merge_threshold=0.45,  # merges users that were wrongly separated, kinda conservative for 5+ people
        redis_db=0,
        clear_on_start=False,  # redis clean on startup
    ):
        self.logger = logging.getLogger("SpeakerManager")
        self.config = Loader().loadWhispConfig()
        self.device = (
            device  # always cpu if in a pip env, could be switched to gpu with conda
        )
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold

        # redis management
        # unix socket
        self.redis_socket_path = self.config.general.redis_socket_path
        self.redis_socket_perm = self.config.general.redis_socket_perm
        self.redis_db = redis_db
        self.redis_passwd = secrets.token_urlsafe(32)  # random passord on startup
        self.redis_process = None
        self.redis_config_file = None

        self.start_redis()

        self.session_id = secrets.token_urlsafe(16)  # redis unique sess id
        self.session_hash = hashlib.sha256(self.session_id.encode()).hexdigest()[:32]
        # unix socket setup
        self.redis_client = redis.Redis(
            port=0,
            unix_socket_path=self.redis_socket_path,
            db=redis_db,
            password=self.redis_passwd,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        self.await_redis()

        self.min_confidence_threshold = 0.17
        self.confidence_increase_rate = 0.20
        self.max_embeddings_per_speaker = (
            7  # small ok number since users are session separated in little groups
        )
        self.min_speeches_before_removal = 3
        self.embedding_dim = 192  # from speechbrain model

        # Session-based FAISS indices
        self.session_indices = {}  # session_id -> faiss_index
        self.session_embeddings_map = {}  # session_id -> {idx: (speaker_id, emb_idx)}
        self.session_next_idx = {}  # session_id -> next_idx
        self.session_locks = {}  # per session lock to allow multiple independant sessions
        self.locks_manager = threading.Lock()
        self.session_last_access = {}  # 1 hour session max time will be tracked

        self.id_lock = threading.Lock()

        # Initialize speaker IDs from Redis
        self.init_speaker_counter()

        # Speechbrain
        self.classifier = None
        self.load_ecapa()

    # REDIS

    def start_redis(self):
        """
        starts a temporary redis server (lifetime = session time)
        with a temporary config with minimal security and no
        command renaming or limitations per user/session
        with a random passwd and unix socket communication for local. faster, secure
        """
        try:
            self.redis_dir = tempfile.mkdtemp(prefix="t2yLLM_redis_")

            config_content = f"""
                            port 0
                            unixsocket {self.redis_socket_path}
                            unixsocketperm {self.redis_socket_perm}
                            dir {self.redis_dir}
                            save ""
                            appendonly no
                            protected-mode yes
                            requirepass {self.redis_passwd}
                            daemonize no
                            loglevel warning
                            """

            self.redis_config_file = os.path.join(self.redis_dir, "redis.conf")
            with open(self.redis_config_file, "w") as f:
                f.write(config_content)

            self.redis_process = subprocess.Popen(
                [self.config.general.redis_path, self.redis_config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)

            if self.redis_process.poll() is not None:
                raise RuntimeError("\033[91mRedis server failed to start\033[0m")

            self.logger.info("\033[92mRedis server started\033[0m")

        except Exception as e:
            raise RuntimeError(f"\033[91mFailed to start Redis server: {e}\033[0m")

    def await_redis(self):
        """Wait for Redis to be ready and
        tries connecting 10 times before raising RuntimeError
        """
        for i in range(10):
            try:
                self.redis_client.ping()
                return
            except redis.ConnectionError:
                if i == 9:
                    raise RuntimeError(
                        "\033[91mRedis server did not start in time\033[0m"
                    )
                time.sleep(0.5)

    def clear_data(self):
        """
        Clears all keys related to the current session and global keys from Redis
        deletes the redis key mathing 'global:*' and '<session_key>:*'
        """
        patterns = ["global:*", f"{self.session_hash}:*"]

        for pattern in patterns:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                self.logger.info(
                    f"Deleted {len(keys)} keys matching pattern: {pattern}"
                )

    def __del__(self):
        """Redis destructor that ends redis
        process and cleans up temporary files and directory if any
        """
        if self.redis_process:
            self.clear_data()
            self.redis_process.terminate()
            try:
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.redis_process.kill()
                self.redis_process.wait()

            self.logger.info(
                "\033[91mRedis server stopped and data cleared successfully\033[0m"
            )

        if self.redis_config_file and os.path.exists(self.redis_config_file):
            os.remove(self.redis_config_file)

        if hasattr(self, "redis_dir") and os.path.exists(self.redis_dir):
            shutil.rmtree(self.redis_dir)

    def session_key(self, session_id: str, key_type: str, *args) -> str:
        """Generates Redis key for a session
        Args:
            session_id : str
            key_type : str
            *args : str, additional elts to append to the key
        Returns str colon sperated redis key string
        """
        parts = [session_id, key_type] + [str(arg) for arg in args]
        return ":".join(parts)

    def global_key(self, key_type: str, *args) -> str:
        """Generate global Redis key (shared across sessions)
        Args:
            key_type : str
            *args : str suffixes for the key
        Returns : str in format 'global:<key_type>:<args>'
        """
        parts = ["global", key_type] + [str(arg) for arg in args]
        return ":".join(parts)

    def init_speaker_counter(self):
        """Initialize or get the global speaker counter
        Creates 'global:next_speaker_id' with initial value 0 if it does not exist
        """
        counter_key = self.global_key("next_speaker_id")
        if not self.redis_client.exists(counter_key):
            self.redis_client.set(counter_key, 0)

    def get_next_speaker_id(self, session_id: str) -> int:
        """Get next speaker ID for a given session
        Args:
            session_id : str
        Returns : int session specific speaker ID
        """
        counter_key = self.session_key(session_id, "next_speaker_id")
        return self.redis_client.incr(counter_key) - 1

    # SPEECHBRAIN
    def load_ecapa(self):
        """Load the SpeechBrain speaker encoder classifier model
        from the ECAPA-TDNN of speechbrain atm
        """
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            self.logger.info(
                f"\033[92mSpeaker diarization model loaded successfully on {self.device}\033[0m"
            )
        except Exception as e:
            self.logger.error(
                f"\033[91mFailed to load speaker diarization model: {e}\033[0m"
            )
            self.classifier = None

    def get_speaker_embedding(self, audio_data, sample_rate=16000) -> torch.Tensor:
        """Generates speaker embedding from audio data
        Args:
            audio_data : np.ndarray of shape (,samples) or (channels, samples)
            sample_rate : int, 16kHz is mandatory and used everywhere so wont be in config
        Returns : torch.Tensor --> (192,)
        """
        if self.classifier is None:
            return None

        try:
            audio_tensor = torch.from_numpy(audio_data).float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                embeddings = self.classifier.encode_batch(audio_tensor, normalize=True)
            # from encode_batch, embeddings[0] would return torch.Tensor of shape (1,192)
            embedding = embeddings[0][0]
            return embedding

        except Exception as e:
            self.logger.warning(f"\033[93mError getting speaker embedding: {e}\033[0m")
            return None

    def load_speaker_embeddings(
        self, session_id: str, speaker_id: int
    ) -> List[torch.Tensor]:
        """Loads speaker realated embeddings from Redis for the session
        Args:
            session_id : str (uuid)
            speaker_id : int
        REturns : List[torch.Tensor] --> List[(192,), ...]
        """
        key = self.session_key(session_id, "embeddings", speaker_id)
        data = self.redis_client.get(key)
        if data:
            buffer = io.BytesIO(data)
            embeddings_list = torch.load(buffer, map_location=self.device)
            return embeddings_list
        return []

    def set_speaker_embeddings(
        self, session_id: str, speaker_id: int, embeddings: List[torch.Tensor]
    ):
        """Save the list of embeddings to store in Redis for a speaker
        Args:
            session_id : str (uuid)
            speaker_id : int
            embeddings : List[torch.Tensor], each of shape (192,)
        """
        key = self.session_key(session_id, "embeddings", speaker_id)
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        self.redis_client.set(key, buffer.getvalue())

    def load_speaker_centroid(
        self, session_id: str, speaker_id: int
    ) -> Optional[torch.Tensor]:
        """Finds the centroid in Redis for a speaker
        Args:
            session_id : str
            speaker_id : int
        Returns : torch.Tensor of shape (192,) or None
        """
        key = self.session_key(session_id, "centroid", speaker_id)
        data = self.redis_client.get(key)
        if data:
            buffer = io.BytesIO(data)
            centroid = torch.load(buffer, map_location=self.device)
            return centroid
        return None

    def set_speaker_centroid(
        self, session_id: str, speaker_id: int, centroid: torch.Tensor
    ):
        """Saves the centroid of a speaker into Redis
        Args:
            session_id : str
            speaker_id : int
            centroid : torch.Tensor of shape (192,)
        """
        key = self.session_key(session_id, "centroid", speaker_id)
        buffer = io.BytesIO()
        torch.save(centroid, buffer)
        self.redis_client.set(key, buffer.getvalue())

    def load_speaker_ids(self, session_id: str) -> List[int]:
        """Gets all speaker IDs for a given session
        Args:
            session_id : str (uuid)
        Returns a list of ints"""
        pattern = self.session_key(session_id, "embeddings", "*")
        keys = self.redis_client.keys(pattern)
        speaker_ids = []
        for key in keys:
            try:
                parts = key.decode("utf-8").split(":")
                speaker_id = int(parts[-1])
                speaker_ids.append(speaker_id)
            except Exception:
                continue
        return speaker_ids

    def update_centroid(self, session_id: str, speaker_id: int) -> None:
        """Updates and saves the centroid for the user and session from their embeddings
        Args:
            session_id : str (uuid)
            speaker_id : int
        """
        embeddings = self.load_speaker_embeddings(session_id, speaker_id)
        if not embeddings:
            return

        weights = np.array([(i + 1) / len(embeddings) for i in range(len(embeddings))])
        weights = weights / weights.sum()

        centroid = sum(w * emb for w, emb in zip(weights, embeddings))
        centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

        self.set_speaker_centroid(session_id, speaker_id, centroid)

    # FAISS

    def session_locker(self, session_id: str) -> threading.Lock:
        """Gets or creates a lock for a specific session
        in order to avoid a global lock for all sessions
        allowing non blocking operations between cross sessions"""
        """
        if session_id in self.session_locks:
            #if session exists just update access time
            # not wanted for now, only 1 hour max sessions
            self.session_last_access[session_id] = time.time()
            return self.session_locks[session_id]
        """
        with self.locks_manager:
            if session_id in self.session_locks:
                return self.session_locks[session_id]
            self.session_locks[session_id] = threading.Lock()
            self.session_last_access[session_id] = time.time()
            return self.session_locks[session_id]

    def session_indexer(self, session_id: str):
        """creates or gets a FAISS index for a session
        Returns : faiss.IndexFlatIP associated to the session
        """
        session_lock = self.session_locker(session_id)

        with session_lock:  # Only blocking the current session
            if session_id not in self.session_indices:
                # new index
                self.session_indices[session_id] = faiss.IndexFlatIP(self.embedding_dim)
                self.session_embeddings_map[session_id] = {}
                self.session_next_idx[session_id] = 0
                self.logger.debug(f"Created new FAISS index for session {session_id}")
            return self.session_indices[session_id]

    def rebuild_session_index(self, session_id: str):
        """
        Rebuilds the FAISS index for a session from data stored inRedis
        """
        session_lock = self.session_locker(session_id)

        with session_lock:
            self.session_indices[session_id] = faiss.IndexFlatIP(self.embedding_dim)
            self.session_embeddings_map[session_id] = {}
            idx = 0

            speaker_ids = self.load_speaker_ids(session_id)

            for speaker_id in speaker_ids:
                embeddings = self.load_speaker_embeddings(session_id, speaker_id)
                for emb_idx, embedding in enumerate(embeddings):
                    emb_np = embedding.cpu().numpy().astype(np.float32)
                    self.session_indices[session_id].add(emb_np.reshape(1, -1))
                    self.session_embeddings_map[session_id][idx] = (speaker_id, emb_idx)
                    idx += 1

            self.session_next_idx[session_id] = idx
            self.logger.debug(
                f"Rebuilt session {session_id} index with {idx} embeddings"
            )

    def __call__(
        self, audio_data, session_id: str, sample_rate=16000
    ) -> Tuple[str, int]:
        """For a given session, it identifies the speaker from
        received audio data
        Args:
            audio_data : np.ndarray shape (samples,) or (channels, samples)
            session_id : str (uuid)
            sample_rate : fixed int of 16kHz, not in config
        Returns :
            (Speaker label : str, Speaker ID : int or -1 if unknown)
        """
        embedding = self.get_speaker_embedding(audio_data, sample_rate)
        if embedding is None:
            return "Unknown", -1

        session_lock = self.session_locker(session_id)
        session_index = self.session_indexer(session_id)
        session_map = self.session_embeddings_map[session_id]

        similarities = {}

        with session_lock:
            if session_index.ntotal > 0:
                query_embedding = (
                    embedding.cpu().numpy().astype(np.float32).reshape(1, -1)
                )
                k = min(
                    7, session_index.ntotal
                )  #  50 50 for neighbors for 15 estimated
                distances, indices = session_index.search(query_embedding, k)

                speaker_candidates = set()
                for idx in indices[0]:
                    if idx != -1 and idx in session_map:
                        speaker_id, _ = session_map[idx]
                        speaker_candidates.add(speaker_id)

                for speaker_id in speaker_candidates:
                    centroid = self.load_speaker_centroid(session_id, speaker_id)
                    if centroid is not None:
                        similarity = F.cosine_similarity(
                            embedding.unsqueeze(0), centroid.unsqueeze(0)
                        ).item()
                        similarities[speaker_id] = similarity

        if similarities:
            best_speaker_id = max(similarities, key=similarities.get)
            best_similarity = similarities[best_speaker_id]

            if best_similarity >= self.similarity_threshold:
                self.add_embedding_atomic(session_id, best_speaker_id, embedding)
                self.increment_state(session_id, best_speaker_id)
                self.update_confidence(session_id, best_speaker_id, matched=True)

                label_key = self.session_key(session_id, "label", best_speaker_id)
                label = self.redis_client.get(label_key)
                if label:
                    label = label.decode("utf-8")
                else:
                    label = f"Speaker_{best_speaker_id:02d}"

                conf = float(
                    self.redis_client.get(
                        self.session_key(session_id, "confidence", best_speaker_id)
                    )
                    or 0.6
                )

                self.logger.info(
                    f"\033[94m[Session {session_id}]\n {label} speaking (conf: {conf:.2f})\033[0m"
                )
                return label, best_speaker_id

        with self.id_lock:
            speaker_id = self.get_next_speaker_id(session_id)

        label = f"Speaker_{speaker_id:02d}"
        self.redis_client.set(self.session_key(session_id, "label", speaker_id), label)
        self.set_speaker_embeddings(session_id, speaker_id, [embedding])
        self.redis_client.set(
            self.session_key(session_id, "confidence", speaker_id), "0.6"
        )
        self.redis_client.set(self.session_key(session_id, "speeches", speaker_id), "1")
        self.update_centroid(session_id, speaker_id)

        with session_lock:
            emb_np = embedding.cpu().numpy().astype(np.float32)
            session_index.add(emb_np.reshape(1, -1))
            idx = self.session_next_idx[session_id]
            session_map[idx] = (speaker_id, 0)
            self.session_next_idx[session_id] = idx + 1

        self.logger.info(
            f"\033[94m[Session : {session_id}] :\n New speaker detected: {label}\033[0m"
        )
        return label, speaker_id

    def add_embedding_atomic(
        self, session_id: str, speaker_id: int, embedding: torch.Tensor
    ):
        """Add embedding atomically for a session speaker and updates the Faiss index
        Args:
            session_id : str
            speaker_id : int
            embedding : torch.Tensor of shape (192,)
        """
        embeddings = self.load_speaker_embeddings(session_id, speaker_id)
        session_lock = self.session_locker(session_id)

        with session_lock:
            session_index = self.session_indices.get(session_id)
            if session_index:
                emb_np = embedding.cpu().numpy().astype(np.float32)
                session_index.add(emb_np.reshape(1, -1))
                idx = self.session_next_idx[session_id]
                self.session_embeddings_map[session_id][idx] = (
                    speaker_id,
                    len(embeddings),
                )
                self.session_next_idx[session_id] = idx + 1

        embeddings.append(embedding)

        if len(embeddings) > self.max_embeddings_per_speaker:
            embeddings = embeddings[-self.max_embeddings_per_speaker :]

        self.set_speaker_embeddings(session_id, speaker_id, embeddings)
        self.update_centroid(session_id, speaker_id)

    def increment_state(self, session_id: str, speaker_id: int):
        """Increment speech count atomically for a session speaker
        and can optionally trigger merging
        Args:
            session_id : str
            speaker_id : int
        """
        key = self.session_key(session_id, "speeches", speaker_id)
        count = self.redis_client.incr(key)

        if count % 5 == 0:
            self.check_merges(session_id)

    def update_confidence(self, session_id: str, speaker_id: int, matched: bool = True):
        """based on the match result, updates the confidence score"""
        key = self.session_key(session_id, "confidence", speaker_id)
        current = float(self.redis_client.get(key) or 0.6)

        if matched:
            new_conf = min(1.0, current + self.confidence_increase_rate)
        else:
            new_conf = max(0.0, current)  # No decay

        self.redis_client.set(key, str(new_conf))

    def check_merges(self, session_id: str):
        """If any speakers in the same session are
        too similar, merges them from threshold
        """
        speaker_ids = self.load_speaker_ids(session_id)

        for i, speaker1_id in enumerate(speaker_ids):
            centroid1 = self.load_speaker_centroid(session_id, speaker1_id)
            if centroid1 is None:
                continue

            for speaker2_id in speaker_ids[i + 1 :]:
                centroid2 = self.load_speaker_centroid(session_id, speaker2_id)
                if centroid2 is None:
                    continue

                # centroids are compared for fast checking first
                centroid_sim = F.cosine_similarity(
                    centroid1.unsqueeze(0), centroid2.unsqueeze(0)
                ).item()

                if centroid_sim > self.merge_threshold:
                    emb1 = self.load_speaker_embeddings(session_id, speaker1_id)
                    emb2 = self.load_speaker_embeddings(session_id, speaker2_id)

                    if emb1 and emb2:
                        similarities = []
                        for e1 in emb1[-5:]:
                            for e2 in emb2[-5:]:
                                sim = F.cosine_similarity(
                                    e1.unsqueeze(0), e2.unsqueeze(0)
                                ).item()
                                similarities.append(sim)

                        if (
                            similarities
                            and np.mean(similarities) > self.merge_threshold
                        ):
                            self.merge_speakers(session_id, speaker1_id, speaker2_id)
                            return

    def merge_speakers(
        self, session_id: str, speaker1_id: int, speaker2_id: int
    ) -> None:
        """Merges two speakers within the same session"""
        conf1 = float(
            self.redis_client.get(
                self.session_key(session_id, "confidence", speaker1_id)
            )
            or 0
        )
        conf2 = float(
            self.redis_client.get(
                self.session_key(session_id, "confidence", speaker2_id)
            )
            or 0
        )
        speech1 = int(
            self.redis_client.get(self.session_key(session_id, "speeches", speaker1_id))
            or 0
        )
        speech2 = int(
            self.redis_client.get(self.session_key(session_id, "speeches", speaker2_id))
            or 0
        )

        score1 = conf1 * speech1
        score2 = conf2 * speech2

        if score1 < score2:
            speaker1_id, speaker2_id = speaker2_id, speaker1_id

        emb1 = self.load_speaker_embeddings(session_id, speaker1_id)
        emb2 = self.load_speaker_embeddings(session_id, speaker2_id)
        merged = emb1 + emb2
        if len(merged) > self.max_embeddings_per_speaker:
            merged = merged[-self.max_embeddings_per_speaker :]

        self.set_speaker_embeddings(session_id, speaker1_id, merged)

        new_speeches = speech1 + speech2
        self.redis_client.set(
            self.session_key(session_id, "speeches", speaker1_id),
            str(new_speeches),
        )

        self.redis_client.set(
            self.session_key(session_id, "confidence", speaker1_id),
            str(max(conf1, conf2)),
        )

        label1 = self.redis_client.get(
            self.session_key(session_id, "label", speaker1_id)
        )
        label2 = self.redis_client.get(
            self.session_key(session_id, "label", speaker2_id)
        )
        if label1:
            label1 = label1.decode("utf-8")
        else:
            label1 = f"Speaker_{speaker1_id:02d}"
        if label2:
            label2 = label2.decode("utf-8")
        else:
            label2 = f"Speaker_{speaker2_id:02d}"

        keys_to_delete = [
            self.session_key(session_id, "embeddings", speaker2_id),
            self.session_key(session_id, "centroid", speaker2_id),
            self.session_key(session_id, "label", speaker2_id),
            self.session_key(session_id, "confidence", speaker2_id),
            self.session_key(session_id, "speeches", speaker2_id),
        ]
        self.redis_client.delete(*keys_to_delete)
        self.update_centroid(session_id, speaker1_id)
        self.logger.info(
            f"\033[93m[Session {session_id[:8]}] Merged {label2} into {label1}\033[0m"
        )

    def cleanup_stale_locks(self, max_age_seconds=3605):
        """Inactive sessions of more than one hour
        are to be unlocked and cleaned up"""
        current_time = time.time()
        sessions_to_clean = []

        with self.locks_manager:
            for session_id, last_access in list(self.session_last_access.items()):
                if current_time - last_access > max_age_seconds:
                    pattern = self.session_key(session_id, "*")
                    if not self.redis_client.keys(pattern):
                        sessions_to_clean.append(session_id)

        for session_id in sessions_to_clean:
            with self.locks_manager:
                if session_id in self.session_locks:
                    del self.session_locks[session_id]
                if session_id in self.session_last_access:
                    del self.session_last_access[session_id]

        if sessions_to_clean:
            self.logger.info(f"Cleaned up {len(sessions_to_clean)} stale session locks")

    def get_session_info(self, session_id: str) -> Dict:
        """all metadata about speakers in the same session"""
        info = {}

        for speaker_id in self.load_speaker_ids(session_id):
            label = self.redis_client.get(
                self.session_key(session_id, "label", speaker_id)
            )
            if label:
                label = label.decode("utf-8")
            else:
                label = f"Speaker_{speaker_id:02d}"

            conf = float(
                self.redis_client.get(
                    self.session_key(session_id, "confidence", speaker_id)
                )
                or 0
            )
            speech = int(
                self.redis_client.get(
                    self.session_key(session_id, "speeches", speaker_id)
                )
                or 0
            )
            embeddings = self.load_speaker_embeddings(session_id, speaker_id)

            info[label] = {
                "confidence": conf,
                "speeches": speech,
                "num_embeddings": len(embeddings),
            }

        return info
