import logging
from datetime import datetime
import numpy as np
from typing import Optional, List, Tuple, Dict
import torch
from speechbrain.inference.speaker import EncoderClassifier


class HASpeakerManager:
    """Home Assisntant Speaker identification module :
    relies on SpeechBrain spkrec-ecapa-voxceleb model
    that gets speech from the dominant speaker and prevents mixing
    different portions of text from different user.
    gets dominant user, ignores other users."""

    def __init__(
        self,
        device="cpu",
        similarity_threshold=0.25,
        merge_threshold=0.45,
        data_file="speaker_data.json",
        max_speakers=8,
    ):
        self.logger = logging.getLogger("HASpeakerManager")
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.max_speakers = max_speakers

        self.speakers = {}  # speaker_id -> {label, embeddings, confidence, speeches, last_seen}
        self.next_speaker_id = 0

        self.max_embeddings_per_speaker = (
            7  # max number of embeddings per user to update and create the centroid
        )
        self.confidence_increase_rate = 0.20
        self.embedding_dim = 192  # speechbrain model dim
        self.classifier = None
        self.load_ecapa()

    def load_ecapa(self):
        """Load SpeechBrain encoder classifier for diarization"""
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            self.logger.info(f"Speaker model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load speaker model: {e}")
            self.classifier = None

    def get_speaker_embedding(
        self, audio_data, sample_rate=16000
    ) -> Optional[np.ndarray]:
        """Get speaker embedding from audio data
        Args :
            audio_data --> numpy array of mono 16k audio
            sample rate --> fixed

        returns a numpy array"""
        if self.classifier is None:
            return None

        try:
            audio_tensor = torch.from_numpy(audio_data).float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                embeddings = self.classifier.encode_batch(audio_tensor, normalize=True)

            embedding = embeddings[0][0].cpu().numpy()
            return embedding

        except Exception as e:
            self.logger.warning(f"Error getting speaker embedding: {e}")
            return None

    def compute_centroid(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Compute weighted centroid from a user embeddings
        Args:
            embeddings --> list of numpy arrays of shape (192,) representing speaker embeddings
        Returns a normalized numpy array of shape (192,) representing the centroid
        """
        if not embeddings:
            return None

        emb_array = np.array(embeddings)
        weights = np.array([(i + 1) / len(embeddings) for i in range(len(embeddings))])
        weights = weights / weights.sum()
        centroid = np.sum(emb_array.T * weights, axis=1)
        centroid = centroid / np.linalg.norm(centroid)

        return centroid

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors
        it could be dot product only since they are already normalized
        Args:
            numpy arrays (pre-normalized) of shape (192,)
        Returns a float in [-1.0, 1.0]"""
        dot_prod = np.dot(emb1, emb2)
        # not really useful because normalized already but coherent
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        cosine_sim = dot_prod / (n1 * n2)

        return cosine_sim

    def closest_neighbor(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        compares the vector to the centroid
        to find the first found closest matching user
        Args:
            embedding --> numpy array of the current speaker's embedding of shape (192,)
        Returns (speaker_id: int, similarity: float) or (None, 0.0) if no match
        """
        best_speaker_id = None
        best_similarity = 0.0

        for speaker_id, data in self.speakers.items():
            if not data["embeddings"]:
                continue

            centroid = self.compute_centroid(data["embeddings"])
            if centroid is not None:
                similarity = self.cosine_similarity(embedding, centroid)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker_id = speaker_id

        return best_speaker_id, best_similarity

    def __call__(
        self, audio_data, session_id=None, sample_rate=16000
    ) -> Tuple[str, int]:
        """Main method to identify or create a new speaker from audio
        Args:
            audio_data --> 1D numpy array of mono 16k audio
            session_id --> optional str identifier (unused), just here for compatibility
            sample_rate --> fixed, defaults to 16000 mandatory for the model and used anywhere anyway

        returns (label: str, speaker_id: int), or ("Unknown", -1) if unrecognized
        """
        embedding = self.get_speaker_embedding(audio_data, sample_rate)
        if embedding is None:
            return "Unknown", -1

        best_speaker_id, best_similarity = self.closest_neighbor(embedding)

        if best_speaker_id is not None and best_similarity >= self.similarity_threshold:
            self.add_embedding(best_speaker_id, embedding)
            speaker = self.speakers[best_speaker_id]
            speaker["speeches"] += 1
            speaker["last_seen"] = datetime.now().isoformat()
            speaker["confidence"] = min(
                1.0, speaker["confidence"] + self.confidence_increase_rate
            )

            label = speaker["label"]
            self.logger.info(
                f"{label} speaking (confidence: {speaker['confidence']:.2f})"
            )

            if speaker["speeches"] % 5 == 0:
                self.check_merges()

            return label, best_speaker_id

        if len(self.speakers) >= self.max_speakers:
            self.logger.warning(f"Maximum speakers ({self.max_speakers}) reached")
            return "Unknown", -1

        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1

        label = f"Speaker_{speaker_id + 1}"

        self.speakers[speaker_id] = {
            "label": label,
            "embeddings": [embedding],
            "confidence": 0.6,
            "speeches": 1,
            "last_seen": datetime.now().isoformat(),
        }

        self.logger.info(f"New speaker detected: {label}")
        return label, speaker_id

    def add_embedding(self, speaker_id: int, embedding: np.ndarray) -> None:
        """Add an embedding to an existing speaker
        Args:
            speaker_id --> integer ID of the speaker
            embedding --> numpy array shape (192,) to add to speaker history
        Updates the speaker's embedding list in-place"""
        if speaker_id not in self.speakers:
            return

        embeddings = self.speakers[speaker_id]["embeddings"]
        embeddings.append(embedding)

        if len(embeddings) > self.max_embeddings_per_speaker:
            self.speakers[speaker_id]["embeddings"] = embeddings[
                -self.max_embeddings_per_speaker :
            ]

    def check_merges(self) -> None:
        """Checks if any speakers should be merged"""
        speaker_ids = list(self.speakers.keys())

        for i, id1 in enumerate(speaker_ids):
            if id1 not in self.speakers:
                continue

            centroid1 = self.compute_centroid(self.speakers[id1]["embeddings"])
            if centroid1 is None:
                continue

            for id2 in speaker_ids[i + 1 :]:
                if id2 not in self.speakers:
                    continue

                centroid2 = self.compute_centroid(self.speakers[id2]["embeddings"])
                if centroid2 is None:
                    continue

                similarity = self.cosine_similarity(centroid1, centroid2)

                if similarity > self.merge_threshold:
                    self.merge_speakers(id1, id2)
                    return

    def merge_speakers(self, id1: int, id2: int) -> None:
        """
        Merges two speakers
        Args:
            id1 --> int ID of first speaker
            id2 --> int ID of second speaker

        Merges embeddings, confidence, and metadata, and removes the second entry
        """
        speaker1 = self.speakers[id1]
        speaker2 = self.speakers[id2]

        if speaker2["speeches"] > speaker1["speeches"]:
            id1, id2 = id2, id1
            speaker1, speaker2 = speaker2, speaker1

        merged_embeddings = speaker1["embeddings"] + speaker2["embeddings"]
        if len(merged_embeddings) > self.max_embeddings_per_speaker:
            merged_embeddings = merged_embeddings[-self.max_embeddings_per_speaker :]

        speaker1["embeddings"] = merged_embeddings
        speaker1["speeches"] += speaker2["speeches"]
        speaker1["confidence"] = max(speaker1["confidence"], speaker2["confidence"])
        speaker1["last_seen"] = max(speaker1["last_seen"], speaker2["last_seen"])

        label1 = speaker1["label"]
        label2 = speaker2["label"]
        del self.speakers[id2]

        self.logger.info(f"Merged {label2} into {label1}")

    def set_speaker_name(self, speaker_id: int, name: str):
        """Set a custom name for a speaker
        Args:
            speaker_id --> integer ID of the speaker
            name --> str
        Updates in memory"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["label"] = name
            self.logger.info(f"Speaker {speaker_id} renamed to {name}")

    def get_speaker_info(self) -> Dict:
        """Get information about all speakers
        Returns dict keyed by speaker label each containing :
        {{ID : int, confidence: float, speeches: int, last_seen: str (timestamp), num_embeddings: int}, ...}
        """
        info = {}
        for speaker_id, data in self.speakers.items():
            info[data["label"]] = {
                "id": speaker_id,
                "confidence": data["confidence"],
                "speeches": data["speeches"],
                "last_seen": data["last_seen"],
                "num_embeddings": len(data["embeddings"]),
            }
        return info

    def reset_speaker(self, speaker_id: int):
        """Reset a speaker's embeddings
        In memory"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["embeddings"] = []
            self.speakers[speaker_id]["confidence"] = 0.6
            self.logger.info(f"Reset speaker {self.speakers[speaker_id]['label']}")

    def remove_speaker(self, speaker_id: int):
        """Remove a speaker"""
        if speaker_id in self.speakers:
            label = self.speakers[speaker_id]["label"]
            del self.speakers[speaker_id]
            self.logger.info(f"Removed speaker {label}")

    def cleanup_session(self):
        """only for compat"""
        pass
