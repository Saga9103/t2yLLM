import hmac
import hashlib
import json
import time
import secrets
from pathlib import Path
from typing import Dict


class HMACAuth:
    def __init__(self, secret_key: bytes = None):
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
