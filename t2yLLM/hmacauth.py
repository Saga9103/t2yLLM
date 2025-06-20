import hmac
import hashlib
import json
import secrets
from pathlib import Path
from typing import Dict, Optional
import os
import sys

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    from t2yLLM.config.yamlConfigLoader import Loader

    config = Loader().loadChatConfig()
    HMAC_ENABLED = config.network.hmac_enabled
except Exception as e:
    print(f"Warning: Could not load config for HMAC status: {e}")
    HMAC_ENABLED = False


class HMACAuth:
    def __init__(self, secret_key: bytes = None):
        if not HMAC_ENABLED:
            self.secret_key = None
            return

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
        if not HMAC_ENABLED:
            return b""
        h = hmac.new(self.secret_key, message, hashlib.sha256)
        return h.digest()

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        if not HMAC_ENABLED:
            return True
        expected = self.create_signature(message)
        return hmac.compare_digest(expected, signature)

    def pack_message(self, data: Dict) -> bytes:
        if not HMAC_ENABLED:
            return json.dumps(data).encode("utf-8")
        else:
            message = json.dumps(data).encode("utf-8")
            signature = self.create_signature(message)
            return signature + b"||" + message

    def unpack_message(self, packed_msg: bytes):
        if not HMAC_ENABLED:
            try:
                return json.loads(packed_msg.decode("utf-8"))
            except Exception:
                return None
        else:
            try:
                parts = packed_msg.split(b"||", 1)
                if len(parts) != 2:
                    return None
                signature, message = parts

                if not self.verify_signature(message, signature):
                    return None

                data = json.loads(message.decode("utf-8"))
                return data
            except Exception:
                return None
