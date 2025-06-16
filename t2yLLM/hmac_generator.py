import secrets

key = secrets.token_bytes(32)
key_hex = key.hex()

print("HMAC Key Generated : ")
print(f"\nKey (hex): {key_hex}")
print("\nTo use this key, edit your ~/.bashrc and set it as an environment variable:")
print(f"export T2YLLM_HMAC_KEY={key_hex}")
