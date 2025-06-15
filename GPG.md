#**How to Verify GPG signatures :**

t2yLLM binaries are signed with GPG. You can verify their authenticity using Saga9103 public key.
All piper and espeak binaries on this repo were downloaded from https://github.com/rhasspy/piper

##**1 - import the public key**

curl -o public-key.asc https://raw.githubusercontent.com/Saga9103/t2yLLM/main/public-key.asc
gpg --import public-key.asc

##**2 - Check GPG signatures**

 - Binaries can be found in the t2yLLM/config/piper directory

Example for the piper binary :
  - gpg --verify piper.asc piper
  - gpg --verify en_US-arctic-medium.onnx.asc en_US-arctic-medium.onnx

- If the signature is valid you should expect something looking like that :
  - gpg: Signature made [date]
  - gpg: using RSA key [EF4D6D6A603A3A5A]
  - gpg: Good signature from "[Saga9103] saga9103@users.noreply.github.com"

- you can also check with sha256sum -c Checksums.txt to verify all checksums
