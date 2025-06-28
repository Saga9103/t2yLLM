
This is just a copy of the structure of Faster-Whisper file tree to get to utils.py.

If you want to use this model in order to limit vram usage **edit Faster-Whisper utils.py just like here** if you want to add models under custom names. <br>
For example here, "quantized" does not exists in the base file.<br>
This is located in the directory where you installed Faster-Whisper with git clone (./faster-whisper/faster_whisper/utils.py).
You can also load your models from a local dir using : model = faster_whisper.WhisperModel("whisper-large-v3-ct2")
instructions are on their github repo.
