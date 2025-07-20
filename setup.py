from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="t2yLLM",
    version="0.1.0",
    author="Saga9103",
    description="An open source Voice Assistant",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Saga9103/t2yLLM",
    license="MIT",
    include_package_data=True,
    package_data={
        "t2yLLM.config": ["piper/**/*", "piper/**/**/*"],  # All files in t2yLLM/config/piper
    },
    packages=find_packages(
        exclude=(
            "examples",
            "faster-whisper",
            "faster-whisper/*",
            "Raspberry",
        )
    ),
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "requests",
        "auto_mix_prep>=0.2.0",
        "chromadb>=0.6.3",
        "config>=0.5.1",
        "datasets>=3.5.0",
        "faster_whisper>=1.1.0",
        "huggingface_hub>=0.30.2",
        "json_repair>=0.30.3",
        "matplotlib>=3.10.3",
        "numpy>=2.2.6",
        "pvporcupine>=3.0.5",
        "pyalsaaudio>=0.11.0",
        "pyttsx3>=2.98",
        "pytz>=2025.1",
        "rapidfuzz>=3.13.0",
        "requests>=2.32.3",
        "scipy>=1.15.3",
        "sentence_transformers>=3.4.1",
        "silero_vad>=5.1.2",
        "sounddevice>=0.5.1",
        "soundfile>=0.13.1",
        "spacy>=3.8.4",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers>=4.51.1",
        "tokenizers>=0.21.1",
        "triton",
        "vllm>=0.5.0",
        "wikipedia>=1.4.0",
        "dacite>=1.9.2",
        "PyYAML>=6.0.2",
        "keybert",
        "pydantic",
        "fastapi[standard]",
        "pyaudio",
        "onnxruntime",
        "netifaces",
        "cryptography",
    ],
)
