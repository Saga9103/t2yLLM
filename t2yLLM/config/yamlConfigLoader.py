import os
from dataclasses import dataclass
import yaml
from dacite import from_dict


# ALL LLM RELATED STUFF
# beware of TypeError
@dataclass
class ChatNetwork:
    RCV_CMD_IP: str  # = field(default="0.0.0.0")
    RCV_CMD_PORT: int
    SEND_DISPATCH_PORT: int
    SEND_PORT: int
    RCV_AUDIO_PORT: int
    BUFFER_SIZE: int
    AUTHORIZED_IPS: list
    NET_ADDR: str
    RASPI_ADDR: str
    hmac_enabled: bool
    domain: str
    local_net_broadcast: bool


@dataclass
class ChatVllm:
    model: str
    enable_chunked_prefill: bool
    quantization: str
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    block_size: int
    max_num_seqs: int
    device: str
    task: str
    multi_step_stream_outputs: bool

    def to_llm_args(self):
        """Return the args formatted for vLLM LLM."""
        return dict(
            model=self.model,
            enable_chunked_prefill=self.enable_chunked_prefill,
            quantization=self.quantization,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
            block_size=self.block_size,
            max_num_seqs=self.max_num_seqs,
            device=self.device,
            task=self.task,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            dtype="float16",
        )


@dataclass
class ChatLLM:
    sentence_embedder: str
    vllm_chat: ChatVllm
    spacy_model: str
    spacy_model_en: str
    searcher_max_tokens: int
    summarizer_max_tokens: int
    memmorizer_max_tokens: int
    pokemoner_max_tokens: int
    json_generator_max_tokens: int
    msg_queue_size: int


@dataclass
class ChatLOC:
    default_location: str
    default_latitude: float
    default_longitude: float
    default_timezone: str


@dataclass
class ChatWeather:
    max_forecast_days: int
    api_timeout: int


@dataclass
class ChatPika:
    pokemon_list_fr: str
    pokemon_list_en: str
    pokemon_phonetics: str
    pokemon_find_threshold: int


@dataclass
class ChatDB:
    knowledge_graph_path: str
    mem_path: str
    chromadb_path: str
    ltm_collection_name: str
    ltm_results_limit: int
    importance: float
    stm_history_limit: int


@dataclass
class ChatWiki:
    wiki_sentence_search_nb: int
    reponse_timeout: int
    summary_size: int


@dataclass
class ChatCommon:
    lang: str
    model_name: str
    unprivileged_user: str
    activate_memory: bool
    web_enabled: bool


@dataclass
class ChatPlugins:
    plugins_dict: dict
    enabled_plugins: list


# main
@dataclass
class ChatConfig:
    network: ChatNetwork
    llms: ChatLLM
    location: ChatLOC
    weather: ChatWeather
    pokemon: ChatPika
    databases: ChatDB
    wikipedia: ChatWiki
    general: ChatCommon
    plugins: ChatPlugins


# All related to audio server config:
@dataclass
class WhispCommon:
    lang: str


@dataclass
class WhispModel:
    whisper_model: str
    threads: int
    keyword: str
    kw_model: str
    phonetic_variants: list
    piper_path: str
    piper_fr_voice_path: str
    piper_en_voice_path: str
    tmpfs_dir: str
    tts_engine: str
    porcupine_path: str
    porcupine_keyword_path: str


@dataclass
class WhispAudio:
    sample_rate: int
    chunk_size: int
    buffer_time: float
    min_cmd_length: float
    min_audio_level: int
    silence_threshold: int
    activity_threshold: int
    keyword_sensitivity: float
    client_audio_window: float
    EOS: float  # in seconds
    virtual_mic_name: str  # from the /etc/asound.conf config also load the alsa module
    stream_buffer_size: int
    channels: int
    period_size: int
    ignore_own: float  # ignore own audio for loopbacks
    format: str


@dataclass
class WhispNET:
    LISTEN_IP: str  # = field(default="0.0.0.0")
    RPI_IP: str
    AUTHORIZED_IPS: list  # = field(default=["127.0.0.1"])
    SEND_RPI_PORT: int
    LISTEN_RPI_PORT: int
    SEND_CHAT_PORT: int
    RCV_END_SIGNAL: int
    SEND_CHAT_COMPLETION: int
    STATUS_REQUEST_PORT: int
    RCV_CHAT_CMD_PORT: int
    MAX_UDP_SIZE: int
    CHAT_ADDR: str
    rcv_buffer_size: int
    server_reset_delay: int
    hmac_enabled: bool


@dataclass
class WhispConfig:
    general: WhispCommon
    model: WhispModel
    audio: WhispAudio
    network: WhispNET


# All classes related to vision
@dataclass
class VizCommon:
    device: str


@dataclass
class VizNetwork:
    LISTEN_ADDR: str
    RPI_IP: str
    RCV_IMG_PORT: int
    WEB_BROWSER_PORT: int
    MAX_UDP_SIZE: int
    BUFFER_SIZE: int
    PAN_TILT_LST_PORT: int


@dataclass
class VizDepth:
    model: str


@dataclass
class VizPanTilt:
    config_file: str


@dataclass
class VizSegment:
    model: str


@dataclass
class VizSensors:
    aceinna: str
    camera_height: int
    camera_width: int
    jpg_quality: int
    arduino: str


@dataclass
class VizConfig:
    general: VizCommon
    network: VizNetwork
    depth: VizDepth
    pantilt: VizPanTilt
    segmentation: VizSegment
    sensors: VizSensors


class Loader:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "server_config.yaml")
        with open(config_path, "r") as file:
            self.confData = yaml.safe_load(file)

    def loadChatConfig(self):
        return from_dict(data_class=ChatConfig, data=self.confData["chat"])

    def loadWhispConfig(self):
        return from_dict(data_class=WhispConfig, data=self.confData["faster_whisper"])

    def loadVizConfig(self):
        return from_dict(data_class=VizConfig, data=self.confData["vision"])
