"""Main spotiy plugin for song playback and voice commands"""

import re
import os
import time
import subprocess
import threading
from typing import List, Dict, Any, Optional
import json
from vllm import SamplingParams
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from .pluginManager import APIBase, logger
from t2yLLM.plugins.injections import PluginInjector


class SpotifyAPI(APIBase, PluginInjector):
    """
    class for spotify playback :
    - device is emulated by librespot
    - calls are made by spotipy (spotify web API under the hood) to the device
    """

    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    schema = {
        "action": None,
        "query": None,
        "type": None,
        "volume": None,
        "params": {},
        "secondary_actions": [],
    }

    examples = [
        (
            "mets moi du rock sur spotify",
            {
                "action": "play",
                "type": "genre",
                "query": "rock",
                "volume": None,
                "params": {"shuffle": False},
                "secondary_actions": [],
            },
        ),
        (
            "mets le volume à fond",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": 100,
                "params": {},
                "secondary_actions": [],
            },
        ),
        (
            "mets la lecture aléatoire",
            {
                "action": "shuffle",
                "type": "player",
                "query": None,
                "volume": None,
                "params": {"state": True},
                "secondary_actions": [],
            },
        ),
        (
            "mets moi un album de ARTISTE au hasard",
            {
                "action": "play",
                "type": "artist",
                "query": "ARTISTE",
                "volume": None,
                "params": {"random_album": True},
                "secondary_actions": [],
            },
        ),
        (
            "augmente le volume de 10%",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": None,  # valeur calculée par volume_context()
                "params": {"delta_pct": 10},  # facultatif : info mémo
                "secondary_actions": [],
            },
        ),
        (
            "diminue le volume de 20%",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": None,
                "params": {"delta_pct": -20},
                "secondary_actions": [],
            },
        ),
        (
            "play some rock on Spotify",
            {
                "action": "play",
                "type": "genre",
                "query": "rock",
                "volume": None,
                "params": {"shuffle": False},
                "secondary_actions": [],
            },
        ),
        (
            "set the volume to maximum",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": 100,
                "params": {},
                "secondary_actions": [],
            },
        ),
        (
            "enable shuffle",
            {
                "action": "shuffle",
                "type": "player",
                "query": None,
                "volume": None,
                "params": {"state": True},
                "secondary_actions": [],
            },
        ),
        (
            "play a random album by ARTISTE",
            {
                "action": "play",
                "type": "artist",
                "query": "ARTISTE",
                "volume": None,
                "params": {"random_album": True},
                "secondary_actions": [],
            },
        ),
        (
            "increase the volume by 10%",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": None,
                "params": {"delta_pct": 10},
                "secondary_actions": [],
            },
        ),
        (
            "decrease the volume by 20%",
            {
                "action": "volume",
                "type": "device",
                "query": None,
                "volume": None,
                "params": {"delta_pct": -20},
                "secondary_actions": [],
            },
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = self.config.general.lang
        self.query = False
        self.activate_memory = False
        self.silent_execution = True
        self.device_name = "t2yLLM"
        self.cache_dir = os.path.expanduser("~/.cache/librespot")
        self.token_cache = os.path.expanduser("~/.spotify_token_cache")
        self.librespot_proc = None
        self.sp = None
        self.device_id = None
        self.current_context = None
        self.last_command = None
        self.last_result = {"success": False, "message": "", "data": {}}
        self.init_thread = threading.Thread(target=self.setup, daemon=False)
        self.init_thread.start()

        self.command_patterns = {
            "fr": {
                "play": r"joue|lance|mets?|démarre|écoute|play|met(?:tre)?",
                "pause": r"pause|arrête|stop(?!pe)|met(?:s|tre)?\s+(?:en\s+)?pause",
                "resume": r"reprends?|continue|relance|remets|play",
                "next": r"suivant|prochain|après|skip|passe|chanson\s+suivante|titre\s+suivant",
                "previous": r"précédent|avant|dernier|retour|chanson\s+précédente|titre\s+précédent",
                "volume": r"volume|son",
                "volume_up": r"plus\s+fort|monte\s+(?:le\s+)?(?:son|volume)|augmente\s+(?:le\s+)?(?:son|volume)|fort|monter\s+(?:le\s+)?(?:son|volume)",
                "volume_down": r"moins\s+fort|baisse\s+(?:le\s+)?(?:son|volume)|diminue\s+(?:le\s+)?(?:son|volume)|doucement|baisser\s+(?:le\s+)?(?:son|volume)|descend(?:s|re)?\s+(?:le\s+)?(?:son|volume)",
                "too_loud": r"trop\s+fort|trop\s+bruyant|ça\s+casse\s+les\s+oreilles|assourdissant",
                "too_quiet": r"(?:pas\s+assez|trop)\s+(?:bas|faible)|(?:j'|on\s+)?entends?\s+(?:rien|pas)|inaudible",
                "shuffle": r"aléatoire|mélange|shuffle|hasard|random",
                "repeat": r"répète|boucle|repeat|encore",
                "repeat_one": r"répète\s+(?:ce|le)\s+(?:morceau|titre)|boucle\s+(?:ce|le)\s+(?:morceau|titre)",
                "continuous": r"continu|suite|enchaîne|joue\s+(?:tout|en\s+continu)",
                "search": r"cherche|trouve|recherche",
                "artist": r"artiste|chanteur|groupe|band",
                "album": r"album|disque",
                "track": r"chanson|titre|morceau|musique|track",
                "playlist": r"playlist|liste",
                "genre": r"genre|style|type",
                "year": r"année|sorti en",
                "similar": r"similaire|comme|ressemble|dans le style",
                "queue": r"file|queue|ensuite|après ça",
                "volume_up_pct": r"(augmente|monte).+?de\s+(\d+)\s*%",
            },
            "en": {
                "play": r"play|start|put on|listen",
                "pause": r"pause|stop",
                "resume": r"resume|continue|unpause|play again",
                "next": r"next|skip|forward",
                "previous": r"previous|back|last",
                "volume": r"volume|sound",
                "volume_up": r"louder|turn\s+up|increase\s+(?:the\s+)?(?:volume|sound)|raise",
                "volume_down": r"quieter|turn\s+down|decrease\s+(?:the\s+)?(?:volume|sound)|lower",
                "too_loud": r"too\s+loud|deafening|hurts\s+my\s+ears",
                "too_quiet": r"too\s+(?:quiet|low)|can't\s+hear|inaudible",
                "shuffle": r"shuffle|random|mix",
                "repeat": r"repeat|loop",
                "repeat_one": r"repeat\s+(?:this|the)\s+(?:song|track)",
                "continuous": r"continuous|keep\s+playing|play\s+all",
                "current": r"currently|now playing|what's playing",
                "search": r"search|find|look for",
                "artist": r"artist|singer|band|group",
                "album": r"album|record",
                "track": r"song|track|music",
                "playlist": r"playlist|list",
                "genre": r"genre|style|type",
                "year": r"year|released in",
                "similar": r"similar|like|sounds like",
                "queue": r"queue|next up|coming up",
                "volume_down_pct": r"(diminue|baisse|descend).+?de\s+(\d+)\s*%",
            },
        }

        self.music_keywords = {
            "fr": [
                "musique",
                "chanson",
                "spotify",
                "écoute",
                "joue",
                "morceau",
                "artiste",
                "album",
                "playlist",
                "son",
                "volume",
                "pause",
                "suivant",
                "précédent",
                "arrête",
                "lance",
                "mets",
                "fort",
                "doucement",
                "boucle",
                "répète",
                "continue",
                "enchaîne",
                "monte",
                "baisse",
                "augmente",
                "diminue",
                "plus",
                "moins",
                "aléatoire",
                "shuffle",
                "skip",
                "titre",
            ],
            "en": [
                "music",
                "song",
                "spotify",
                "listen",
                "play",
                "track",
                "artist",
                "album",
                "playlist",
                "sound",
                "volume",
                "pause",
                "next",
                "previous",
                "stop",
                "start",
                "put",
                "loud",
                "quiet",
                "repeat",
                "loop",
                "continue",
                "shuffle",
                "turn",
                "increase",
                "decrease",
                "raise",
                "lower",
            ],
        }

    def spoticheck(self):
        """Check if Spotify app is already running else launches it. t2yLLM
        device will be emulated in it and will be used as soon as
        your order the program to play a song"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "spotify"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.info("Spotify is not running, launching it")
                spotify_paths = [
                    "/usr/bin/spotify",
                    "/usr/local/bin/spotify",
                ]
                spotify_launched = False

                for spotify_path in spotify_paths:
                    try:
                        check_result = subprocess.run(
                            ["which", spotify_path],
                            capture_output=True,
                            text=True,
                            check=False,
                        )

                        if check_result.returncode == 0:
                            actual_path = check_result.stdout.strip()
                            subprocess.Popen(
                                [actual_path],
                                start_new_session=True,
                            )
                            logger.info(f"Spotify app launched from {actual_path}")
                            spotify_launched = True
                            time.sleep(5)
                            break
                    except Exception as e:
                        logger.debug(f"Failed to launch from {spotify_path}: {e}")
                        continue

                if not spotify_launched:
                    logger.error(
                        "Spotify app not found. Please install it with: "
                        "sudo snap install spotify or sudo apt install spotify-client"
                    )
            else:
                logger.info("Spotify app is already running")
        except Exception as e:
            logger.error(f"Error checking/launching Spotify app: {e}")

    def setup(self):
        """Initialize Spotify pipeline:
        - is the spotify launched
        - sets up emulated device with librespot
        - connects via API and connects to device of self.device_name"""
        try:
            self.spoticheck()
            self.start_librespot()
            time.sleep(7)
            self.connect()
            self.get_device()

            logger.info(
                f"Spotify plugin initialized successfully on device {self.device_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Spotify: {e}")

    def start_librespot(self):
        """Starts librespot subprocess with Alsa backend
        so we can control the volume
        else it fails for now"""
        librespot_path = os.path.expanduser("~/.cargo/bin/librespot")
        if not os.path.exists(librespot_path):
            logger.error(
                "librespot not found. Please follow : https://github.com/librespot-org/librespot"
            )
            return

        try:
            self.librespot_proc = subprocess.Popen(
                [
                    librespot_path,
                    "--name",
                    self.device_name,
                    "--bitrate",
                    "320",
                    "--cache",
                    self.cache_dir,
                    "--device-type",
                    "speaker",
                    "--backend",
                    "alsa",
                    "--initial-volume",
                    "50",
                    "--volume-ctrl",
                    "linear",
                    "--enable-volume-normalisation",
                    "--normalisation-pregain",
                    "-10",
                    "--autoplay",  # not really working...
                    "on",
                ],
            )
            logger.info(f"Started librespot as '{self.device_name}'")
        except Exception as e:
            logger.error(f"Failed to start librespot: {e}")

    def set_alsa_volume(self, volume_percent: int) -> bool:
        """Control system volume with either PulseAudio or ALSA"""
        try:
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume_percent}%"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"System volume set to {volume_percent}% via PulseAudio")
                return True

            result = subprocess.run(
                ["amixer", "-D", "pulse", "sset", "Master", f"{volume_percent}%"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"System volume set to {volume_percent}% via ALSA-Pulse")
                return True

            for control in ["Master", "PCM", "Speaker"]:
                result = subprocess.run(
                    ["amixer", "sset", control, f"{volume_percent}%"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    logger.info(f"ALSA {control} volume set to {volume_percent}%")
                    return True

            logger.error("Failed to set system volume with any method")
            return False

        except Exception as e:
            logger.error(f"Error setting system volume: {e}")
            return False

    def connect(self):
        """Initialize Spotify API connection"""
        required_vars = ["SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET"]
        for var in required_vars:
            if not os.environ.get(var):
                logger.error(
                    f"{var} environment variable not set, please edit your ~/.bashrc"
                )
                return

        try:
            # https://developer.spotify.com/documentation/web-api/concepts/scopes
            self.sp = spotipy.Spotify(
                auth_manager=SpotifyOAuth(
                    client_id=os.environ["SPOTIPY_CLIENT_ID"],
                    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
                    redirect_uri="http://127.0.0.1:8888/callback",
                    scope="user-read-playback-state user-modify-playback-state "
                    "user-read-currently-playing streaming user-read-private "
                    "user-read-email user-library-read playlist-read-private user-top-read",
                    cache_path=self.token_cache,
                    open_browser=True,  # only for the first loggin, links your account
                )
            )
            user = self.sp.current_user()
            logger.info(
                f"Connected to Spotify as display : {
                    user.get('display_name', 'Unknown')
                }"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Spotify API: {e}")

    def get_device(self, retry_count=3):
        """Attempts to find the librespot device of device_name"""
        if not self.sp:
            return False

        for attempt in range(retry_count):
            try:
                devices = self.sp.devices()
                for d in devices["devices"]:
                    if d["name"].lower() == self.device_name.lower():
                        self.device_id = d["id"]
                        logger.info(f"Spotify device: {self.device_name}")
                        return True

                if attempt < retry_count - 1:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Error finding device for Spotify : {e}")

        logger.warning(f"Spotify device '{self.device_name}' not found")
        return False

    def refresh_device(self) -> bool:
        """Refresh device and ensure it's active and prevents sleep"""
        try:
            devices = self.sp.devices()
            for d in devices["devices"]:
                if d["name"].lower() == self.device_name.lower():
                    self.device_id = d["id"]
                    if not d["is_active"]:
                        self.sp.transfer_playback(self.device_id, force_play=False)
                        time.sleep(0.5)
                    return True

            return self.get_device()
        except Exception as e:
            logger.debug(f"Failed to refresh device: {e}")
        return False

    @property
    def name(self) -> str:
        return "SpotifyAPI"

    @property
    def filename(self) -> str:
        return "spotify"

    @property
    def is_enabled(self) -> bool:
        return self.name in self.config.plugins.enabled_plugins

    @property
    def memory(self) -> bool:
        return self.activate_memory

    def is_query(self, user_input: str) -> bool:
        user_input_lower = user_input.lower()
        lang = self.language

        for keyword in self.music_keywords.get(lang, self.music_keywords["en"]):
            if keyword in user_input_lower:
                self.query = True
                return True

        for pattern in self.command_patterns[lang].values():
            if re.search(pattern, user_input_lower):
                self.query = True
                return True

        self.query = False
        return False

    async def convert(self, user_input, llm, tokenizer, request_id=None):
        if self.language == "fr":
            sys_prompt = """Tu es SpotifyCommandFormatter.\n
                Ta réponse DOIT être un JSON unique, sans rien d'autre
                et respectant la clé EXACTE :
                {"action","query","type","volume","params","secondary_actions"}.\n
                Valeurs attendues :\n
                action : play | pause | resume | next | previous | shuffle | repeat | volume | add_to_queue\n
                type   : track | artist | album | playlist | genre | device | player\n
                query  : chaîne décrivant ce qu'il faut jouer (ou null).\n
                volume : 0-100 ou null.\n
                params : dict optionnel pour détails (ex. {'shuffle':true}).\n
                secondary_actions : liste de commandes du même format
                (souvent vide).\n
                Respecte à la lettre la casse des clés.\n
                Ne mets jamais de texte libre autour du JSON.\n"""
        else:
            sys_prompt = """You are SpotifyCommandFormatter.\n
                        Your response MUST be a single JSON, with nothing else
                        and respecting the EXACT keys:
                        {"action","query","type","volume","params","secondary_actions"}.\n
                        Expected values:\n
                        action : play | pause | resume | next | previous | shuffle | repeat | volume | add_to_queue\n
                        type   : track | artist | album | playlist | genre | device | player\n
                        query  : string describing what to play (or null).\n
                        volume : 0-100 or null.\n
                        params : optional dict for details (e.g. {'shuffle':true}).\n
                        secondary_actions : list of commands in the same format
                        (often empty).\n
                        Respect the key names' capitalization exactly.\n
                        Never put any free text around the JSON.\n"""

        msgs = [{"role": "system", "content": sys_prompt}]
        for u, js in self.examples:
            msgs += [
                {"role": "user", "content": u},
                {"role": "assistant", "content": json.dumps(js, ensure_ascii=False)},
            ]
        msgs.append({"role": "user", "content": user_input})

        params = SamplingParams(max_tokens=128, temperature=0.0, top_p=0.8)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        gen = llm.generate(prompt=prompt, sampling_params=params, request_id=request_id)
        async for out in gen:
            if out.finished:
                break
        try:
            data = json.loads(out.outputs[0].text.strip())
            full = self.schema | data
            return full
        except Exception:
            logger.warning("Injection JSON invalide")
            return {}

    def volume_context(self, user_input: str) -> Optional[int]:
        """Gets user intent about volume"""
        user_input_lower = user_input.lower()
        lang = self.language
        patterns = self.command_patterns[lang]
        volume_match = re.search(r"(\d+)\s*%?", user_input_lower)
        if volume_match:
            if any(word in user_input_lower for word in ["volume", "son", "à"]):
                return min(100, max(0, int(volume_match.group(1))))
        if re.search(patterns.get("too_loud", ""), user_input_lower):
            return 30
        if re.search(patterns.get("too_quiet", ""), user_input_lower):
            return 70

        rel_up = re.search(
            r"(?:augmente|monte|increase|turn\s+up).*?(\d+)\s*%", user_input_lower
        )
        rel_down = re.search(
            r"(?:diminue|baisse|descend|decrease|turn\s+down).*?(\d+)\s*%",
            user_input_lower,
        )

        current = self.current_volume()
        if current is None:
            current = 50

        if rel_up:
            delta = int(rel_up.group(1))
            return min(100, current + delta)
        if rel_down:
            delta = int(rel_down.group(1))
            return max(0, current - delta)

        if re.search(patterns.get("volume_up", ""), user_input_lower):
            if any(
                word in user_input_lower
                for word in [
                    "très",
                    "beaucoup",
                    "bien plus",
                    "vraiment",
                    "much",
                    "lot",
                    "really",
                ]
            ):
                return min(100, current + 30)
            return min(100, current + 20)

        if re.search(patterns.get("volume_down", ""), user_input_lower):
            if any(
                word in user_input_lower
                for word in [
                    "très",
                    "beaucoup",
                    "bien moins",
                    "vraiment",
                    "much",
                    "lot",
                    "really",
                ]
            ):
                return max(0, current - 30)
            return max(0, current - 20)

        if self.language == "fr":
            if any(
                word in user_input_lower
                for word in ["dormir", "nuit", "coucher", "dodo"]
            ):
                return 20
            if any(
                word in user_input_lower
                for word in ["calme", "doux", "doucement", "tranquille"]
            ):
                return 30
            if any(
                word in user_input_lower
                for word in ["fête", "party", "ambiance", "danse", "danser"]
            ):
                return 80
            if any(
                word in user_input_lower
                for word in ["travail", "concentration", "étude", "bosser"]
            ):
                return 40
        else:
            if any(
                word in user_input_lower
                for word in ["sleep", "night", "bed", "bedtime"]
            ):
                return 20
            if any(
                word in user_input_lower
                for word in ["quiet", "calm", "soft", "peaceful"]
            ):
                return 30
            if any(
                word in user_input_lower
                for word in ["party", "loud", "dance", "dancing"]
            ):
                return 80
            if any(
                word in user_input_lower
                for word in ["work", "focus", "study", "concentrate"]
            ):
                return 40
        return None

    def current_volume(self) -> Optional[int]:
        """Gets current volume from active device"""
        if not self.sp:
            logger.warning("Could not set volume because no device was detected")
            return None
        try:
            playback = self.sp.current_playback()
            if playback and playback.get("device"):
                return playback["device"]["volume_percent"]
        except Exception as e:
            logger.debug(f"Error getting current volume: {e}")
        return None

    def get_alsa_mixers(self) -> List[str]:
        """Debugging method : Get available ALSA mixer controls"""
        try:
            result = subprocess.run(
                ["amixer", "scontrols"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                mixers = []
                for line in result.stdout.split("\n"):
                    if line.strip():
                        match = re.search(r"'([^']+)'", line)
                        if match:
                            mixers.append(match.group(1))
                return mixers
        except Exception as e:
            logger.error(f"Error getting ALSA mixers: {e}")
        return []

    def set_volume(self, volume_percent: int) -> bool:
        """Set volume - try Spotify API first, then system volume"""
        if not self.sp or not self.device_id:
            logger.warning("Spotify client or device not initialized")
            return False
        success = False
        try:
            if self.refresh_device():
                self.sp.volume(volume_percent, device_id=self.device_id)
                logger.info(f"Spotify API volume set to {volume_percent}%")
                success = True
                return True
        except Exception as e:
            logger.warning(
                f"Spotify API volume control failed: {e}, trying system volume"
            )
        if self.set_alsa_volume(volume_percent):
            success = True

        return success

    def delim_query(self, user_input: str):
        query = user_input
        volume_indicators = ["volume", "son", "fort", "doucement", "%"]
        if any(indicator in query.lower() for indicator in volume_indicators):
            return ""
        command_keywords = [
            r"\b(play|launch|joue|lance|démarre|spotify|cherche|trouve|écoute)\b",
            r"\bsur\s+spotify\b",
            r"\bon\s+spotify\b",
        ]
        for pattern in command_keywords:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)
        query = re.sub(r"\s+", " ", query).strip()

        return query

    def __call__(self, user_input: str) -> Dict[str, Any]:
        """User command checking with intent detection"""
        user_input_lower = user_input.lower()
        lang = self.language
        patterns = self.command_patterns[lang]

        command = {
            "action": None,
            "query": None,
            "type": None,
            "volume": None,
            "params": {},
            "secondary_actions": [],
        }

        volume_keywords = {
            "fr": [
                "fort",
                "volume",
                "son",
                "doucement",
                "bruyant",
                "faible",
                "bas",
                "haut",
            ],
            "en": ["loud", "volume", "sound", "quiet", "louder", "quieter"],
        }

        has_volume_intent = any(
            keyword in user_input_lower for keyword in volume_keywords.get(lang, [])
        )

        if has_volume_intent:
            if re.search(patterns.get("too_loud", ""), user_input_lower):
                command["action"] = "volume"
                command["volume"] = self.volume_context(user_input)
                return command

            if re.search(patterns.get("too_quiet", ""), user_input_lower):
                command["action"] = "volume"
                command["volume"] = self.volume_context(user_input)
                return command

            if re.search(patterns.get("volume_up", ""), user_input_lower):
                command["action"] = "volume"
                command["volume"] = self.volume_context(user_input)
                return command

            if re.search(patterns.get("volume_down", ""), user_input_lower):
                command["action"] = "volume"
                command["volume"] = self.volume_context(user_input)
                return command

            volume_match = re.search(r"(\d+)\s*%?", user_input_lower)
            if volume_match and re.search(patterns.get("volume", ""), user_input_lower):
                command["action"] = "volume"
                command["volume"] = min(100, max(0, int(volume_match.group(1))))
                return command

        control_actions = [
            "pause",
            "next",
            "previous",
            "shuffle",
            "repeat",
            "repeat_one",
            "continuous",
        ]

        for action in control_actions:
            if re.search(patterns.get(action, ""), user_input_lower):
                if action in ["repeat_one", "continuous"]:
                    command["action"] = "repeat"
                    command["params"]["mode"] = action
                else:
                    command["action"] = action
                return command

        if re.search(
            r"\b(reprends?|continue|relance|remets)\b", user_input_lower
        ) and not re.search(
            r"\b(chanson|titre|morceau|musique|artiste|album)\b", user_input_lower
        ):
            command["action"] = "resume"
            return command

        if re.search(patterns.get("play", ""), user_input_lower) or re.search(
            patterns.get("search", ""), user_input_lower
        ):
            query = self.delim_query(user_input)
            if query and len(query) > 2:
                command["action"] = "play"
                command["query"] = query

                if re.search(patterns.get("artist", ""), user_input_lower):
                    command["type"] = "artist"
                elif re.search(patterns.get("album", ""), user_input_lower):
                    command["type"] = "album"
                elif re.search(patterns.get("playlist", ""), user_input_lower):
                    command["type"] = "playlist"
                else:
                    command["type"] = "track"

                if re.search(patterns.get("shuffle", ""), user_input_lower):
                    command["secondary_actions"].append(
                        {"action": "shuffle", "value": True}
                    )
                if re.search(patterns.get("continuous", ""), user_input_lower):
                    command["secondary_actions"].append(
                        {"action": "repeat", "value": "context"}
                    )

                return command

        if not command["action"] and any(
            word in user_input_lower
            for word in ["spotify", "musique", "chanson", "music", "song"]
        ):
            query = self.delim_query(user_input)
            if query:
                command["action"] = "search"
                command["query"] = query
                command["type"] = "track"

        return command

    def get_content(self, query: str, search_type: str = "track") -> Optional[Dict]:
        """Search content on Spotify based on query type"""
        if not self.sp or not query:
            return None

        try:
            results = self.sp.search(q=query, type=search_type, limit=5)

            if search_type == "track" and results["tracks"]["items"]:
                track = results["tracks"]["items"][0]
                return {
                    "type": "track",
                    "uri": track["uri"],
                    "name": track["name"],
                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                    "album": track["album"]["name"],
                }
            if search_type == "artist" and results["artists"]["items"]:
                artist = results["artists"]["items"][0]
                top_tracks = self.sp.artist_top_tracks(artist["id"])
                if top_tracks["tracks"]:
                    return {
                        "type": "artist",
                        "uri": artist["uri"],
                        "name": artist["name"],
                        "tracks": [t["uri"] for t in top_tracks["tracks"][:10]],
                    }
            if search_type == "album" and results["albums"]["items"]:
                album = results["albums"]["items"][0]
                return {
                    "type": "album",
                    "uri": album["uri"],
                    "name": album["name"],
                    "artist": ", ".join([a["name"] for a in album["artists"]]),
                }
            if search_type == "playlist" and results["playlists"]["items"]:
                playlist = results["playlists"]["items"][0]
                return {
                    "type": "playlist",
                    "uri": playlist["uri"],
                    "name": playlist["name"],
                    "owner": playlist["owner"]["display_name"],
                }

        except Exception as e:
            logger.error(f"Error searching Spotify: {e}")

        return None

    def play_content(self, content: Dict) -> bool:
        """Plays content on Spotify"""
        if not self.sp:
            return False
        if not self.device_id or not self.refresh_device():
            logger.error("No active device found")
            return False

        try:
            if content["type"] == "track":
                self.sp.start_playback(device_id=self.device_id, uris=[content["uri"]])
            elif content["type"] == "artist":
                self.sp.start_playback(device_id=self.device_id, uris=content["tracks"])
            elif content["type"] in ["album", "playlist"]:
                self.sp.start_playback(
                    device_id=self.device_id, context_uri=content["uri"]
                )

            self.current_context = content
            return True

        except Exception as e:
            logger.error(f"Error playing content: {e}")
            if "Device not found" in str(e) or "NO_ACTIVE_DEVICE" in str(e):
                if self.refresh_device():
                    return self.play_content(content)
            return False

    def get_current_track(self) -> Optional[Dict]:
        """Gets current track information"""
        if not self.sp:
            return None

        try:
            current = self.sp.current_playback()
            if current and current["item"]:
                track = current["item"]
                return {
                    "name": track["name"],
                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                    "album": track["album"]["name"],
                    "progress": current["progress_ms"] // 1000,
                    "duration": track["duration_ms"] // 1000,
                    "is_playing": current["is_playing"],
                    "shuffle": current.get("shuffle_state", False),
                    "repeat": current.get("repeat_state", "off"),
                }
        except Exception as e:
            logger.error(f"Error getting current track: {e}")

        return None

    def ensure_device_active(self) -> bool:
        """Ensure the device is active before sending commands"""
        if not self.refresh_device():
            return False
        try:
            playback = self.sp.current_playback()
            if not playback or not playback.get("device"):
                logger.info("No active playback, starting default playlist")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking playback state: {e}")
            return False

    def search(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Main search/control method with improved error handling
        so here spotipy makes the API call easier
        https://spotipy.readthedocs.io/en/latest/#module-spotipy.client
        that links API calls from spotify dev"""
        try:
            command_override: Dict[str, Any] | None = kwargs.get("command_dict")
            self.last_command = user_input
            command = command_override if command_override else self(user_input)
            result = {"success": False, "message": "", "data": {}}

            if not self.sp:
                result["message"] = "Spotify not connected"
                self.last_result = result
                return result

            if command["action"] == "play" and command["query"]:
                content = self.get_content(command["query"], command["type"])
                if content:
                    if self.play_content(content):
                        result["success"] = True
                        result["data"] = content

                        for secondary in command.get("secondary_actions", []):
                            if secondary["action"] == "shuffle":
                                self.sp.shuffle(True, device_id=self.device_id)
                            elif secondary["action"] == "repeat":
                                self.sp.repeat("context", device_id=self.device_id)

                        if content["type"] == "track":
                            result["message"] = (
                                f"Playing {content['name']} by {content['artist']}"
                            )
                        elif content["type"] == "artist":
                            result["message"] = f"Playing songs by {content['name']}"
                        elif content["type"] == "album":
                            result["message"] = f"Playing album {content['name']} by {
                                content['artist']
                            }"
                        else:
                            result["message"] = f"Playing playlist {content['name']}"
                    else:
                        result["message"] = "Failed to play content"
                else:
                    result["message"] = f"No results found for: {command['query']}"

            elif command["action"] == "pause":
                try:
                    self.sp.pause_playback(device_id=self.device_id)
                    result["success"] = True
                    result["message"] = "Playback paused"
                except Exception as e:
                    if "already paused" in str(e).lower():
                        result["success"] = True
                        result["message"] = "Already paused"
                    else:
                        result["message"] = "Failed to pause"

            elif command["action"] == "resume":
                try:
                    self.sp.start_playback(device_id=self.device_id)
                    result["success"] = True
                    result["message"] = "Playback resumed"
                except Exception:
                    result["message"] = "Failed to resume"

            elif command["action"] == "next":
                if self.ensure_device_active():
                    try:
                        self.sp.next_track(device_id=self.device_id)
                        result["success"] = True
                        result["message"] = "Skipped to next track"
                    except Exception:
                        result["message"] = "Failed to skip"
                else:
                    result["message"] = "No active playback"

            elif command["action"] == "previous":
                if self.ensure_device_active():
                    try:
                        self.sp.previous_track(device_id=self.device_id)
                        result["success"] = True
                        result["message"] = "Back to previous track"
                    except Exception:
                        result["message"] = "Failed to go back"
                else:
                    result["message"] = "No active playback"

            elif command["action"] == "volume" and command["volume"] is not None:
                mixers = self.get_alsa_mixers()
                if mixers:
                    logger.debug(f"Available ALSA mixers: {mixers}")

                if self.set_volume(command["volume"]):
                    result["success"] = True
                    result["message"] = f"Volume set to {command['volume']}%"
                else:
                    result["message"] = "Failed to set volume"

            elif command["action"] == "current":
                current = self.get_current_track()
                if current:
                    result["success"] = True
                    result["data"] = current
                    status = "playing" if current["is_playing"] else "paused"
                    result["message"] = (
                        f"Currently {status}: {current['name']} by {current['artist']}"
                    )
                else:
                    result["message"] = "Nothing currently playing"

            elif command["action"] == "shuffle":
                if self.ensure_device_active():
                    try:
                        current = self.sp.current_playback()
                        new_state = not current.get("shuffle_state", False)
                        self.sp.shuffle(new_state, device_id=self.device_id)
                        result["success"] = True
                        result["message"] = f"Shuffle {'on' if new_state else 'off'}"
                    except Exception as e:
                        result["message"] = f"Failed to toggle shuffle {e}"
                else:
                    result["message"] = "No active playback"

            elif command["action"] == "repeat":
                if self.ensure_device_active():
                    try:
                        mode = command["params"].get("mode", "")
                        if mode == "repeat_one":
                            self.sp.repeat("track", device_id=self.device_id)
                            result["message"] = "Repeating current track"
                        elif mode == "continuous":
                            self.sp.repeat("context", device_id=self.device_id)
                            result["message"] = "Continuous playback enabled"
                        else:
                            current = self.sp.current_playback()
                            current_mode = current.get("repeat_state", "off")
                            next_mode = {
                                "off": "context",
                                "context": "track",
                                "track": "off",
                            }
                            self.sp.repeat(
                                next_mode[current_mode], device_id=self.device_id
                            )
                            result["message"] = (
                                f"Repeat mode: {next_mode[current_mode]}"
                            )
                        result["success"] = True
                    except Exception:
                        result["message"] = "Failed to change repeat mode"
                else:
                    result["message"] = "No active playback"

            else:
                result["message"] = "Command not understood"

            self.last_result = result
            return result

        except Exception as e:
            logger.error(f"Error in search: {e}")
            result = {"success": False, "message": str(e), "data": {}}
            self.last_result = result
            return result

    def format(self) -> str:
        """Format response for TTS"""
        result = self.last_result
        lang = self.language

        if not result["success"]:
            if lang == "fr":
                return f"Erreur dans le contrôle de Spotify : {result['message']}"
            return f"I couldn't control Spotify : {result['message']}"

        if lang == "fr":
            message_map = {
                "Playing": "Je lance",
                "Playback paused": "Musique mise en pause",
                "Already paused": "La musique est déjà en pause",
                "Playback resumed": "Lecture reprise",
                "Skipped to next track": "Morceau suivant",
                "Back to previous track": "Morceau précédent",
                "Volume set to": "Volume réglé à",
                "Shuffle on": "Lecture aléatoire activée",
                "Shuffle off": "Lecture aléatoire désactivée",
                "Currently playing": "En cours de lecture",
                "Currently paused": "En pause",
                "Nothing currently playing": "Aucune lecture en cours",
                "No active playback": "Aucune lecture active",
                "Repeating current track": "Je répète ce morceau en boucle",
                "Continuous playback enabled": "Lecture continue activée",
                "Repeat mode: track": "Mode répétition: morceau actuel",
                "Repeat mode: context": "Mode répétition: album ou playlist",
                "Repeat mode: off": "Répétition désactivée",
                "Failed to play content": "Impossible de lancer la lecture",
                "No results found for": "Aucun résultat trouvé pour",
            }

            for eng, fr in message_map.items():
                if eng in result["message"]:
                    return result["message"].replace(eng, fr)

            if "Playing songs by" in result["message"]:
                artist = result["data"].get("name", "")
                return f"Je lance les chansons de {artist}"
            if "Playing album" in result["message"]:
                album = result["data"].get("name", "")
                artist = result["data"].get("artist", "")
                return f"Je lance l'album {album} de {artist}"
            if "Playing playlist" in result["message"]:
                playlist = result["data"].get("name", "")
                return f"Je lance la playlist {playlist}"
            if "Playing" in result["message"] and result["data"].get("type") == "track":
                track = result["data"].get("name", "")
                artist = result["data"].get("artist", "")
                return f"Je lance {track} de {artist}"

        return result["message"]

    def search_terms(self, user_input: str) -> List[str]:
        terms = []
        command = self(user_input)

        if command["action"]:
            terms.append(command["action"])
        if command["type"]:
            terms.append(command["type"])
        if command["query"]:
            words = command["query"].split()
            terms.extend(words[:3])

        return terms

    def __del__(self):
        """Clean up librespot when plugin is destroyed"""
        if hasattr(self, "librespot_proc") and self.librespot_proc:
            try:
                self.librespot_proc.terminate()
                self.librespot_proc.wait(timeout=2)
            except Exception:
                if self.librespot_proc.poll() is None:
                    self.librespot_proc.kill()
            logger.info("Librespot process terminated")
