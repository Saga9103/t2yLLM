"""
module for spotify using spotipy and librespot
to play and control music on spotify app (not webbrowser)
from snap install spotify
"""

import re
import os
import time
import subprocess
import threading
from typing import List, Dict, Any, Optional
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from .pluginManager import APIBase, logger


class SpotifyAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
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
                "play": r"joue|lance|mets?|démarre|écoute",
                "pause": r"pause|arrête|stop",
                "resume": r"reprends?|continue|relance",
                "next": r"suivant|prochain|après|skip",
                "previous": r"précédent|avant|dernier",
                "volume": r"volume|son",
                "shuffle": r"aléatoire|mélange|shuffle",
                "repeat": r"répète|boucle|repeat",
                "current": r"actuellement|en cours|maintenant|joue quoi",
                "search": r"cherche|trouve|recherche",
                "artist": r"artiste|chanteur|groupe|band",
                "album": r"album|disque",
                "track": r"chanson|titre|morceau|musique|track",
                "playlist": r"playlist|liste",
                "genre": r"genre|style|type",
                "year": r"année|sorti en",
                "similar": r"similaire|comme|ressemble|dans le style",
            },
            "en": {
                "play": r"play|start|put on|listen",
                "pause": r"pause|stop",
                "resume": r"resume|continue|unpause",
                "next": r"next|skip|forward",
                "previous": r"previous|back|last",
                "volume": r"volume|sound",
                "shuffle": r"shuffle|random|mix",
                "repeat": r"repeat|loop",
                "current": r"currently|now playing|what's playing",
                "search": r"search|find|look for",
                "artist": r"artist|singer|band|group",
                "album": r"album|record",
                "track": r"song|track|music",
                "playlist": r"playlist|list",
                "genre": r"genre|style|type",
                "year": r"year|released in",
                "similar": r"similar|like|sounds like",
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
            ],
        }

    def spoticheck(self):
        """Check if Spotify app is already running else launches it"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "spotify"], capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.info("Spotify is not running, launching it")
                spotify_paths = [
                    "/usr/bin/spotify",
                    "/usr/local/bin/spotify",
                    "spotify",
                ]
                spotify_launched = False

                for spotify_path in spotify_paths:
                    try:
                        check_result = subprocess.run(
                            ["which", spotify_path], capture_output=True, text=True
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
                        "sudo apt install spotify-client"
                    )
            else:
                logger.info("Spotify app is already running")
        except Exception as e:
            logger.error(f"Error checking/launching Spotify app: {e}")

    def setup(self):
        """Initialize Spotify connection"""
        try:
            self.spoticheck()
            self.start_librespot()
            time.sleep(7)
            self.connect()
            self.get_device()

            logger.info(
                f"Spotify plugin initialized successfully with : {self.device_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Spotify: {e}")

    def start_librespot(self):
        """Starts librespot subprocess"""
        librespot_path = os.path.expanduser("~/.cargo/bin/librespot")
        if not os.path.exists(librespot_path):
            logger.error(
                "librespot not found. Please folllow : https://github.com/librespot-org/librespot"
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
                ],
            )
            logger.info(f"Started librespot as '{self.device_name}'")
        except Exception as e:
            logger.error(f"Failed to start librespot: {e}")

    def connect(self):
        """Initialize Spotify API connection"""
        required_vars = ["SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET"]
        for var in required_vars:
            if not os.environ.get(var):
                logger.error(f"{var} environment variable not set")
                return

        try:
            self.sp = spotipy.Spotify(
                auth_manager=SpotifyOAuth(
                    client_id=os.environ["SPOTIPY_CLIENT_ID"],
                    client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
                    redirect_uri="http://127.0.0.1:8888/callback",
                    scope="user-read-playback-state user-modify-playback-state "
                    "user-read-currently-playing streaming user-read-private "
                    "user-read-email user-library-read playlist-read-private",
                    cache_path=self.token_cache,
                    open_browser=True,
                )
            )
            user = self.sp.current_user()
            logger.info(
                f"Connected to Spotify as: {user.get('display_name', 'Unknown')}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Spotify API: {e}")

    def get_device(self, retry_count=3):
        """Find the librespot device"""
        if not self.sp:
            return False

        for attempt in range(retry_count):
            try:
                devices = self.sp.devices()
                for d in devices["devices"]:
                    if d["name"].lower() == self.device_name.lower():
                        self.device_id = d["id"]
                        logger.info(f"Found Spotify device: {self.device_name}")
                        return True

                if attempt < retry_count - 1:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Error finding device: {e}")

        logger.warning(f"Spotify device '{self.device_name}' not found")
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

    def __call__(self, user_input: str) -> Dict[str, Any]:
        """Parse user input to extract command and parameters"""
        user_input_lower = user_input.lower()
        lang = self.language
        patterns = self.command_patterns[lang]

        command = {
            "action": None,
            "query": None,
            "type": None,
            "volume": None,
            "params": {},
        }

        for action, pattern in patterns.items():
            if re.search(pattern, user_input_lower):
                command["action"] = action
                break

        if command["action"] in ["play", "search"]:
            if re.search(patterns.get("artist", ""), user_input_lower):
                command["type"] = "artist"
            elif re.search(patterns.get("album", ""), user_input_lower):
                command["type"] = "album"
            elif re.search(patterns.get("playlist", ""), user_input_lower):
                command["type"] = "playlist"
            else:
                command["type"] = "track"
            query = user_input
            for word_pattern in patterns.values():
                query = re.sub(word_pattern, "", query, flags=re.IGNORECASE)
            query = re.sub(r"\s+", " ", query).strip()
            if query:
                command["query"] = query
        volume_match = re.search(r"(\d+)\s*%?", user_input_lower)
        if volume_match and command["action"] == "volume":
            command["volume"] = min(100, max(0, int(volume_match.group(1))))

        return command

    def get_content(self, query: str, search_type: str = "track") -> Optional[Dict]:
        """Search content on Spotify"""
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
        """Play content on Spotify"""
        if not self.sp or not self.device_id:
            if not self.get_device():
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
            if "Device not found" in str(e):
                if self.get_device():
                    return self.play_content(content)
            return False

    def get_current_track(self) -> Optional[Dict]:
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
                }
        except Exception as e:
            logger.error(f"Error getting current track: {e}")

        return None

    def search(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Main search/control method"""
        try:
            self.last_command = user_input
            command = self(user_input)
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
                        if content["type"] == "track":
                            result["message"] = (
                                f"Playing {content['name']} by {content['artist']}"
                            )
                        elif content["type"] == "artist":
                            result["message"] = f"Playing songs by {content['name']}"
                        elif content["type"] == "album":
                            result["message"] = (
                                f"Playing album {content['name']} by {content['artist']}"
                            )
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
                except Exception:
                    result["message"] = "Failed to pause"

            elif command["action"] == "resume":
                try:
                    self.sp.start_playback(device_id=self.device_id)
                    result["success"] = True
                    result["message"] = "Playback resumed"
                except Exception:
                    result["message"] = "Failed to resume"

            elif command["action"] == "next":
                try:
                    self.sp.next_track(device_id=self.device_id)
                    result["success"] = True
                    result["message"] = "Skipped to next track"
                except Exception:
                    result["message"] = "Failed to skip"

            elif command["action"] == "previous":
                try:
                    self.sp.previous_track(device_id=self.device_id)
                    result["success"] = True
                    result["message"] = "Back to previous track"
                except Exception:
                    result["message"] = "Failed to go back"

            elif command["action"] == "volume" and command["volume"] is not None:
                try:
                    self.sp.volume(command["volume"], device_id=self.device_id)
                    result["success"] = True
                    result["message"] = f"Volume set to {command['volume']}%"
                except Exception:
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
                try:
                    current = self.sp.current_playback()
                    new_state = not current.get("shuffle_state", False)
                    self.sp.shuffle(new_state, device_id=self.device_id)
                    result["success"] = True
                    result["message"] = f"Shuffle {'on' if new_state else 'off'}"
                except Exception:
                    result["message"] = "Failed to toggle shuffle"

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
        result = self.last_result
        lang = self.language

        if not result["success"]:
            if lang == "fr":
                return f"Je n'ai pas pu contrôler Spotify: {result['message']}"
            return f"I couldn't control Spotify: {result['message']}"

        if lang == "fr":
            message_map = {
                "Playing": "Je lance",
                "Playback paused": "Musique mise en pause",
                "Playback resumed": "Lecture reprise",
                "Skipped to next track": "Morceau suivant",
                "Back to previous track": "Morceau précédent",
                "Volume set to": "Volume réglé à",
                "Shuffle on": "Lecture aléatoire activée",
                "Shuffle off": "Lecture aléatoire désactivée",
                "Currently playing": "En cours de lecture",
                "Currently paused": "En pause",
                "Nothing currently playing": "Aucune lecture en cours",
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
            terms.extend(command["query"].split()[:3])  # First 3 words

        return terms

    def __del__(self):
        """Cleans up librespot when plugin is destroyed"""
        if self.librespot_proc:
            try:
                self.librespot_proc.terminate()
                self.librespot_proc.wait(timeout=2)
            except Exception:
                if self.librespot_proc.poll() is None:
                    self.librespot_proc.kill()
