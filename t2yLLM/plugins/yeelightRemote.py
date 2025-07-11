import re
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Any
import yaml
from yeelight import discover_bulbs, Bulb
from .pluginManager import APIBase, logger


class XiaomiLightAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = self.config.general.lang
        self.query = False
        self.activate_memory = False
        self.silent_execution = True
        self.bulb_groups = {}
        self.active_bulbs = {}
        self.last_command = ""
        self.config_file = Path(__file__).parent / "yeelightConfig.yaml"
        self._last_result = {"success": False, "message": "", "data": {}}

        self.load_config()

        self.command_patterns = {
            "fr": {
                "turn_on": r"allum(?:e|es?|er)|ouvre|démarre",
                "turn_off": r"éteins?|éteindre|ferme|stop",
                "brightness": r"luminosité|brillance|intensité",
                "brightness_increase": r"plus\s+(?:lumineux|clair|de\s+lumière)|augment(?:e|er)\s+(?:la\s+)?lumière|monte(?:r)?\s+(?:la\s+)?lumière|éclair(?:e|er)\s+plus",
                "brightness_decrease": r"moins\s+(?:lumineux|clair|de\s+lumière)|diminu(?:e|er)\s+(?:la\s+)?lumière|baiss(?:e|er)\s+(?:la\s+)?lumière|tamis(?:e|er)",
                "too_bright": r"trop\s+(?:lumineux|clair|éblouissant|fort)|(?:ça|c'est)\s+(?:trop\s+)?ébloui(?:ssant|t)",
                "too_dark": r"(?:trop\s+)?sombre|pas\s+assez\s+(?:lumineux|clair|de\s+lumière)|(?:il\s+)?fait\s+(?:trop\s+)?(?:sombre|noir)|manque\s+de\s+lumière",
                "ambient": r"ambiance|(?:lumière\s+)?tamis(?:ée|er)|(?:lumière\s+)?douce|cosy|relaxant",
                "color": r"couleur|teinte",
                "temperature": r"température|chaleur",
                "toggle": r"basculer|switch|inverse",
                "discover": r"cherche|trouve|découvre|scan",
                "list": r"liste|montre|affiche",
            },
            "en": {
                "turn_on": r"turn on|switch on|light up|power on",
                "turn_off": r"turn off|switch off|power off",
                "brightness": r"brightness|bright|dim|intensity",
                "brightness_increase": r"bright(?:er|en)|more\s+light|increase\s+(?:the\s+)?light|raise\s+(?:the\s+)?light",
                "brightness_decrease": r"dimmer|less\s+light|decrease\s+(?:the\s+)?light|lower\s+(?:the\s+)?light",
                "too_bright": r"too\s+bright|blinding|glaring",
                "too_dark": r"too\s+dark|not\s+bright\s+enough|(?:it'?s\s+)?dark",
                "ambient": r"ambient|mood|cozy|relaxing|soft\s+light",
                "color": r"color|colour|hue",
                "temperature": r"temperature|warmth|cool|warm",
                "toggle": r"toggle|switch",
                "discover": r"discover|find|scan|search",
                "list": r"list|show|display",
            },
        }

        self.colors = {
            "red": (0, 100, 100),
            "green": (120, 100, 100),
            "blue": (240, 100, 100),
            "yellow": (60, 100, 100),
            "purple": (270, 100, 100),
            "orange": (30, 100, 100),
            "pink": (300, 100, 100),
            "white": (0, 0, 100),
            "rouge": (0, 100, 100),
            "vert": (120, 100, 100),
            "bleu": (240, 100, 100),
            "jaune": (60, 100, 100),
            "violet": (270, 100, 100),
            "rose": (300, 100, 100),
            "blanc": (0, 0, 100),
            "chaud": (30, 40, 100),
            "froid": (200, 20, 100),
            "warm": (30, 40, 100),
            "cool": (200, 20, 100),
        }

    @property
    def name(self) -> str:
        return "XiaomiLightAPI"

    @property
    def filename(self) -> str:
        return "yeelightRemote"

    @property
    def is_enabled(self) -> bool:
        if self.name or self.filename in self.config.plugins.enabled_plugins:
            return True
        return False

    @property
    def memory(self) -> bool:
        return self.activate_memory

    def load_config(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                    self.bulb_groups = data.get("rooms", {})
                    saved_bulbs = data.get("bulbs", {})

                    for bulb_id, bulb_info in saved_bulbs.items():
                        try:
                            bulb = Bulb(bulb_info["ip"])
                            self.active_bulbs[bulb_id] = {
                                "bulb": bulb,
                                "info": bulb_info,
                            }
                        except Exception as e:
                            logger.warning(
                                f"Could not connect to Yeelight device {bulb_id}: {e}"
                            )
            else:
                logger.warning(
                    "No configuration file found. Please run the management script first."
                )
        except Exception as e:
            logger.error(f"Error loading config : {e}")

    def discover(self) -> List[Dict[str, Any]]:
        try:
            discovered_bulbs = discover_bulbs()
            return discovered_bulbs
        except Exception as e:
            logger.error(f"Error discovering Devices : {e}")
            return []

    def get_bulbs_by_room(self, room: str) -> List[Bulb]:
        bulbs = []
        if room in self.bulb_groups:
            for bulb_id in self.bulb_groups[room]:
                if bulb_id in self.active_bulbs:
                    bulbs.append(self.active_bulbs[bulb_id]["bulb"])
        return bulbs

    def get_all_bulbs(self) -> List[Bulb]:
        return [info["bulb"] for info in self.active_bulbs.values()]

    def apply_action(self, bulb: Bulb, action: str, value: Any):
        try:
            if action == "turn_on":
                bulb.turn_on(
                    effect="smooth", duration=200
                )  # or sudden with no duration
            elif action == "turn_off":
                bulb.turn_off(effect="smooth", duration=200)
            elif action == "toggle":
                bulb.toggle(effect="smooth", duration=200)
            elif action == "brightness" and value is not None:
                bulb.set_brightness(int(value))
            elif action == "color" and value is not None and isinstance(value, tuple):
                bulb.set_hsv(*value)
            elif action == "temperature" and value is not None:
                bulb.set_color_temp(int(value))
            return True
        except Exception as e:
            logger.error(f"Error controlling bulb {bulb}: {e}")
            if "closed the connection" in str(e):  # standard answer from yeelight
                return True

            return False

    def control_lights(self, bulbs: List[Bulb], action: str, value: Any = None) -> bool:
        if not bulbs:
            return False

        if len(bulbs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(bulbs)) as pool:
                results = list(
                    pool.map(lambda bulb: self.apply_action(bulb, action, value), bulbs)
                )
            return all(results)
        return self.apply_action(bulbs[0], action, value)

    def get_context(self, user_input: str) -> int:
        user_input_lower = user_input.lower()
        lang = "fr" if self.language == "fr" else "en"
        patterns = self.command_patterns[lang]

        if re.search(patterns.get("too_bright", ""), user_input_lower):
            return 30
        if re.search(patterns.get("too_dark", ""), user_input_lower):
            return 80
        if re.search(patterns.get("brightness_increase", ""), user_input_lower):
            return 100
        if re.search(patterns.get("brightness_decrease", ""), user_input_lower):
            return 20
        if re.search(patterns.get("ambient", ""), user_input_lower):
            return 40

        if self.language == "fr":
            if any(
                word in user_input_lower
                for word in ["lire", "lecture", "travail", "travailler"]
            ):
                return 90
            if any(
                word in user_input_lower for word in ["film", "télé", "tv", "cinéma"]
            ):
                return 25
            if any(word in user_input_lower for word in ["dormir", "coucher", "nuit"]):
                return 10
            if any(
                word in user_input_lower
                for word in ["manger", "repas", "dîner", "déjeuner"]
            ):
                return 70
        else:
            if any(
                word in user_input_lower
                for word in ["read", "reading", "work", "working"]
            ):
                return 90
            if any(
                word in user_input_lower
                for word in ["movie", "tv", "television", "cinema"]
            ):
                return 25
            if any(word in user_input_lower for word in ["sleep", "bed", "night"]):
                return 10
            if any(
                word in user_input_lower for word in ["eat", "meal", "dinner", "lunch"]
            ):
                return 70

        return None

    def parse_command(self, user_input: str) -> Dict[str, Any]:
        user_input_lower = user_input.lower()
        lang = "fr" if self.language == "fr" else "en"
        patterns = self.command_patterns[lang]
        command = {
            "action": None,
            "rooms": [],
            "value": None,
            "color": None,
            "bulb_id": None,
            "inferred_brightness": None,
        }

        for action, pattern in patterns.items():
            if re.search(pattern, user_input_lower):
                if action in [
                    "too_bright",
                    "too_dark",
                    "brightness_increase",
                    "brightness_decrease",
                    "ambient",
                ]:
                    command["action"] = "brightness"
                    command["inferred_brightness"] = self.get_context(user_input)
                else:
                    command["action"] = action
                break

        for room in self.bulb_groups.keys():
            if room.lower() in user_input_lower:
                command["rooms"].append(room)
                # break

        brightness_match = re.search(r"(\d+)\s*%?", user_input)
        if brightness_match:
            command["value"] = min(100, max(1, int(brightness_match.group(1))))
        elif command["inferred_brightness"]:
            command["value"] = command["inferred_brightness"]

        for color, hsv in self.colors.items():
            if color in user_input_lower:
                command["color"] = hsv
                break

        temp_match = re.search(r"(\d{4})\s*k", user_input_lower)
        if temp_match:
            command["value"] = int(temp_match.group(1))

        return command

    def is_query(self, user_input: str) -> bool:
        user_input = user_input.lower()
        lang = "fr" if self.language == "fr" else "en"

        light_keywords = {
            "fr": [
                "lumière",
                "lumières",
                "lampe",
                "lampes",
                "éclairage",
                "ampoule",
                "luminaire",
                "luminaires",
                "plafonnier",
                "led",
                "leds",
                "sombre",
                "lumineux",
                "clair",
                "éclaire",
                "éclairer",
                "ambiance",
                "luminosité",
                "tamisé",
                "éblouissant",
            ],
            "en": [
                "light",
                "lamp",
                "bulb",
                "lighting",
                "illumination",
                "dark",
                "bright",
                "dim",
                "illuminate",
                "glow",
                "ambient",
                "brightness",
                "glaring",
            ],
        }

        for keyword in light_keywords.get(lang, light_keywords["en"]):
            if keyword in user_input:
                self.query = True
                return True

        for pattern in self.command_patterns[lang].values():
            if re.search(pattern, user_input):
                self.query = True
                return True

        for room in self.bulb_groups.keys():
            if room.lower() in user_input:
                context_words = {
                    "fr": ["sombre", "lumineux", "allume", "éteins", "clair"],
                    "en": ["dark", "bright", "turn", "light", "dim"],
                }
                if any(word in user_input for word in context_words.get(lang, [])):
                    self.query = True
                    return True

        self.query = False
        return False

    def search(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """
        handles multiple commands for multiple rooms
        """
        try:
            self.last_command = user_input
            commands = self.parse_multiple_commands(user_input)
            if not commands:  # fallback
                commands = [self.parse_command(user_input)]

            combined = {
                "success": True,
                "message": [],
                "data": {"results": []},
            }
            for cmd in commands:
                sub_result = self.single_command(cmd)
                combined["data"]["results"].append(sub_result)
                combined["success"] &= sub_result.get("success", False)
                combined["message"].append(sub_result.get("message", ""))

            combined["message"] = " ⟶ ".join(combined["message"])
            self._last_result = combined
            return combined

        except Exception as e:
            logger.error(f"Error in search: {e}")
            self._last_result = {"success": False, "message": str(e), "data": {}}
            return self._last_result

    def single_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        executes one command from user
        """
        result = {"success": False, "message": "", "data": {}}

        if command["action"] == "discover":
            bulbs = self.discover()
            result.update(
                success=True,
                message=f"Found {len(bulbs)} Device(s) on the network",
                data={"discovered": len(bulbs)},
            )
            return result

        if command["action"] == "list":
            result.update(
                success=True,
                message=(
                    f"Found {len(self.active_bulbs)} configured devices in "
                    f"{len(self.bulb_groups)} rooms"
                ),
                data={
                    "rooms": self.bulb_groups,
                    "bulbs": {
                        bid: info["info"] for bid, info in self.active_bulbs.items()
                    },
                },
            )
            return result

        if command["rooms"]:
            bulbs, room_names = [], []
            for room in command["rooms"]:
                bulbs.extend(self.get_bulbs_by_room(room))
                room_names.append(f"'{room}'")
            bulbs = list({id(b): b for b in bulbs}.values())  # unicité

            target = (
                "rooms " + ", ".join(room_names[:-1]) + " and " + room_names[-1]
                if len(room_names) > 1
                else "room " + room_names[0]
            )
        else:
            bulbs = self.get_all_bulbs()
            target = "all lights"

        act = command["action"]
        val = command["value"] or command.get("color")

        if act == "turn_on":
            ok = self.control_lights(bulbs, "turn_on")
            msg = f"Turned on {target}"
        elif act == "turn_off":
            ok = self.control_lights(bulbs, "turn_off")
            msg = f"Turned off {target}"
        elif act == "toggle":
            ok = self.control_lights(bulbs, "toggle")
            msg = f"Toggled {target}"
        elif act == "brightness" and val is not None:
            ok = self.control_lights(bulbs, "brightness", val)
            msg = f"Set brightness to {val}% for {target}"
        elif act in ("color",) or command.get("color"):
            ok = self.control_lights(bulbs, "color", val)
            msg = f"Changed color for {target}"
        elif act == "temperature" and val is not None:
            ok = self.control_lights(bulbs, "temperature", val)
            msg = f"Set temperature to {val}K for {target}"
        else:
            ok = False
            msg = "Could not understand the command"

        result.update(success=ok, message=msg)
        return result

    def format(self) -> str:
        result = self._last_result
        lang = "fr" if self.language == "fr" else "en"

        if lang == "fr":
            if result["success"]:
                action_messages = {
                    "Turned on": "J'ai allumé",
                    "Turned off": "J'ai éteint",
                    "Set brightness": "J'ai réglé la luminosité",
                    "Changed color": "J'ai changé la couleur",
                }

                for eng, fr in action_messages.items():
                    if eng in result["message"]:
                        return result["message"].replace(eng, fr)

                return f"Commande lumière exécutée: {result['message']}"

            return f"Je n'ai pas pu contrôler les lumières: {result['message']}"

    def parse_multiple_commands(self, user_input: str) -> List[Dict[str, Any]]:
        commands = []
        parts = re.split(
            r"\s*(?:,|(?:\b(?:et|puis|and|then)\b))\s+", user_input, flags=re.I
        )

        last_action = last_value = last_color = None
        for part in parts:
            cmd = self.parse_command(part)
            if not cmd["action"] and last_action:
                cmd["action"] = last_action
            if cmd["value"] is None and last_value is not None:
                cmd["value"] = last_value
            if cmd["color"] is None and last_color is not None:
                cmd["color"] = last_color
            if cmd["action"]:
                last_action = cmd["action"]
                last_value = cmd["value"]
                last_color = cmd["color"]
                commands.append(cmd)

        return commands

    def search_terms(self, user_input: str) -> List[str]:
        terms = []
        command = self.parse_command(user_input)

        if command["action"]:
            terms.append(command["action"])
        for room in command["rooms"]:
            terms.append(room)
        if command["color"]:
            terms.append("color")
        if command["inferred_brightness"]:
            terms.append("action")

        return terms
