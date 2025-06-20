from .pluginManager import APIBase, logger
from typing import List, Dict, Any
from yeelight import discover_bulbs, Bulb
import re
from pathlib import Path
import yaml
import concurrent.futures


class XiaomiLightAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language") or (
            self.config.general.lang if self.config else "en"
        )
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
        }

    @property
    def name(self) -> str:
        return "XiaomiLightAPI"

    @property
    def filename(self) -> str:
        return "yeelightRemote"

    @property
    def is_enabled(self) -> bool:
        if self.name() or self.filename() in self.config.plugins.enabled_plugins:
            return True
        else:
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
        else:
            return self.apply_action(bulbs[0], action, value)

    def parse_command(self, user_input: str) -> Dict[str, Any]:
        user_input_lower = user_input.lower()
        lang = "fr" if self.language == "fr" else "en"
        patterns = self.command_patterns[lang]
        command = {
            "action": None,
            "room": None,
            "value": None,
            "color": None,
            "bulb_id": None,
        }

        for action, pattern in patterns.items():
            if re.search(pattern, user_input_lower):
                command["action"] = action
                break

        for room in self.bulb_groups.keys():
            if room.lower() in user_input_lower:
                command["room"] = room
                break

        brightness_match = re.search(r"(\d+)\s*%?", user_input)
        if brightness_match:
            command["value"] = min(100, max(1, int(brightness_match.group(1))))

        for color, hsv in self.colors.items():
            if color in user_input_lower:
                command["color"] = hsv
                break

        temp_match = re.search(r"(\d{4})\s*k", user_input_lower)
        if temp_match:
            command["value"] = int(temp_match.group(1))

        return command

    def is_query(self, user_input: str, threshold: float = 0.7) -> bool:
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
            ],
            "en": ["light", "lamp", "bulb", "lighting", "illumination"],
        }

        for keyword in light_keywords.get(lang, light_keywords["en"]):
            if keyword in user_input:
                self.query = True
                return True

        for pattern in self.command_patterns[lang].values():
            if re.search(pattern, user_input):
                for room in self.bulb_groups.keys():
                    if room.lower() in user_input:
                        self.query = True
                        return True
                for color in self.colors.keys():
                    if color in user_input:
                        self.query = True
                        return True

        self.query = False
        return False

    def search(self, user_input: str, **kwargs) -> Dict[str, Any]:
        try:
            self.last_command = user_input
            command = self.parse_command(user_input)
            result = {"success": False, "message": "", "data": {}}

            if command["action"] == "discover":
                bulbs = self.discover()
                result["success"] = True
                result["data"]["discovered"] = len(bulbs)
                result["message"] = f"Found {len(bulbs)} Device(s) on the network"
                self._last_result = result
                return result

            if command["action"] == "list":
                result["success"] = True
                result["data"]["rooms"] = self.bulb_groups
                result["data"]["bulbs"] = {
                    bulb_id: info["info"] for bulb_id, info in self.active_bulbs.items()
                }
                result["message"] = (
                    f"Found {len(self.active_bulbs)} configured devices in {len(self.bulb_groups)} rooms"
                )
                self._last_result = result
                return result

            if command["room"]:
                bulbs = self.get_bulbs_by_room(command["room"])
                target = f"room '{command['room']}'"
            else:
                bulbs = self.get_all_bulbs()
                target = "all lights"

            if not bulbs:
                result["message"] = f"No yeelight devices found for {target}"
                self._last_result = result
                return result

            if command["action"] == "turn_on":
                success = self.control_lights(bulbs, "turn_on")
                result["success"] = success
                result["message"] = f"Turned on {target}"

            elif command["action"] == "turn_off":
                success = self.control_lights(bulbs, "turn_off")
                result["success"] = success
                result["message"] = f"Turned off {target}"

            elif command["action"] == "toggle":
                success = self.control_lights(bulbs, "toggle")
                result["success"] = success
                result["message"] = f"Toggled {target}"

            elif command["action"] == "brightness" and command["value"]:
                success = self.control_lights(bulbs, "brightness", command["value"])
                result["success"] = success
                result["message"] = (
                    f"Set brightness to {command['value']}% for {target}"
                )

            elif command["action"] == "color" or command["color"]:
                color_value = command["color"] or command["value"]
                if color_value:
                    success = self.control_lights(bulbs, "color", color_value)
                    result["success"] = success
                    result["message"] = f"Changed color for {target}"

            elif command["action"] == "temperature" and command["value"]:
                success = self.control_lights(bulbs, "temperature", command["value"])
                result["success"] = success
                result["message"] = (
                    f"Set temperature to {command['value']}K for {target}"
                )

            else:
                result["message"] = "Could not understand the command"

            self._last_result = result
            return result

        except Exception as e:
            logger.error(f"Error in search: {e}")
            result = {"success": False, "message": str(e), "data": {}}
            self._last_result = result
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
            else:
                return f"Je n'ai pas pu contrôler les lumières: {result['message']}"

    def parse_multiple_commands(self, user_input: str) -> List[Dict[str, Any]]:
        commands = []
        parts = re.split(r"\s+(?:et|puis)\s+", user_input)

        for part in parts:
            command = self.parse_command(part)
            if command["action"]:
                commands.append(command)

        return commands

    def search_terms(self, user_input: str) -> List[str]:
        terms = []
        command = self.parse_command(user_input)

        if command["action"]:
            terms.append(command["action"])
        if command["room"]:
            terms.append(command["room"])
        if command["color"]:
            terms.append("color")

        return terms
