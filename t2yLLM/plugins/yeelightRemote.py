from .pluginManager import APIBase, not_implemented
from typing import List, Dict, Any
from yeelight import discover_bulbs
from yeelight import Bulb
from yeelight import LightType


@not_implemented
class XiaomiLightAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False
        self.bulb_list = []

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

    def discover(self) -> Dict[str, Any]:
        discover_bulbs()
        """and now add method to
        add them to the yaml file if
        empty else, interactively ask user if he wants
        to add the new ones that were discovered"""

    def define(self, ip: str, effect="smooth", duration=1000) -> Bulb:
        bulb = Bulb(ip, effect, duration)
        return bulb

    def brightness(self, brightness_value: int, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.set_brightness(brightness_value, light_type=LightType.Ambient)

    def colorTemp(self, temperature: int, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.set_color_temp(temperature)

    def setDefault(self, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.set_default()

    def setHSV(
        self, hue: int, saturation: int, value: int, bulb_list: List[Bulb]
    ) -> None:
        for bulb in bulb_list:
            bulb.set_hsv(hue, saturation, value)

    def toggle(self, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.toggle()

    def poweron(self, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.turn_on()

    def poweroff(self, bulb_list: List[Bulb]) -> None:
        for bulb in bulb_list:
            bulb.turn_off()

    def properties(self, bulb_list: List[Bulb]) -> List[str, Any]:
        for bulb in bulb_list:
            bulb.get_properties()

    def is_query(self, user_input, threshold=0.91) -> bool:
        pass

    def search(self, user_input=None, **kwargs):
        pass

    def format(self) -> str:
        pass

    def search_terms(self, user_input):
        pass
