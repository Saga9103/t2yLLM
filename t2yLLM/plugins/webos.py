from .pluginManager import APIBase, not_implemented
from pywebostv.discovery import discover
from pywebostv.connection import WebOSClient
from pywebostv.controls import (
    MediaControl,
    TvControl,
    SystemControl,
    ApplicationControl,
    InputControl,
)


@not_implemented
class WebOsAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False
        self.silent_execution = False

    @property
    def name(self) -> str:
        return "WebOsAPI"

    @property
    def filename(self) -> str:
        return "webos"

    @property
    def is_enabled(self) -> bool:
        if self.name() or self.filename() in self.config.plugins.enabled_plugins:
            return True
        else:
            return False

    @property
    def memory(self) -> bool:
        return self.activate_memory

    def is_query(self, user_input, threshold=0.91) -> bool:
        pass

    def search(self, user_input=None, **kwargs):
        pass

    def format(self) -> str:
        pass

    def search_terms(self, user_input):
        pass
