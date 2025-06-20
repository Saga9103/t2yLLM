from .pluginManager import APIBase, logger
from datetime import datetime


class TimeAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = None
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False
        self.timeinfo = None
        self.silent_execution = False

    @property
    def name(self) -> str:
        return "TimeAPI"

    @property
    def filename(self) -> str:
        return "time"

    @property
    def is_enabled(self) -> bool:
        if self.name or self.filename in self.config.plugins.enabled_plugins:
            return True
        else:
            return False

    @property
    def memory(self):
        return self.activate_memory

    def is_query(self, user_input) -> bool:
        time_keywords = self.language.time_keywords
        try:
            return any(keyword in user_input.lower() for keyword in time_keywords)
        except KeyError:
            return False

    def search(self, user_input=None, **kwargs):
        if self.is_query(user_input):
            self.timeinfo = datetime.now()
            return {"success": True, "data": self.timeinfo}
        else:
            return {"success": False, "data": None}

    def format(self) -> str:
        now = self.timeinfo
        return f"{now.hour} heure{'s' if now.hour > 1 else ''} {now.minute:02d}"

    def search_terms(self, user_input):
        return None
