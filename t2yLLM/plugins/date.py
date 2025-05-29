from .pluginManager import APIBase, logger
from datetime import datetime


class DateAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False
        self.date_info = None

    @property
    def name(self) -> str:
        return "DateAPI"

    @property
    def filename(self) -> str:
        return "date"

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
        user_input_lower = user_input.lower()
        date_keywords = self.language.date_keywords
        self.query = any(keyword in user_input_lower for keyword in date_keywords)

        return self.query

    def search(self, user_input=None, **kwargs):
        if not self.query:
            return {"success": False, "error": "Not a date query"}

        now = datetime.now()
        day_names = self.language.day_names
        month_names = self.language.month_names
        day_of_week = day_names[now.weekday()]
        month_name = month_names[now.month - 1]

        self.date_info = {
            "day_of_week": day_of_week,
            "day": now.day,
            "month": month_name,
            "month_number": now.month,
            "year": now.year,
            "date_object": now,
        }

        return {"success": True, "data": self.date_info}

    def format(self) -> str:
        if not self.date_info:
            return "No date information available"

        day_of_week = self.date_info["day_of_week"]
        day = self.date_info["day"]
        month = self.date_info["month"]
        year = self.date_info["year"]

        if self.config.general.lang == "fr":
            return f"Nous sommes le {day_of_week} {day} {month} {year}"
        else:
            return f"Today is {day_of_week}, {month} {day}, {year}"

    def search_terms(self, user_input):
        return None
