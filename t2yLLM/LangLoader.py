from pathlib import Path
import json
import os
import sys
import logging

from t2yLLM.config.yamlConfigLoader import Loader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("LangLoader")


class LangLoader:
    """Main class that loads responsible for language
    and languages keywords"""

    def __init__(self):
        self.config = (Loader()).loadWhispConfig()
        self.config_dir = Path(__file__).parent / "config"
        # should add checks here
        self.lang = self.config.general.lang
        self.lang_file = f"lang_{self.lang}.json"
        self.lang_path = None
        self.lang_data = {}
        self.load_language()

    def load_language(self):
        """
        loads language keywords from config files
        """
        self.lang_path = self.config_dir / self.lang_file
        if not self.lang_path.exists():
            self.lang_path = self.config_dir / "lang_en.json"
            logger.warning(
                "Language file not found or incorrectly configured, defaulting to english"
            )
        try:
            with open(self.lang_path, "r", encoding="utf-8") as f:
                self.lang_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError Could not load language file {e}")
        except Exception as e:
            logger.error(
                f"Could not load language file, check permissions and format : {e}"
            )

    def get_val(self, key, value):
        return self.lang_data.get(key, value)

    @property
    def weather_keywords(self):
        return self.lang_data.get("weather_keywords", [])

    @property
    def time_keywords(self):
        return self.lang_data.get("time_keywords", [])

    @property
    def date_keywords(self):
        return self.lang_data.get("date_keywords", [])

    @property
    def location_keywords(self):
        return self.lang_data.get("location_keywords", [])

    @property
    def temporal_indicators(self):
        return self.lang_data.get("temporal_indicators", [])

    @property
    def condition_patterns(self):
        return self.lang_data.get("condition_patterns", [])

    @property
    def forecast_patterns(self):
        return self.lang_data.get("forecast_patterns", [])

    @property
    def week_days(self):
        return self.lang_data.get("weekdays", {})

    @property
    def month_names(self):
        return self.lang_data.get("months", [])

    @property
    def day_names(self):
        return self.lang_data.get("day_names", [])

    @property
    def location_prepositions(self):
        return self.lang_data.get("location_prepositions", [])

    @property
    def temp_patterns(self):
        return self.lang_data.get("temp_patterns", {})

    @property
    def condition_matches(self):
        return self.lang_data.get("condition_matches", {})
