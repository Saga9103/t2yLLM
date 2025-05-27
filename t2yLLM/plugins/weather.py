from pluginManager import APIBase


class WeatherAPI(APIBase):
    def name(self):
        return "weather"

    def is_enabled(self):
        return self.config.general.weather_api

    def is_query(self, user_input):
        """placeholder"""

    async def search(self, user_input, **kwargs):
        """placeholder -> replace with metacontext/llm_logic logic"""

    def format(self, input) -> str:
        """placeholder -> replace with metacontext logic"""
