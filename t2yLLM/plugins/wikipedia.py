from pluginManager import APIBase


class WikiAPI(APIBase):
    def name(self):
        return "wikipedia"

    def is_enabled(self):
        return self.config.general.wikipedia_api

    def is_query(self, user_input):
        """placeholder"""

    async def search(self, user_input, **kwargs):
        """placeholder -> replace with metacontext/llm_logic logic"""

    def format(self, input) -> str:
        """placeholder -> replace with metacontext logic"""
