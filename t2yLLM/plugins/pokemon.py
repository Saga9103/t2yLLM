from pluginManager import APIBase


class PokeAPI(APIBase):
    def name(self):
        return "pokemon"

    def is_enabled(self):
        return self.config.general.pokemon_api

    def is_query(self, user_input):
        """uses detect_pokemon_in_text"""

    async def search(self, user_input, **kwargs):
        """placeholder -> replace with metacontext/llm_logic logic"""

    def format(self, input) -> str:
        """placeholder -> replace with metacontext logic"""
