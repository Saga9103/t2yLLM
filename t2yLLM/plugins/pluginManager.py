from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("Plugins")


class APIBase(ABC):
    """base class for all APIs to register"""

    def __init__(self, config):
        self.config = config
        self.enabled = self.is_enabled()

    @abstractmethod
    @property
    def name(self):
        pass

    @abstractmethod
    @property
    def is_enabled(self) -> bool:
        pass

    @abstractmethod
    def is_query(self) -> bool:
        pass

    @abstractmethod
    async def search(self):
        """plugin search"""
        pass

    @abstractmethod
    def format(self) -> str:
        """correctly formats answer for LLM"""
        pass

    @abstractmethod
    def search_terms(self):
        pass


class PluginManager:
    def __init__(self, plugin_list):
        self.existing = plugin_list
        self.plugins = []
        self.register()

    def register(self, plugin: APIBase):
        for plugin in self.existing:
            if plugin.enabled:
                self.plugins.append(plugin)
                logger.info(f"Plugin Enabled : {plugin.name}")

    def enumerate(self) -> list:
        listing = []
        for elt in self.plugins:
            listing.append(elt.name)
        return listing

    def identify(self, user_input) -> list:
        """tells which plugin(s)
        will handle the query. returns
        a list of APIBase objects"""
        handlers = []
        for plugin in self.plugins:
            if plugin.is_query(user_input):
                handlers.append(plugin)
        return handlers

    async def __call__(self, user_input, **kwargs):
        """unified search from relevant plugins"""
        results = {}
        identified = self.identify(user_input)
        for plugin in identified:
            try:
                search = await plugin.search(user_input, **kwargs)
                if search.get("success", False):
                    results[plugin.name] = search
            except Exception:
                pass

        return results
