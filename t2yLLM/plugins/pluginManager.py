from abc import ABC, abstractmethod
import os
import spacy
import logging
import sys
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t2yLLM.config.yamlConfigLoader import Loader
from t2yLLM.LangLoader import LangLoader

logger = logging.getLogger("Plugins")


class APIBase(ABC):
    """base class for all APIs to register"""

    def __init__(self, config: Loader = None, language: LangLoader = None, nlp=None):
        self.config = config
        self.language = language
        self.nlp = nlp
        self.enabled = self.is_enabled
        self.query = False
        self.activates_memory = False

    @classmethod
    @abstractmethod
    def init(cls, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self):
        "returns the name of the API related class"
        pass

    @property
    @abstractmethod
    def filename(self):
        "file name of the module"
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        pass

    @abstractmethod
    def is_query(self) -> bool:
        pass

    @abstractmethod
    def search(self):
        """plugin search"""
        pass

    @abstractmethod
    def format(self) -> str:
        """correctly formats answer for LLM"""
        pass

    @abstractmethod
    def search_terms(self):
        pass

    @property
    def memory(self) -> bool:
        """if plugin should activate memorization process or not"""
        return self.activate_memory


class PluginManager:
    def __init__(self, plugin_dict=None):
        self.config = Loader().loadChatConfig()
        try:
            self.language = LangLoader()
        except Exception as e:
            logger.error(f"Error Loading language : {e}")
            self.language = None
        try:
            if self.config.general.lang == "fr":
                self.nlp = spacy.load(self.config.llms.spacy_model)
                # spacy is kinda very fast but sadly limited in french compared to
                # its english counterpart
            else:
                self.nlp = spacy.load(self.config.llms.spacy_model_en)
        except Exception:
            if self.config.general.lang == "fr":
                print("downloading model Spacy model")
                os.system(f"python -m spacy download {self.config.llms.spacy_model}")
                self.nlp = spacy.load(self.config.llms.spacy_model)
            else:
                print("downloading model Spacy model")
                os.system(f"python -m spacy download {self.config.llms.spacy_model_en}")
                self.nlp = spacy.load(self.config.llms.spacy_model_en)
        self.existing = plugin_dict
        self.plugins = []
        self.memory_plugins = []
        self.register()

    def register(self):
        for key, value in self.existing.items():
            try:
                plugin_path = f"t2yLLM.plugins.{key}"
                if key in self.config.plugins.enabled_plugins:
                    plugin_module = importlib.import_module(plugin_path)
                    plugin_cls = getattr(plugin_module, value)
                    plugin = plugin_cls.init(
                        config=self.config, language=self.language, nlp=self.nlp
                    )
                    self.plugins.append(plugin)
                    logger.info(f"\033[94mPlugin Enabled : {plugin.name}\033[0m")
            except Exception as e:
                logger.warning(
                    f"\033[94mfailed to initialize plugin '{value}' from module '{key}': {e}\033[0m"
                )

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
            try:
                if plugin.is_query(user_input):
                    handlers.append(plugin)
                    logger.info(f"Query type detect as True for : {plugin.name}")
            except Exception:
                pass
        return handlers

    def __call__(self, user_input, **kwargs):
        self.memory_plugins = []
        results = {}
        identified = self.identify(user_input)

        if not identified:
            return results

        for plugin in identified:
            try:
                if plugin.memory:
                    self.memory_plugins.append(plugin.name)
                search_result = plugin.search(user_input, **kwargs)

                if search_result and search_result.get("success", False):
                    results[plugin.name] = {
                        "data": search_result,
                        "formatted": plugin.format(),
                        "search_terms": plugin.search_terms(user_input),
                    }

            except Exception as e:
                logger.error(f"Error in plugin {plugin.name} : {e}")

        return plugin.format()  # results
