from abc import ABC, abstractmethod
import os
import logging
import sys
import importlib
import re
import unicodedata
from functools import wraps
import spacy
import html
from urllib.parse import quote_plus

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t2yLLM.config.yamlConfigLoader import Loader
from t2yLLM.LangLoader import LangLoader
from t2yLLM.plugins.injections import PluginInjector

logger = logging.getLogger("Plugins")


# DECORATORS
def optional(decorator):
    """just ignores if not defined"""
    if decorator is None:
        return lambda f: f
    return decorator


def not_implemented(obj):
    if isinstance(obj, type):
        original_init = obj.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            raise NotImplementedError(f"{obj.__name__} is not implemented")

        obj.__init__ = new_init

        return obj

    @wraps(obj)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{obj.__name__}() is not implemented")

    return wrapper


class APIBase(ABC):
    """base class for all APIs to register"""

    def __init__(
        self,
        config: Loader = None,
        language: LangLoader = None,
        nlp=None,
        embedding_model=None,
    ):
        self.config = config
        self.language = language
        self.nlp = nlp
        self.enabled = self.is_enabled
        self.query = False
        self.activates_memory = False
        self.silent_execution = False

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

    @abstractmethod
    def format(self) -> str:
        """correctly formats answer for LLM"""

    @abstractmethod
    def search_terms(self):
        pass

    @property
    def memory(self) -> bool:
        """if plugin should activate memorization process or not"""
        return self.activates_memory

    @property
    def silent(self) -> bool:
        """if plugin should execute without voice feedback"""
        return self.silent_execution


class PluginManager:
    def __init__(self, plugin_dict=None, embedding_model=None):
        self.config = Loader().loadChatConfig()
        self.embedding_model = embedding_model
        self.safe_chars = re.compile(r"[^a-z0-9À-ÿ\s'\"/:\-_%]", re.I)
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
        self.injectors = []
        self.memory_plugins = []
        self.silent_plugins = []
        self.handlers = []
        self.is_silent = False
        self.override = True
        self.register()

    def get_injector(self, handler):
        return next((p for p in self.injectors if p.name == handler), None)

    def sanitize(self, user_input: str, url_safe=False) -> str:
        text = unicodedata.normalize("NFKC", user_input).strip()
        text = text[:512]
        text = self.safe_chars.sub("", text)
        text = html.escape(text)
        return quote_plus(text) if url_safe else text

    def register(self):
        for key, value in self.existing.items():
            try:
                plugin_path = f"t2yLLM.plugins.{key}"
                if key in self.config.plugins.enabled_plugins:
                    plugin_module = importlib.import_module(plugin_path)
                    plugin_cls = getattr(plugin_module, value)
                    plugin = plugin_cls.init(
                        config=self.config,
                        language=self.language,
                        nlp=self.nlp,
                        embedding_model=self.embedding_model,
                    )
                    self.plugins.append(plugin)
                    if isinstance(plugin, PluginInjector):
                        self.injectors.append(plugin)
                    logger.info(f"\033[94mPlugin Enabled : {plugin.name}\033[0m")
            except Exception as e:
                logger.warning(
                    f"\033[94mfailed to initialize plugin '{value}' from module '{key}': {e}\033[0m"
                )

    def unregister(self, plugin_name: str):
        if len(self.memory_plugins) > 0:
            if self.plugin_name in self.memory_plugins:
                self.memory_plugins.remove(plugin_name)
        else:
            pass

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
        self.silent_plugins = []
        self.handlers = []
        results = {}
        clean_input = self.sanitize(user_input)
        identified = self.identify(clean_input)

        self.handlers = identified

        self.override = True  # clean reset
        self.is_silent = False

        if not identified:
            return results

        if all(plugin.silent for plugin in identified):
            self.is_silent = True

        if all(plugin.memory for plugin in identified):
            self.override = False

        for plugin in identified:
            try:
                if plugin.silent:
                    self.silent_plugins.append(plugin.name)

                if not plugin.memory:
                    search_result = plugin.search(user_input, **kwargs)
                elif plugin.memory and not self.override:
                    self.memory_plugins.append(plugin.name)
                    search_result = plugin.search(user_input, **kwargs)
                else:
                    continue

                if search_result and search_result.get("success", False):
                    results[plugin.name] = {
                        "data": search_result,
                        "formatted": plugin.format(),
                        "search_terms": plugin.search_terms(user_input),
                    }

            except Exception as e:
                logger.error(f"Error in plugin {plugin.name} : {e}")

        # return plugin.format()
        if results:
            return results
        return {}
