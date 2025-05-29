from .pluginManager import APIBase, logger


class ExampleAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.activate_memory = False

    @property
    def name(self) -> str:
        return "ExampleAPI"

    @property
    def filename(self) -> str:
        return "example"

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
