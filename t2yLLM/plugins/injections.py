class PluginInjector:
    """
    any plugin can inherit if needed so it can provide examples
    for the LLM, formatting like JSON, specialized functions etc..
    """

    schema: dict
    examples: list[tuple[str, dict]]

    async def convert(
        self, user_input: str, llm, tokenizer, request_id: str | None = None
    ) -> dict:
        raise NotImplementedError
