from difflib import SequenceMatcher
from keybert import KeyBERT
import wikipedia
from .pluginManager import APIBase, logger


class WikiAPI(APIBase):
    @classmethod
    def init(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, **kwargs):
        self.config = kwargs.get("config")
        self.nlp = kwargs.get("nlp")
        self.language = kwargs.get("language")
        self.query = False
        self.wiki_search_terms = ""
        self.wiki_result = None
        self.original_query = ""
        self.activate_memory = True
        self.kw_model = kwargs.get("embedding_model")
        self.kw_extractor = KeyBERT(self.kw_model)

    @property
    def name(self) -> str:
        return "WikipediaAPI"

    @property
    def filename(self) -> str:
        return "wikipedia"

    @property
    def is_enabled(self) -> bool:
        if self.name or self.filename in self.config.plugins.enabled_plugins:
            return True
        return False

    @property
    def memory(self) -> bool:
        return self.activate_memory

    def is_query(self, user_input) -> bool:
        if not self.config.general.web_enabled:
            return False
        user_input_lower = user_input.lower()
        self.original_query = user_input

        wikipedia_keywords = ["wikipedia", "wiki", "encyclopédie", "encyclopedia"]
        has_wiki_keyword = any(
            keyword in user_input_lower for keyword in wikipedia_keywords
        )
        knowledge_patterns = [
            "qu'est-ce que",
            "qu'est ce que",
            "c'est quoi",
            "définition",
            "définir",
            "expliquer",
            "explique",
            "comment",
            "pourquoi",
            "qui est",
            "qui était",
            "what is",
            "who is",
            "who was",
            "define",
            "explain",
            "how",
            "why",
            "tell me about",
            "dis-moi",
            "parle-moi",
            "histoire de",
            "history of",
        ]

        is_knowledge_query = any(
            pattern in user_input_lower for pattern in knowledge_patterns
        )

        self.query = has_wiki_keyword or (
            is_knowledge_query and len(user_input.split()) > 2
        )

        if self.query:
            self.wiki_search_terms = self.extract_search_terms(user_input)

        return self.query

    def extract_search_terms(self, user_input):
        kw_search = ""
        try:
            keywords = self.kw_extractor.extract_keywords(user_input, top_n=3)
            for elt in keywords:
                kw_search += elt[0] + " "
        except Exception as e:
            logger.warning(f"Error in keyword search : {e}")
        logger.info(f"keywords returned for Wikipedia : {kw_search}")
        return kw_search

    def search(self, user_input=None, **kwargs):
        if not self.query or not self.wiki_search_terms:
            return {"success": False, "error": "Not a Wikipedia query"}
        wikipedia.set_lang(self.config.general.lang)

        try:
            search_results = wikipedia.search(
                self.wiki_search_terms,
                results=self.config.wikipedia.wiki_sentence_search_nb
                if hasattr(self.config.wikipedia, "wiki_sentence_search_nb")
                else 5,
            )

            if not search_results:
                return {
                    "success": False,
                    "error": "No results found on Wikipedia",
                    "query": self.wiki_search_terms,
                }

            candidates = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(
                        title,
                        sentences=self.config.wikipedia.summary_size
                        if hasattr(self.config.wikipedia, "summary_size")
                        else 3,
                        auto_suggest=False,
                    )

                    relevance_score = self.calculate_relevance_score(
                        title, summary, self.original_query, self.wiki_search_terms
                    )

                    candidates.append(
                        {
                            "title": page.title,
                            "summary": summary,
                            "url": page.url,
                            "content": page.content[:1000],
                            "relevance": relevance_score,
                        }
                    )

                except (
                    wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError,
                ) as e:
                    logger.warning(f"Wikipedia page error for '{title}': {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing Wikipedia page '{title}': {str(e)}")
                    continue

            if not candidates:
                return {
                    "success": False,
                    "error": "No valid content found",
                    "query": self.wiki_search_terms,
                }

            candidates.sort(key=lambda x: x["relevance"], reverse=True)
            best_candidate = candidates[0]
            if best_candidate["relevance"] < 0.3:
                return {
                    "success": False,
                    "error": "Results not relevant enough",
                    "best_match": best_candidate["title"],
                    "relevance": best_candidate["relevance"],
                    "query": self.wiki_search_terms,
                }

            self.wiki_result = best_candidate
            return {"success": True, "data": best_candidate}

        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return {
                "success": False,
                "error": f"Search error: {str(e)}",
                "query": self.wiki_search_terms,
            }

    def calculate_relevance_score(self, title, summary, original_query, search_terms):
        score = 0.0

        title_lower = title.lower()
        query_lower = original_query.lower()
        summary_lower = summary.lower()
        search_words = search_terms.lower().split()
        term_in_title_count = sum(1 for term in search_words if term in title_lower)
        title_term_score = term_in_title_count / max(1, len(search_words))
        score += title_term_score * 0.3
        best_sequence_match = 0
        for term in search_words:
            sequence_similarity = SequenceMatcher(None, term, title_lower).ratio()
            best_sequence_match = max(best_sequence_match, sequence_similarity)
        score += best_sequence_match * 0.4
        term_occurrences = sum(summary_lower.count(term) for term in search_words)
        summary_term_score = min(1.0, term_occurrences / 5)  # Cap at 5 occurrences
        score += summary_term_score * 0.2
        query_words = set(query_lower.split())
        common_words = sum(
            1 for word in query_words if word in summary_lower and len(word) > 3
        )
        context_score = min(1.0, common_words / max(5, len(query_words)))
        score += context_score * 0.1

        if len(summary) < 100:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def format(self) -> str:
        if not self.wiki_result:
            return "No Wikipedia information available"

        result = self.wiki_result

        if self.config.general.lang == "fr":
            formatted = f"Information de Wikipedia sur '{result['title']}':\n\n"
            formatted += f"{result['summary']}\n\n"
            formatted += f"Source: {result['url']}"
        else:
            formatted = f"Wikipedia information about '{result['title']}':\n\n"
            formatted += f"{result['summary']}\n\n"
            formatted += f"Source: {result['url']}"

        return formatted

    def search_terms(self, user_input):
        return self.wiki_search_terms
