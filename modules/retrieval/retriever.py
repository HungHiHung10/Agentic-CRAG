import os
import re

from modules.text_utils import normalize_for_search


class InMemoryVectorStore:
    """In-memory vector store used when a persistent backend is not selected."""

    def __init__(self, documents, embeddings, embedding_model):
        self.documents = documents
        self.embeddings = embeddings
        self.embedding_model = embedding_model

    @classmethod
    def from_documents(cls, documents, embedding_model):
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        return cls(documents, embeddings, embedding_model)

    def as_retriever(self, search_type=None, search_kwargs=None):
        return HybridRetriever(self.documents, self.embeddings, self.embedding_model, search_kwargs or {})


class HybridRetriever:
    """Dense-plus-lexical retriever with definition-aware scoring and source diversification."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "in", "is", "of", "on", "or", "the", "to", "what",
        "answer", "english", "only", "question", "please",
        "la", "gi", "nghia", "cho", "hoi", "ve", "hay", "nhung", "cac", "mot", "va",
        "cach", "dieu", "tri",
    }

    def __init__(self, documents, embeddings, embedding_model, search_kwargs):
        self.documents = documents
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.k = int(search_kwargs.get("k", 3))
        self.score_threshold = float(search_kwargs.get("score_threshold", -1.0))
        self.normalized_documents = [self._compose_searchable_document_text(doc) for doc in documents]

    @staticmethod
    def _dense_similarity(left, right):
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _normalize_search_text(text):
        return normalize_for_search(text)

    @classmethod
    def _extract_query_terms(cls, text):
        normalized = cls._normalize_search_text(text)
        return [tok for tok in re.findall(r"\w+", normalized, flags=re.UNICODE) if tok not in cls.STOPWORDS]

    @classmethod
    def _build_query_ngrams(cls, question):
        question = re.sub(r"answer\s+in\s+english\s+only\.?,?", " ", question or "", flags=re.I)
        tokens = cls._extract_query_terms(question)
        ngrams = []
        for i in range(len(tokens) - 1):
            ngrams.append(" ".join(tokens[i:i + 2]))
        for i in range(len(tokens) - 2):
            ngrams.append(" ".join(tokens[i:i + 3]))
        return ngrams, tokens

    @classmethod
    def _compose_searchable_document_text(cls, doc):
        metadata = getattr(doc, "metadata", {}) or {}
        metadata_text = " ".join(str(value) for value in metadata.values())
        return cls._normalize_search_text(f"{metadata_text}\n{getattr(doc, 'page_content', '')}")

    @classmethod
    def _extract_definition_focus(cls, question):
        normalized = cls._normalize_search_text(
            re.sub(r"answer\s+in\s+english\s+only\.?,?", " ", question or "", flags=re.I)
        )
        normalized = normalized.strip(" ?:;,.\n\t")
        match = re.search(r"(.+?)\s+(?:la\s+gi|nghia\s+la\s+gi)$", normalized)
        if not match:
            return ""
        tokens = [
            token
            for token in re.findall(r"\w+", match.group(1), flags=re.UNICODE)
            if token not in cls.STOPWORDS
        ]
        return " ".join(tokens)

    @classmethod
    def _score_definition_match(cls, question, normalized_doc):
        focus = cls._extract_definition_focus(question)
        if not focus:
            return 0.0

        early_doc = normalized_doc[:900]
        score = 0.0
        if f"{focus} la" in early_doc:
            score += 110.0
        if focus in early_doc[:450]:
            score += 45.0

        if focus.startswith("chung "):
            base_focus = focus.split(" ", 1)[1]
            if f"{base_focus} la" in early_doc and focus in early_doc:
                score += 90.0

        return score

    @classmethod
    def _score_lexical_relevance(cls, question, normalized_doc):
        ngrams, tokens = cls._build_query_ngrams(question)
        if not tokens:
            return 0.0

        score = cls._score_definition_match(question, normalized_doc)
        for phrase in ngrams:
            if phrase in normalized_doc:
                score += 35.0

        matched = 0
        for token in tokens:
            count = normalized_doc.count(token)
            if count:
                matched += 1
                score += min(count, 4) * 1.5

        if matched == len(tokens) and tokens:
            score += 12.0

        return score

    @staticmethod
    def _document_source_id(doc):
        metadata = getattr(doc, "metadata", {}) or {}
        return str(metadata.get("source") or metadata.get("title") or "<unknown>")

    def _select_diverse_sources(self, scored_items):
        if os.getenv("DIVERSIFY_SOURCES", "true").strip().lower() not in {"1", "true", "yes", "on"}:
            return [doc for *_, doc in scored_items[: self.k]]

        selected = []
        selected_ids = set()
        max_unique = min(self.k, int(os.getenv("MIN_SOURCE_DIVERSITY", "3")))
        min_strong_lexical = float(os.getenv("DIVERSITY_MIN_LEXICAL", "25"))
        strong_items = [item for item in scored_items if item[1] >= min_strong_lexical]
        diversity_pool = strong_items or scored_items

        for item in diversity_pool:
            doc = item[-1]
            source_id = self._document_source_id(doc)
            if source_id in selected_ids:
                continue
            selected.append(doc)
            selected_ids.add(source_id)
            if len(selected) >= max_unique:
                break

        for item in scored_items:
            doc = item[-1]
            if doc in selected:
                continue
            selected.append(doc)
            if len(selected) >= self.k:
                break

        return selected[: self.k]

    def invoke(self, question):
        query_embedding = self.embedding_model.embed_query(question)
        scored = []
        for embedding, doc, normalized_doc in zip(self.embeddings, self.documents, self.normalized_documents):
            dense_score = self._dense_similarity(query_embedding, embedding)
            lexical_score = self._score_lexical_relevance(question, normalized_doc)
            total_score = (lexical_score * 10.0) + dense_score
            scored.append((total_score, lexical_score, dense_score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        lexical_hits = [item for item in scored if item[1] > 0]
        if lexical_hits:
            return self._select_diverse_sources(lexical_hits)

        filtered = [item for item in scored if item[0] >= self.score_threshold]
        if not filtered:
            filtered = scored[: self.k]
        return self._select_diverse_sources(filtered)


# Backward-compatible names for older notebooks/imports.
SimpleVectorDB = InMemoryVectorStore
SimpleRetriever = HybridRetriever