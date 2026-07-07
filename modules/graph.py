# modules/graph.py
import os
import re
import unicodedata
from typing import List, TypedDict

import numpy as np
from ddgs import DDGS
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from config import HYBRID_SCORE_THRESHOLD
from modules.components import get_rerank_score


class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]


class CRAGWorkflow:
    def __init__(self, retriever, grader_chain, rewriter_chain, generator_chain):
        self.retriever = retriever
        self.grader = grader_chain
        self.rewriter = rewriter_chain
        self.generator = generator_chain
        self.llm_failed = False
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def retrieve(self, state):
        print("\n[PROCESS] RETRIEVAL")
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs, "question": state["question"]}

    def grade_documents_hybrid(self, state):
        print("[PROCESS] HYBRID GRADING")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"

        if documents:
            if not self.llm_failed:
                try:
                    inputs = [{"question": question, "document": doc.page_content} for doc in documents]
                    llm_results = self.grader.batch(inputs)
                except Exception as exc:
                    print(f"[WARNING] LLM grader batch failed: {exc}")
                    self.llm_failed = True
                    llm_results = [{"binary_score": "yes"} for _ in documents]
            else:
                llm_results = [{"binary_score": "yes"} for _ in documents]

            for doc, llm_res in zip(documents, llm_results):
                llm_val = str(llm_res.get("binary_score", "no")).strip().lower()
                llm_norm = 1.0 if llm_val == "yes" else 0.0

                rerank_score = get_rerank_score(question, doc.page_content)
                if rerank_score is None:
                    final_score = llm_norm
                else:
                    rerank_norm = 1 / (1 + np.exp(-rerank_score))
                    final_score = (rerank_norm * 0.6) + (llm_norm * 0.4)

                if final_score >= HYBRID_SCORE_THRESHOLD:
                    filtered_docs.append(doc)

        if not filtered_docs:
            if documents:
                print("[INFO] No document passed grading; keeping top local retrieval results as fallback evidence.")
                filtered_docs = documents
                web_search = "Yes" if self.enable_web_search else "No"
            else:
                web_search = "Yes" if self.enable_web_search else "No"

        return {"documents": filtered_docs, "question": question, "web_search_needed": web_search}

    def rewrite_query(self, state):
        print("[PROCESS] REWRITE QUERY")
        try:
            if self.llm_failed:
                raise RuntimeError("LLM unavailable in this request")
            new_q = self.rewriter.invoke({"question": state["question"]})
        except Exception as exc:
            print(f"[WARNING] Query rewrite failed, using original question: {exc}")
            self.llm_failed = True
            new_q = state["question"]
        return {"documents": state["documents"], "question": new_q}

    def web_search(self, state):
        print("[PROCESS] WEB SEARCH (DuckDuckGo)")
        question = state["question"]
        documents = state.get("documents", []) or []

        if not self.enable_web_search:
            print("[INFO] Web search disabled. Set ENABLE_WEB_SEARCH=true to enable it.")
            return {"documents": documents, "question": question}

        try:
            with DDGS() as ddg:
                results = list(ddg.text(question, max_results=3))

            if results:
                web_results_content = "\n\n".join(
                    f"Title: {item.get('title', '')}\n"
                    f"Link: {item.get('href', '')}\n"
                    f"Content: {item.get('body', item.get('snippet', ''))}"
                    for item in results
                )
            else:
                web_results_content = "[ERROR] Not Found"
        except Exception as exc:
            print(f"[ERROR] {exc}")
            web_results_content = f"[ERROR] {exc}"

        documents.append(Document(page_content=web_results_content, metadata={"source": "duckduckgo_search"}))
        return {"documents": documents, "question": question}

    @staticmethod
    def _normalize(text):
        text = unicodedata.normalize("NFKC", text or "").lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("\u0111", "d")
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _question_focus(cls, question):
        clean = re.sub(r"answer\s+in\s+english\s+only\.?", " ", question or "", flags=re.I).strip()
        normalized = cls._normalize(clean)
        match = re.search(r"(.+?)\s+(?:la\s+gi|nghia\s+la\s+gi)\??$", normalized, flags=re.I)
        if match:
            normalized_focus = match.group(1).strip(" ?:;,.\n\t")
            words = normalized_focus.split()
            original_words = re.findall(r"\w+", clean, flags=re.UNICODE)
            if len(original_words) >= len(words):
                return " ".join(original_words[: len(words)])
            return normalized_focus

        tokens = [
            tok for tok in re.findall(r"\w+", clean, flags=re.UNICODE)
            if cls._normalize(tok) not in {"la", "gi"}
        ]
        return " ".join(tokens[:3]).strip() or clean

    @classmethod
    def _relevant_excerpt(cls, text, focus):
        compact = " ".join((text or "").split())
        normalized_text = cls._normalize(compact)
        normalized_focus = cls._normalize(focus)
        index = normalized_text.find(normalized_focus) if normalized_focus else -1
        if index < 0:
            return compact[:900]

        start = max(0, index - 260)
        end = min(len(compact), index + 760)
        excerpt = compact[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(compact):
            excerpt += "..."
        return excerpt

    @classmethod
    def _source_label(cls, doc, index):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("title") or metadata.get("disease") or "unknown source"
        section = metadata.get("section") or ""
        label = f"[{index}] {source}"
        if section:
            label += f" - {section}"
        return label

    @classmethod
    def _fallback_answer(cls, question, documents):
        if not documents:
            return (
                "LLM API \u0111ang l\u1ed7i v\u00e0 h\u1ec7 th\u1ed1ng ch\u01b0a t\u00ecm \u0111\u01b0\u1ee3c "
                "ng\u1eef c\u1ea3nh li\u00ean quan trong t\u00e0i li\u1ec7u."
            )

        focus = cls._question_focus(question)
        excerpts = []
        for index, doc in enumerate(documents[:3], start=1):
            text = doc.page_content or ""
            excerpt = cls._relevant_excerpt(text, focus)
            if excerpt:
                excerpts.append(f"- {cls._source_label(doc, index)}\n  {excerpt}")

        if not excerpts:
            return (
                "LLM API \u0111ang l\u1ed7i v\u00e0 h\u1ec7 th\u1ed1ng ch\u01b0a tr\u00edch xu\u1ea5t "
                "\u0111\u01b0\u1ee3c \u0111o\u1ea1n t\u00e0i li\u1ec7u li\u00ean quan."
            )

        if re.search(r"(?:l\u00e0 g\u00ec|la gi|ngh\u0129a l\u00e0 g\u00ec|nghia la gi)", question or "", flags=re.I):
            lead = (
                "LLM API \u0111ang l\u1ed7i n\u00ean h\u1ec7 th\u1ed1ng ch\u01b0a t\u1ed5ng h\u1ee3p "
                "\u0111\u01b0\u1ee3c c\u00e2u tr\u1ea3 l\u1eddi ho\u00e0n ch\u1ec9nh. "
                f"D\u01b0\u1edbi \u0111\u00e2y l\u00e0 c\u00e1c b\u1eb1ng ch\u1ee9ng li\u00ean quan nh\u1ea5t t\u1edbi '{focus}':\n\n"
            )
        else:
            lead = (
                "LLM API \u0111ang l\u1ed7i n\u00ean h\u1ec7 th\u1ed1ng tr\u1ea3 l\u1eddi theo d\u1ea1ng "
                f"tr\u00edch xu\u1ea5t t\u00e0i li\u1ec7u li\u00ean quan t\u1edbi '{focus}':\n\n"
            )

        return lead + "\n\n".join(excerpts)
    def generate_answer(self, state):
        print("[PROCESS] GENERATE")
        try:
            if self.llm_failed:
                raise RuntimeError("LLM unavailable in this request")
            gen = self.generator.invoke({"context": state["documents"], "question": state["question"]})
        except Exception as exc:
            print(f"[WARNING] Generation failed, returning retrieval fallback: {exc}")
            self.llm_failed = True
            gen = self._fallback_answer(state["question"], state["documents"])
        return {"generation": gen, "documents": state["documents"]}

    def decide_to_generate(self, state):
        if state["web_search_needed"] == "Yes":
            return "rewrite_query"
        return "generate_answer"

    def build(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents_hybrid)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
        )
        workflow.add_edge("rewrite_query", "web_search")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()
