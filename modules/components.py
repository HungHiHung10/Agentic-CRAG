# modules/components.py
import os
from functools import lru_cache
from operator import itemgetter

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool

from config import RERANKER_MODEL_NAME


def _flag_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_reranker_model():
    if not _flag_enabled("RERANKER_ENABLED"):
        print("[INFO] Re-ranker disabled. Set RERANKER_ENABLED=true to use it.")
        return None

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        print(f"[WARNING] Re-ranker unavailable: {exc}")
        return None

    print(f"[INIT] Loading Re-ranker: {RERANKER_MODEL_NAME}...")
    try:
        return CrossEncoder(RERANKER_MODEL_NAME, device="cpu", local_files_only=True)
    except Exception as exc:
        print(f"[WARNING] Re-ranker disabled because it could not be loaded locally: {exc}")
        return None


def get_rerank_score(question: str, document: str):
    reranker = get_reranker_model()
    if reranker is None:
        return None

    try:
        return float(reranker.predict([(question, document)])[0])
    except Exception as exc:
        print(f"[WARNING] Re-ranker scoring failed: {exc}")
        return None


# --- CHAINS ---
def build_grader_chain(llm):
    def parse_grader_output(text: str):
        return {"binary_score": "yes" if "yes" in text.lower() else "no"}

    sys_prompt = """
You are a strict relevance grader for a Vietnamese traditional medicine (YHCT) retrieval-augmented QA system.
Your only job is to decide whether the document contains direct evidence that can answer the user's question.
Return "yes" if the document mentions the exact medical term, a direct synonym, or a clearly related explanation.
Return "no" if the document is only about a generic or different-looking term.
Important examples:
- Do not confuse "chung ung" with "ung thu".
- Do not confuse "chung ty" with "chung te".
- Do not confuse one "duong" term with another unless the target phrase is present.
Return only one word: yes or no.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ])
    return prompt | llm | StrOutputParser() | parse_grader_output


def build_rewriter_chain(llm):
    sys_prompt = """
Rewrite the user question into a concise search query.
Preserve Vietnamese medical terms exactly when present, including accents.
Do not translate the key term. Return only the rewritten query.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Question:\n{question}\n\nImproved search query:"),
    ])
    return prompt | llm | StrOutputParser()


def build_generator_chain(llm):
    prompt_text = """
You are a precise Vietnamese traditional medicine (YHCT) question-answering assistant.
Use only the provided Context. Do not use outside knowledge.
If the Context is not enough, say exactly:
"Tai lieu hien tai khong chua du thong tin chinh xac de tra loi cau hoi nay."

Rules:
- Answer in the same language as the user's question. If the question is Vietnamese, answer in Vietnamese.
- Be concise, practical, and faithful to the retrieved context.
- Do not confuse similar medical terms. For example, "chung ung" is not "ung thu", and "chung ty" is not "chung te".
- Cite the most relevant source markers such as [1], [2] when making claims.
- Do not provide diagnosis, prescription, or dosage advice beyond what the context directly supports.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    def format_docs(docs):
        formatted = []
        for index, doc in enumerate(docs or [], start=1):
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source") or metadata.get("title") or metadata.get("disease") or "unknown source"
            section = metadata.get("section") or metadata.get("title") or ""
            label = f"[{index}] {source}"
            if section:
                label += f" - {section}"
            formatted.append(f"{label}\n{doc.page_content}")
        return "\n\n".join(formatted)

    return (
        {"context": (itemgetter("context") | RunnableLambda(format_docs)), "question": itemgetter("question")}
        | prompt | llm | StrOutputParser()
    )

# --- TOOLS ---
def build_web_search_tool():
    ddg_search = DuckDuckGoSearchResults(backend="api", num_results=3)
    return Tool(name="web_search", func=ddg_search.invoke, description="Search internet")
