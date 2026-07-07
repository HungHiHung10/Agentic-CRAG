import time
from functools import lru_cache
from typing import Any, Dict, Tuple

from modules.models import load_embedding_backend, load_generation_model
from modules.rag.chains import build_answer_generator, build_query_rewriter, build_relevance_grader
from modules.rag.graph import CRAGWorkflow
from modules.retrieval.vector_store import create_vector_store
from modules.settings import RETRIEVAL_K, SIMILARITY_THRESHOLD


@lru_cache(maxsize=1)
def get_cached_retriever():
    embedding_backend = load_embedding_backend()
    vector_store = create_vector_store(embedding_backend, force_rebuild=False)
    if vector_store is None:
        raise RuntimeError("Could not create or load vector store. Check the Data directory and path configuration.")

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD},
    )


def clear_cached_retriever() -> None:
    get_cached_retriever.cache_clear()


def rebuild_vector_store() -> str:
    started = time.perf_counter()
    clear_cached_retriever()
    embedding_backend = load_embedding_backend()
    vector_store = create_vector_store(embedding_backend, force_rebuild=True)
    if vector_store is None:
        raise RuntimeError("Could not rebuild vector store. Check the Data directory and uploaded sources.")
    clear_cached_retriever()
    elapsed = time.perf_counter() - started
    return f"Vector store rebuilt in {elapsed:.1f}s. The next question will use the refreshed index."


def build_crag_workflow():
    generation_model = load_generation_model()
    workflow = CRAGWorkflow(
        retriever=get_cached_retriever(),
        grader_chain=build_relevance_grader(generation_model),
        rewriter_chain=build_query_rewriter(generation_model),
        generator_chain=build_answer_generator(generation_model),
    )
    return workflow.compile_workflow()


def answer_question_with_workflow(question: str) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    graph = build_crag_workflow()
    result = graph.invoke({"question": question, "documents": [], "generation": ""})
    return result, time.perf_counter() - started


# Backward-compatible names for older imports.
get_retriever = get_cached_retriever
clear_retriever_cache = clear_cached_retriever
rebuild_vector_db = rebuild_vector_store
build_graph = build_crag_workflow
invoke_question = answer_question_with_workflow