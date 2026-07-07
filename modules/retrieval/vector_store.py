import os
import shutil

from langchain_chroma import Chroma

from modules.settings import CHROMA_DB_DIR
from modules.ingestion.sources import split_uploaded_sources
from modules.ingestion.splitters import (
    split_builtin_source_with_fallback,
    split_benh_ngu_quan,
    split_nhi_khoa,
    split_noi_khoa_general,
    split_noi_khoa_gs_chau,
)
from modules.retrieval.retriever import InMemoryVectorStore

def create_vector_store(embedding_model, force_rebuild=False):
    use_chroma = os.getenv("VECTOR_STORE_BACKEND", "memory").strip().lower() == "chroma"

    if use_chroma and os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
        try:
            print(f"[INFO] ChromaDB already exists at {CHROMA_DB_DIR}. Loading...")
            return Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embedding_model,
                collection_name="y_hoc_co_truyen",
            )
        except Exception as exc:
            print(f"[WARNING] Failed to load ChromaDB, rebuilding in memory: {exc}")
            use_chroma = False

    if use_chroma and force_rebuild and os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR)
            print("[SUCCESS] Removed old Chroma directory")
        except Exception as exc:
            print(f"[WARNING] Could not remove old Chroma directory: {exc}")

    splitters = [
        ("noi_khoa_gs_chau", "Noi_Khoa_YHCT_GS_Hoang_Bao_Chau", split_noi_khoa_gs_chau),
        ("benh_ngu_quan", "Benh_Ngu_Quan_YHCT", split_benh_ngu_quan),
        ("nhi_khoa", "Nhi_Khoa_YHCT", split_nhi_khoa),
        ("noi_khoa_general", "noi-khoa-y-hoc-co-truyen", split_noi_khoa_general),
    ]

    all_chunks = []
    for file_key, source_id, splitter in splitters:
        docs = split_builtin_source_with_fallback(file_key, source_id, splitter)
        print(f"[INFO] {source_id}: {len(docs)} chunks")
        all_chunks.extend(docs)

    user_chunks = split_uploaded_sources()
    all_chunks.extend(user_chunks)

    if not all_chunks:
        print("[WARNING] No chunks created. Check file paths.")
        return None

    print(f"[PROCESS] Creating vector store with {len(all_chunks)} chunks...")
    if use_chroma:
        try:
            chroma_db = Chroma.from_documents(
                documents=all_chunks,
                collection_name="y_hoc_co_truyen",
                embedding=embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=CHROMA_DB_DIR,
            )
            print(f"[SUCCESS] Vector store created with count: {chroma_db._collection.count()}")
            return chroma_db
        except Exception as exc:
            print(f"[WARNING] ChromaDB unavailable, using in-memory vector store: {exc}")

    vector_store = InMemoryVectorStore.from_documents(all_chunks, embedding_model)
    print(f"[SUCCESS] In-memory vector store created with count: {len(all_chunks)}")
    return vector_store


# Backward-compatible name for older imports.
create_vector_db = create_vector_store
