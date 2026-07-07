import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

from modules.ingestion.sources import (
    SUPPORTED_USER_SOURCE_EXTS,
    list_available_sources,
    list_deleted_uploaded_sources,
    list_uploaded_source_names,
    mark_uploaded_source_deleted,
    restore_uploaded_source,
    sanitize_uploaded_filename,
)
from modules.rag.service import answer_question_with_workflow, clear_cached_retriever, rebuild_vector_store
from modules.settings import USER_SOURCES_DIR
from modules.ui.formatting import finalize_answer, format_retrieved_sources

WELCOME_MESSAGES = [
    {
        "role": "assistant",
        "content": (
            "Hello, I am TMedQA. Ask a question about the knowledge base, "
            "and I will answer with retrieved source evidence."
        ),
    }
]


def build_source_inventory_rows() -> List[List[str]]:
    rows = []
    for index, item in enumerate(list_available_sources(), start=1):
        rows.append([
            str(index),
            item["kind"],
            item["name"],
            item["status"],
            item["path"],
        ])
    return rows


def refresh_uploaded_source_dropdown():
    return gr.update(choices=list_uploaded_source_names(), value=None)


def refresh_source_inventory():
    return build_source_inventory_rows(), "Source list refreshed.", refresh_uploaded_source_dropdown()


def resolve_uploaded_temp_path(file_obj: Any) -> str:
    if isinstance(file_obj, (str, os.PathLike)):
        return str(file_obj)
    return str(getattr(file_obj, "name", ""))


def save_uploaded_sources(files: List[Any]):
    if not files:
        return build_source_inventory_rows(), "Choose one or more .docx, .txt, or .md files first.", refresh_uploaded_source_dropdown()

    USER_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []

    for file_obj in files:
        source_path = resolve_uploaded_temp_path(file_obj)
        if not source_path or not os.path.exists(source_path):
            skipped.append("<missing temporary file>")
            continue

        suffix = Path(source_path).suffix.lower()
        if suffix not in SUPPORTED_USER_SOURCE_EXTS:
            skipped.append(Path(source_path).name)
            continue

        filename = sanitize_uploaded_filename(source_path)
        target = USER_SOURCES_DIR / filename
        counter = 1
        while target.exists() and target.name not in list_deleted_uploaded_sources():
            target = USER_SOURCES_DIR / f"{target.stem}_{counter}{target.suffix}"
            counter += 1

        shutil.copy2(source_path, target)
        restore_uploaded_source(target.name)
        saved.append(target.name)

    clear_cached_retriever()
    parts = []
    if saved:
        parts.append(f"Uploaded {len(saved)} file(s): " + ", ".join(saved))
        parts.append("Retriever cache cleared. Click Rebuild vector store to refresh immediately, or ask the next question.")
    if skipped:
        parts.append("Skipped unsupported/missing file(s): " + ", ".join(skipped))

    return build_source_inventory_rows(), " | ".join(parts) if parts else "No files were uploaded.", refresh_uploaded_source_dropdown()


def delete_uploaded_source_file(filename: str):
    filename = (filename or "").strip()
    if not filename:
        return build_source_inventory_rows(), "Choose an uploaded source to delete.", refresh_uploaded_source_dropdown()

    safe_name = Path(filename).name
    target = USER_SOURCES_DIR / safe_name
    try:
        source_dir = USER_SOURCES_DIR.resolve()
        target_path = target.resolve()
        if target_path.parent != source_dir:
            raise ValueError("Invalid uploaded source path.")
        if target_path.suffix.lower() not in SUPPORTED_USER_SOURCE_EXTS:
            raise ValueError("Only uploaded .docx, .txt, and .md files can be deleted.")
        if not target_path.exists():
            return build_source_inventory_rows(), f"Uploaded source not found: {safe_name}", refresh_uploaded_source_dropdown()

        target_path.unlink()
        clear_cached_retriever()
        return (
            build_source_inventory_rows(),
            f"Deleted uploaded source: {safe_name}. Retriever cache cleared.",
            refresh_uploaded_source_dropdown(),
        )
    except Exception as exc:
        mark_uploaded_source_deleted(safe_name)
        clear_cached_retriever()
        return (
            build_source_inventory_rows(),
            f"Removed uploaded source from active index: {safe_name}. Physical delete was blocked by Windows: {exc}",
            refresh_uploaded_source_dropdown(),
        )


def rebuild_vector_store_for_chat():
    try:
        return rebuild_vector_store()
    except Exception as exc:
        return f"Rebuild failed: {exc}"


def rebuild_vector_store_for_sources():
    try:
        return build_source_inventory_rows(), rebuild_vector_store(), refresh_uploaded_source_dropdown()
    except Exception as exc:
        return build_source_inventory_rows(), f"Rebuild failed: {exc}", refresh_uploaded_source_dropdown()


def answer_user_question(question: str):
    question = (question or "").strip()
    if not question:
        return "", [], "Enter a question to run the demo."

    try:
        result, elapsed = answer_question_with_workflow(question)
    except Exception as exc:
        return "", [], f"Error: {exc}"

    documents = result.get("documents", [])
    answer = finalize_answer(question, result.get("generation", ""), documents)
    status = f"Done in {elapsed:.1f}s | {len(documents)} retrieved sources"
    return answer, format_retrieved_sources(documents), status


def submit_chat_message(message: str, history: List[Dict[str, str]]):
    message = (message or "").strip()
    history = list(history or WELCOME_MESSAGES)
    if not message:
        return history, "", [], "Enter a question to continue."

    history.append({"role": "user", "content": message})
    answer, sources, status = answer_user_question(message)
    history.append({"role": "assistant", "content": answer or status})
    return history, "", sources, status


def reset_chat_session():
    return list(WELCOME_MESSAGES), "", [], "Ready."


# Backward-compatible names for older UI wiring.
available_source_rows = build_source_inventory_rows
uploaded_source_dropdown_update = refresh_uploaded_source_dropdown
refresh_available_sources = refresh_source_inventory
uploaded_file_path = resolve_uploaded_temp_path
upload_sources = save_uploaded_sources
delete_uploaded_source = delete_uploaded_source_file
rebuild_vector_db_for_chat = rebuild_vector_store_for_chat
rebuild_vector_db_for_sources = rebuild_vector_store_for_sources
answer_question = answer_user_question
chat_submit = submit_chat_message
clear_chat = reset_chat_session