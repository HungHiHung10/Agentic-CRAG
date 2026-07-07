# app.py
import os
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "15")
os.environ.setdefault("LLM_MAX_RETRIES", "0")

import gradio as gr

from config import RETRIEVAL_K, SIMILARITY_THRESHOLD, USER_SOURCES_DIR


APP_TITLE = "TMedQA"
DEFAULT_QUESTION = "Li\u1ec7t d\u01b0\u01a1ng l\u00e0 g\u00ec?"
SUPPORTED_UPLOAD_SUFFIXES = {".docx", ".txt", ".md"}
DELETED_USER_SOURCES_FILE = USER_SOURCES_DIR / ".deleted_sources.txt"
WELCOME_MESSAGES = [
    {
        "role": "assistant",
        "content": (
            "Hello, I am TMedQA. Ask a question about the knowledge base, "
            "and I will answer with retrieved source evidence."
        ),
    }
]


def _deleted_uploaded_source_names() -> set[str]:
    if not DELETED_USER_SOURCES_FILE.exists():
        return set()
    return {
        line.strip()
        for line in DELETED_USER_SOURCES_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }


def _mark_uploaded_source_deleted(filename: str) -> None:
    USER_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    deleted = _deleted_uploaded_source_names()
    deleted.add(filename)
    DELETED_USER_SOURCES_FILE.write_text("\n".join(sorted(deleted)) + "\n", encoding="utf-8")


@lru_cache(maxsize=1)
def get_retriever():
    from modules.ingestion import create_vector_db
    from utils import load_embedding_model

    embed_model = load_embedding_model()
    db = create_vector_db(embed_model, force_rebuild=False)
    if db is None:
        raise RuntimeError("Could not create or load vector DB. Check the Data directory and path configuration.")

    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD},
    )


def rebuild_vector_db():
    from modules.ingestion import create_vector_db
    from utils import load_embedding_model

    started = time.perf_counter()
    get_retriever.cache_clear()
    embed_model = load_embedding_model()
    db = create_vector_db(embed_model, force_rebuild=True)
    if db is None:
        raise RuntimeError("Could not rebuild vector DB. Check the Data directory and uploaded sources.")
    get_retriever.cache_clear()
    elapsed = time.perf_counter() - started
    return f"Vector DB rebuilt in {elapsed:.1f}s. The next question will use the refreshed index."


def rebuild_vector_db_for_chat():
    try:
        return rebuild_vector_db()
    except Exception as exc:
        return f"Rebuild failed: {exc}"


def rebuild_vector_db_for_sources():
    try:
        return _available_source_rows(), rebuild_vector_db(), _uploaded_source_dropdown_update()
    except Exception as exc:
        return _available_source_rows(), f"Rebuild failed: {exc}", _uploaded_source_dropdown_update()


def build_app():
    from modules.components import build_generator_chain, build_grader_chain, build_rewriter_chain
    from modules.graph import CRAGWorkflow
    from utils import load_llm_model

    llm = load_llm_model()
    workflow = CRAGWorkflow(
        retriever=get_retriever(),
        grader_chain=build_grader_chain(llm),
        rewriter_chain=build_rewriter_chain(llm),
        generator_chain=build_generator_chain(llm),
    )
    return workflow.build()


def _source_type(metadata: Dict[str, Any]) -> str:
    source = str(metadata.get("source", "")).lower()
    if "duckduckgo" in source or "web" in source:
        return "Web search"
    if source.startswith("user:"):
        return "Uploaded source"
    return "Local knowledge base"


def _source_label(metadata: Dict[str, Any]) -> str:
    for key in ("source", "title", "disease"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    return _source_type(metadata)


def _section_label(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("section", "")).strip() or "-"


def _preview(text: str, limit: int = 260) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."



def _is_vietnamese_text(text: str) -> bool:
    lowered = (text or "").lower()
    vietnamese_markers = "\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0\u00e1\u00e0\u1ea3\u00e3\u1ea1\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u00e9\u00e8\u1ebb\u1ebd\u1eb9\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u00ed\u00ec\u1ec9\u0129\u1ecb\u00f3\u00f2\u1ecf\u00f5\u1ecd\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3\u00fa\u00f9\u1ee7\u0169\u1ee5\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u00fd\u1ef3\u1ef7\u1ef9\u1ef5"
    common_words = {" la ", " gi", " benh", " trieu", " dieu tri", " dau "}
    return any(ch in lowered for ch in vietnamese_markers) or any(word in f" {lowered} " for word in common_words)


def _citation_label(metadata: Dict[str, Any], index: int) -> str:
    source = _source_label(metadata)
    section = _section_label(metadata)
    if section and section != "-":
        return f"[{index}] {source} - {section}"
    return f"[{index}] {source}"


def _answer_footer(question: str, documents: List[Any]) -> str:
    if _is_vietnamese_text(question):
        disclaimer = (
            "L\u01b0u \u00fd: C\u00e2u tr\u1ea3 l\u1eddi ch\u1ec9 d\u00f9ng \u0111\u1ec3 tham kh\u1ea3o t\u1eeb t\u00e0i li\u1ec7u \u0111\u00e3 n\u1ea1p, "
            "kh\u00f4ng thay th\u1ebf ch\u1ea9n \u0111o\u00e1n ho\u1eb7c \u0111i\u1ec1u tr\u1ecb c\u1ee7a nh\u00e2n vi\u00ean y t\u1ebf."
        )
        sources_title = "Ngu\u1ed3n"
    else:
        disclaimer = (
            "Note: This answer is for reference from the indexed sources only "
            "and does not replace medical diagnosis or treatment."
        )
        sources_title = "Sources"

    lines = []
    seen = set()
    for index, doc in enumerate(documents or [], start=1):
        metadata: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
        label = _citation_label(metadata, index)
        if label in seen:
            continue
        seen.add(label)
        lines.append(label)
        if len(lines) >= 5:
            break

    source_block = "\n".join(lines) if lines else "No retrieved source."
    return f"{disclaimer}\n\n{sources_title}:\n{source_block}"


def _finalize_answer(question: str, answer: str, documents: List[Any]) -> str:
    answer = (answer or "").strip()
    if not answer:
        answer = "No answer was generated."
    if "Sources:" in answer or "Nguon:" in answer:
        return answer
    return f"{answer}\n\n---\n{_answer_footer(question, documents)}"

def _format_sources(documents: List[Any]) -> List[List[str]]:
    rows = []
    for index, doc in enumerate(documents or [], start=1):
        metadata: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", str(doc)) or ""
        rows.append([
            str(index),
            _source_type(metadata),
            _source_label(metadata),
            _section_label(metadata),
            _preview(content),
            str(len(content)),
        ])
    return rows


def _available_source_rows() -> List[List[str]]:
    from modules.ingestion import list_source_inventory

    rows = []
    for index, item in enumerate(list_source_inventory(), start=1):
        rows.append([
            str(index),
            item["kind"],
            item["name"],
            item["status"],
            item["path"],
        ])
    return rows


def _uploaded_source_names() -> List[str]:
    USER_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    deleted = _deleted_uploaded_source_names()
    for path_obj in sorted(USER_SOURCES_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path_obj.is_file() or path_obj.name.startswith("~$") or path_obj.name == "test-user-source.txt":
            continue
        if path_obj.name in deleted:
            continue
        if path_obj.suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES:
            names.append(path_obj.name)
    return names


def _uploaded_source_dropdown_update():
    return gr.update(choices=_uploaded_source_names(), value=None)


def refresh_available_sources():
    return _available_source_rows(), "Source list refreshed.", _uploaded_source_dropdown_update()


def _uploaded_file_path(file_obj: Any) -> str:
    if isinstance(file_obj, (str, os.PathLike)):
        return str(file_obj)
    return str(getattr(file_obj, "name", ""))


def _safe_upload_name(path: str) -> str:
    name = Path(path).name or "source"
    clean = "".join(ch if ch.isalnum() or ch in "._ -" else "_" for ch in name).strip(" ._")
    return clean or "source"


def _unmark_uploaded_source_deleted(filename: str) -> None:
    deleted = _deleted_uploaded_source_names()
    if filename in deleted:
        deleted.remove(filename)
        DELETED_USER_SOURCES_FILE.write_text("\n".join(sorted(deleted)) + "\n", encoding="utf-8")


def upload_sources(files: List[Any]):
    if not files:
        return _available_source_rows(), "Choose one or more .docx, .txt, or .md files first.", _uploaded_source_dropdown_update()

    USER_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []

    for file_obj in files:
        source_path = _uploaded_file_path(file_obj)
        if not source_path or not os.path.exists(source_path):
            skipped.append("<missing temporary file>")
            continue

        suffix = Path(source_path).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
            skipped.append(Path(source_path).name)
            continue

        filename = _safe_upload_name(source_path)
        target = USER_SOURCES_DIR / filename
        counter = 1
        while target.exists() and target.name not in _deleted_uploaded_source_names():
            target = USER_SOURCES_DIR / f"{target.stem}_{counter}{target.suffix}"
            counter += 1

        shutil.copy2(source_path, target)
        _unmark_uploaded_source_deleted(target.name)
        saved.append(target.name)

    get_retriever.cache_clear()
    parts = []
    if saved:
        parts.append(f"Uploaded {len(saved)} file(s): " + ", ".join(saved))
        parts.append("Retriever cache cleared. Click Rebuild vector DB to refresh immediately, or ask the next question.")
    if skipped:
        parts.append("Skipped unsupported/missing file(s): " + ", ".join(skipped))

    return _available_source_rows(), " | ".join(parts) if parts else "No files were uploaded.", _uploaded_source_dropdown_update()


def delete_uploaded_source(filename: str):
    filename = (filename or "").strip()
    if not filename:
        return _available_source_rows(), "Choose an uploaded source to delete.", _uploaded_source_dropdown_update()

    safe_name = Path(filename).name
    target = USER_SOURCES_DIR / safe_name
    try:
        source_dir = USER_SOURCES_DIR.resolve()
        target_path = target.resolve()
        if target_path.parent != source_dir:
            raise ValueError("Invalid uploaded source path.")
        if target_path.suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError("Only uploaded .docx, .txt, and .md files can be deleted.")
        if not target_path.exists():
            return _available_source_rows(), f"Uploaded source not found: {safe_name}", _uploaded_source_dropdown_update()

        target_path.unlink()
        get_retriever.cache_clear()
        return (
            _available_source_rows(),
            f"Deleted uploaded source: {safe_name}. Retriever cache cleared.",
            _uploaded_source_dropdown_update(),
        )
    except Exception as exc:
        _mark_uploaded_source_deleted(safe_name)
        get_retriever.cache_clear()
        return (
            _available_source_rows(),
            f"Removed uploaded source from active index: {safe_name}. Physical delete was blocked by Windows: {exc}",
            _uploaded_source_dropdown_update(),
        )


def answer_question(question: str):
    question = (question or "").strip()
    if not question:
        return "", [], "Enter a question to run the demo."

    started = time.perf_counter()
    try:
        app = build_app()
        result = app.invoke({"question": question, "documents": [], "generation": ""})
    except Exception as exc:
        return "", [], f"Error: {exc}"

    elapsed = time.perf_counter() - started
    documents = result.get("documents", [])
    answer = _finalize_answer(question, result.get("generation", ""), documents)
    status = f"Done in {elapsed:.1f}s | {len(documents)} retrieved sources"
    return answer, _format_sources(documents), status


def chat_submit(message: str, history: List[Dict[str, str]]):
    message = (message or "").strip()
    history = list(history or WELCOME_MESSAGES)
    if not message:
        return history, "", [], "Enter a question to continue."

    history.append({"role": "user", "content": message})
    answer, sources, status = answer_question(message)
    history.append({"role": "assistant", "content": answer or status})
    return history, "", sources, status


def clear_chat():
    return list(WELCOME_MESSAGES), "", [], "Ready."


CSS = """
:root {
    --body-background-fill: #f8fafc;
    --body-text-color: #111827;
    --block-background-fill: #ffffff;
    --block-border-color: #cbd5e1;
    --input-background-fill: #ffffff;
    --input-border-color: #94a3b8;
    --button-primary-background-fill: #0f766e;
    --button-primary-background-fill-hover: #115e59;
    --button-primary-text-color: #ffffff;
}
body {
    background: #f8fafc !important;
    color: #111827 !important;
}
.gradio-container {
    background: #f8fafc !important;
    color: #111827 !important;
}
.gradio-container, .gradio-container label, .gradio-container p,
.gradio-container span, .gradio-container textarea, .gradio-container input,
.gradio-container table, .gradio-container th, .gradio-container td {
    color: #111827 !important;
}
.gradio-container ::placeholder {
    color: #475569 !important;
    opacity: 1 !important;
}
.gradio-container textarea, .gradio-container input {
    background: #ffffff !important;
    border-color: #94a3b8 !important;
}
.gradio-container label, .gradio-container .label-wrap {
    color: #0f172a !important;
    font-weight: 600 !important;
}
#title-block {
    margin-bottom: 8px;
}
#title-block h1 {
    margin-bottom: 2px;
    letter-spacing: 0;
    color: #0f172a !important;
}
#title-block p {
    margin-top: 0;
    color: #334155 !important;
}
#chatbot {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff !important;
}
#chatbox textarea {
    min-height: 54px !important;
}
#source-panel, #available-source-panel {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px;
    background: #ffffff !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
#source-panel *, #available-source-panel * {
    color: #111827 !important;
}
#source-panel th, #available-source-panel th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}
#status-line, #source-status-line {
    color: #334155 !important;
    min-height: 28px;
}
#action-row, #upload-row, #delete-row, #composer-row {
    align-items: center !important;
}
#composer-row {
    gap: 8px !important;
}
button.primary-button {
    min-height: 42px;
    background: #0f766e !important;
    border-color: #0f766e !important;
    color: #ffffff !important;
}
button.primary-button * {
    color: #ffffff !important;
}
.send-icon-button {
    min-width: 48px !important;
    max-width: 56px !important;
    height: 54px !important;
    padding: 0 !important;
    font-size: 22px !important;
    line-height: 1 !important;
}
button:not(.primary-button) {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #111827 !important;
}
.gradio-container a {
    color: #0f766e !important;
    font-weight: 600;
}
"""


def create_demo():
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_id="title-block"):
            gr.Markdown(f"# {APP_TITLE}\nAgentic Graph Correctness RAG")

        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    value=list(WELCOME_MESSAGES),
                    label="Conversation",
                    elem_id="chatbot",
                    height=520,
                    layout="bubble",
                    show_label=False,
                    placeholder="Ask a question about the indexed sources.",
                )
                with gr.Row(elem_id="composer-row"):
                    chatbox = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about the indexed sources...",
                        lines=2,
                        max_lines=5,
                        autofocus=True,
                        scale=18,
                        elem_id="chatbox",
                    )
                    submit = gr.Button(">", variant="primary", elem_classes=["primary-button", "send-icon-button"], scale=1)

                with gr.Row(elem_id="action-row"):
                    # clear = gr.Button("Clear chat")
                    rebuild_button = gr.Button("Rebuild vector DB")

                # status = gr.Markdown("Ready.", elem_id="status-line")
                with gr.Accordion("Retrieved evidence", open=True):
                    retrieved_sources = gr.Dataframe(
                        headers=["#", "Type", "Source", "Section", "Preview", "Chars"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        row_count=(0, "dynamic"),
                        column_count=(6, "fixed"),
                        wrap=True,
                        interactive=False,
                        label="Retrieved sources",
                        elem_id="source-panel",
                    )

            with gr.Tab("Sources"):
                gr.Markdown("## Available sources")
                uploader = gr.File(
                    label="Upload source files (.docx, .txt, .md)",
                    file_count="multiple",
                    file_types=[".docx", ".txt", ".md"],
                )
                with gr.Row(elem_id="upload-row"):
                    upload_button = gr.Button("Add uploaded sources", variant="primary", elem_classes=["primary-button"])
                    refresh_button = gr.Button("Refresh source list")
                    rebuild_sources_button = gr.Button("Rebuild vector DB")
                with gr.Row(elem_id="delete-row"):
                    uploaded_source_select = gr.Dropdown(
                        choices=_uploaded_source_names(),
                        label="Uploaded source to delete",
                        interactive=True,
                        scale=5,
                    )
                    delete_source_button = gr.Button("Delete uploaded source", scale=2)
                source_status = gr.Markdown("Ready.", elem_id="source-status-line")
                available_sources = gr.Dataframe(
                    value=_available_source_rows(),
                    headers=["#", "Kind", "Name", "Status", "Path"],
                    datatype=["str", "str", "str", "str", "str"],
                    row_count=(0, "dynamic"),
                    column_count=(5, "fixed"),
                    wrap=True,
                    interactive=False,
                    label="Available sources",
                    elem_id="available-source-panel",
                )

        submit.click(
            chat_submit,
            inputs=[chatbox, chatbot],
            outputs=[chatbot, chatbox, retrieved_sources, status],
        )
        chatbox.submit(
            chat_submit,
            inputs=[chatbox, chatbot],
            outputs=[chatbot, chatbox, retrieved_sources, status],
        )
        # clear.click(clear_chat, outputs=[chatbot, chatbox, retrieved_sources, status])
        rebuild_button.click(rebuild_vector_db_for_chat, outputs=[status])
        upload_button.click(upload_sources, inputs=[uploader], outputs=[available_sources, source_status, uploaded_source_select])
        refresh_button.click(refresh_available_sources, outputs=[available_sources, source_status, uploaded_source_select])
        rebuild_sources_button.click(rebuild_vector_db_for_sources, outputs=[available_sources, source_status, uploaded_source_select])
        delete_source_button.click(
            delete_uploaded_source,
            inputs=[uploaded_source_select],
            outputs=[available_sources, source_status, uploaded_source_select],
        )

    return demo


def launch_demo():
    launch_kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        "show_error": True,
        "theme": gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        "css": CSS,
    }

    port = os.getenv("GRADIO_SERVER_PORT", "").strip()
    if port:
        launch_kwargs["server_port"] = int(port)

    return create_demo().launch(**launch_kwargs)


if __name__ == "__main__":
    launch_demo()

