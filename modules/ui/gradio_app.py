import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")
os.environ.setdefault("LLM_TIMEOUT_SECONDS", "15")
os.environ.setdefault("LLM_MAX_RETRIES", "0")

import gradio as gr

from modules.ingestion.sources import list_uploaded_source_names
from modules.ui.handlers import (
    WELCOME_MESSAGES,
    build_source_inventory_rows,
    submit_chat_message,
    delete_uploaded_source_file,
    refresh_source_inventory,
    rebuild_vector_store_for_chat,
    rebuild_vector_store_for_sources,
    save_uploaded_sources,
)

APP_TITLE = "TMedQA"
DEFAULT_QUESTION = "Li\u1ec7t d\u01b0\u01a1ng l\u00e0 g\u00ec?"

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
                    submit = gr.Button("←", variant="primary", elem_classes=["primary-button", "send-icon-button"], scale=1)

                with gr.Row(elem_id="action-row"):
                    rebuild_button = gr.Button("Rebuild vector store")

                status = gr.Markdown("Ready.", elem_id="status-line")
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
                    rebuild_sources_button = gr.Button("Rebuild vector store")
                with gr.Row(elem_id="delete-row"):
                    uploaded_source_select = gr.Dropdown(
                        choices=list_uploaded_source_names(),
                        label="Uploaded source to delete",
                        interactive=True,
                        scale=5,
                    )
                    delete_source_button = gr.Button("Delete uploaded source", scale=2)
                source_status = gr.Markdown("Ready.", elem_id="source-status-line")
                available_sources = gr.Dataframe(
                    value=build_source_inventory_rows(),
                    headers=["#", "Kind", "Name", "Status", "Path"],
                    datatype=["str", "str", "str", "str", "str"],
                    row_count=(0, "dynamic"),
                    column_count=(5, "fixed"),
                    wrap=True,
                    interactive=False,
                    label="Available sources",
                    elem_id="available-source-panel",
                )

        submit.click(submit_chat_message, inputs=[chatbox, chatbot], outputs=[chatbot, chatbox, retrieved_sources, status])
        chatbox.submit(submit_chat_message, inputs=[chatbox, chatbot], outputs=[chatbot, chatbox, retrieved_sources, status])
        rebuild_button.click(rebuild_vector_store_for_chat, outputs=[status])
        upload_button.click(save_uploaded_sources, inputs=[uploader], outputs=[available_sources, source_status, uploaded_source_select])
        refresh_button.click(refresh_source_inventory, outputs=[available_sources, source_status, uploaded_source_select])
        rebuild_sources_button.click(rebuild_vector_store_for_sources, outputs=[available_sources, source_status, uploaded_source_select])
        delete_source_button.click(
            delete_uploaded_source_file,
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