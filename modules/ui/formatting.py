from typing import Any, Dict, List

from modules.text_utils import clean_mojibake_label, friendly_source_name, normalize_for_search


def source_type(metadata: Dict[str, Any]) -> str:
    source = str(metadata.get("source", "")).lower()
    if "duckduckgo" in source or "web" in source:
        return "Web search"
    if source.startswith("user:"):
        return "Uploaded source"
    return "Local knowledge base"


def source_label(metadata: Dict[str, Any]) -> str:
    source = str(metadata.get("source", "")).strip()
    if source:
        return friendly_source_name(source)
    for key in ("title", "disease"):
        value = clean_mojibake_label(metadata.get(key, ""), fallback="")
        if value:
            return value
    return source_type(metadata)


def section_label(metadata: Dict[str, Any]) -> str:
    return clean_mojibake_label(metadata.get("section", ""), fallback="-")


def preview(text: str, limit: int = 260) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def is_vietnamese_text(text: str) -> bool:
    normalized = normalize_for_search(text)
    common_words = {"la", "gi", "benh", "trieu", "dieu", "tri", "chung", "dau"}
    tokens = set(normalized.split())
    return bool(tokens & common_words)


def citation_label(metadata: Dict[str, Any], index: int) -> str:
    source = source_label(metadata)
    section = section_label(metadata)
    if section and section != "-":
        return f"[{index}] {source} - {section}"
    return f"[{index}] {source}"


def answer_footer(question: str, documents: List[Any]) -> str:
    if is_vietnamese_text(question):
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
        label = citation_label(metadata, index)
        if label in seen:
            continue
        seen.add(label)
        lines.append(label)
        if len(lines) >= 5:
            break

    source_block = "\n".join(lines) if lines else "No retrieved source."
    return f"{disclaimer}\n\n{sources_title}:\n{source_block}"


def finalize_answer(question: str, answer: str, documents: List[Any]) -> str:
    answer = (answer or "").strip()
    if not answer:
        answer = "No answer was generated."
    if "Sources:" in answer or "Nguon:" in answer or "Ngu\u1ed3n:" in answer:
        return answer
    return f"{answer}\n\n---\n{answer_footer(question, documents)}"


def format_retrieved_sources(documents: List[Any]) -> List[List[str]]:
    rows = []
    for index, doc in enumerate(documents or [], start=1):
        metadata: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", str(doc)) or ""
        rows.append([
            str(index),
            source_type(metadata),
            source_label(metadata),
            section_label(metadata),
            preview(content),
            str(len(content)),
        ])
    return rows