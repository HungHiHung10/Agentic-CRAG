import re
import unicodedata

SOURCE_DISPLAY_NAMES = {
    "Noi_Khoa_YHCT_GS_Hoang_Bao_Chau": "Noi Khoa YHCT - GS Hoang Bao Chau",
    "Benh_Ngu_Quan_YHCT": "Benh Ngu Quan YHCT",
    "Nhi_Khoa_YHCT": "Nhi Khoa YHCT",
    "noi-khoa-y-hoc-co-truyen": "Noi Khoa Y Hoc Co Truyen",
}

MOJIBAKE_PAREN_RE = re.compile("\\([^)]*(?:\u00c3|\u00c6|\u00e2|\u20ac)[^)]*\\)")


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("\u0111", "d")


def normalize_for_search(text: str) -> str:
    text = strip_accents(text)
    text = re.sub(r"\bti\b", "ty", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    markers = ("\u00c3", "\u00c6", "\u00e2", "\u20ac")
    return any(marker in text for marker in markers)


def clean_mojibake_label(text: str, fallback: str = "-") -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return fallback
    compact = MOJIBAKE_PAREN_RE.sub("", compact)
    compact = re.sub(r"\s+", " ", compact).strip(" -:;>")
    if not compact:
        return fallback
    if looks_mojibake(compact):
        clean_parts = [part.strip(" -:;>") for part in re.split(r"\s*>\s*|\s+-\s+", compact) if not looks_mojibake(part)]
        compact = " > ".join(part for part in clean_parts if part)
    compact = compact.strip(" -:;>")
    if not compact or looks_mojibake(compact):
        return fallback
    return compact


def friendly_source_name(source: str) -> str:
    source = str(source or "").strip()
    if source in SOURCE_DISPLAY_NAMES:
        return SOURCE_DISPLAY_NAMES[source]
    if source.startswith("user:"):
        return "Uploaded: " + source[5:]
    return clean_mojibake_label(source, fallback=source or "Unknown source")

