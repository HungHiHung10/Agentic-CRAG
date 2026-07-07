# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _data_file(pattern: str) -> str:
    matches = sorted(DATA_DIR.glob(pattern))
    if matches:
        return str(matches[0])
    return str(DATA_DIR / pattern.replace("*", ""))


_load_dotenv_file(BASE_DIR / ".env")

# --- MODEL CONFIG ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# API-backed LLM config. Values default to the provider chain declared in .env.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_PROVIDER_CHAIN = [
    provider.strip().lower()
    for provider in os.getenv("LLM_PROVIDER_CHAIN", LLM_PROVIDER).split(",")
    if provider.strip()
]
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower()
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()

LLM_API_PROVIDERS = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
        "base_url": "https://api.mistral.ai/v1",
    },
    "cerebras": {
        "api_key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "base_url": "https://api.cerebras.ai/v1",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_url": None,
    },
}

# --- PATHS ---
DATA_DIR = BASE_DIR / "Data"
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "database"))
USER_SOURCES_DIR = Path(os.getenv("USER_SOURCES_DIR", str(BASE_DIR / "user_sources")))

FILES = {
    "noi_khoa_gs_chau": _data_file("2010. Noi Khoa YHCT - GS Hoang Bao Chau. NXB Thoi Dai.docx"),
    "benh_ngu_quan": _data_file("*quan.docx"),
    "nhi_khoa": _data_file("nhi-khoa-y-hoc-co-truyen.docx"),
    "noi_khoa_general": _data_file("noi-khoa-y-hoc-co-truyen.docx"),
    "eval_json": _data_file("*quan.json"),
    "predictions": str(BASE_DIR / "crag_predictions.txt"),
}

# --- PARAMETERS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5
SIMILARITY_THRESHOLD = 0.3
HYBRID_SCORE_THRESHOLD = 0.5