# config.py
import os

# --- MODEL CONFIG ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# --- PATHS ---
DATA_DIR = "./Agentic CRAG/Data/"
CHROMA_DB_DIR = "./YHCT"

FILES = {
    "noi_khoa_gs_chau": os.path.join(DATA_DIR, "2010. Noi Khoa YHCT - GS Hoang Bao Chau. NXB Thoi Dai.docx"),
    "benh_ngu_quan": os.path.join(DATA_DIR, "bệnh ngũ quan.docx"),
    "nhi_khoa": os.path.join(DATA_DIR, "nhi-khoa-y-hoc-co-truyen.docx"),
    "noi_khoa_general": os.path.join(DATA_DIR, "noi-khoa-y-hoc-co-truyen.docx"),
    "eval_json": os.path.join(DATA_DIR, "bệnh ngũ quan.json"),
    "predictions": "/content/drive/MyDrive/crag_predictions.txt"
}

# --- PARAMETERS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3
SIMILARITY_THRESHOLD = 0.3
HYBRID_SCORE_THRESHOLD = 0.5