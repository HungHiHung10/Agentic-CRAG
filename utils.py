# utils.py
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import EMBEDDING_MODEL_NAME, LLM_MODEL_ID

def load_embedding_model():
    print(f"[INIT] Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # model_kwargs={'device': 'cuda'},
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_llm_model():
    print(f"[INIT] Loading LLM Model: {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

# Hàm tạo Judge Model riêng (nếu cần config khác, còn không dùng chung LLM trên cũng được)
def load_judge_model():
    print(f"[INIT] Loading Judge Model...")
    # Tận dụng code load giống LLM chính nhưng max_new_tokens nhỏ hơn
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256, # Judge chỉ cần output ngắn
        do_sample=False,
        return_full_text=False
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))