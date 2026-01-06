# evaluation.py
import pandas as pd
import json
import re
import os
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import FILES

# ==============================================================================
# 1. DEFINE JUDGE CHAINS
# ==============================================================================
def build_eval_chains(judge_llm):
    """
    Khởi tạo các Chain dùng để chấm điểm: Faithfulness, Relevance, Correctness.
    """
    
    # --- PROMPT 1: Faithfulness (Trung thực với ngữ cảnh) ---
    faithfulness_prompt = ChatPromptTemplate.from_template("""
    You are an evaluator for a RAG system.
    Task: Compare the AI ANSWER with the PROVIDED CONTEXT.

    Provided Context:
    {context}

    AI Answer:
    {answer}

    Assess whether the answer is fully supported by the provided context.
    - If the answer contains information NOT present in the context (hallucination) -> Low score.
    - If the answer is completely grounded in the context -> High score.

    Return the result in the following single format:
    Score: [Score from 1 to 5]
    Reason: [Concise explanation]
    """)

    # --- PROMPT 2: Relevance (Đúng trọng tâm câu hỏi) ---
    relevance_prompt = ChatPromptTemplate.from_template("""
    You are an evaluator for a RAG system.
    Task: Compare the AI ANSWER with the USER QUESTION.

    User Question:
    {question}

    AI Answer:
    {answer}

    Assess whether the answer correctly addresses the user's intent and question.
    Return the result in the following single format:
    Score: [Score from 1 to 5]
    Reason: [Concise explanation]
    """)

    # --- PROMPT 3: Correctness (Chính xác so với đáp án chuẩn) ---
    correctness_prompt = ChatPromptTemplate.from_template("""
    You are an examiner grading a Traditional Medicine exam.
    Compare the AI ANSWER with the GROUND TRUTH.

    Question: {question}

    GROUND TRUTH:
    {ground_truth}

    AI ANSWER:
    {generated_answer}

    Evaluation Criteria:
    - Compare the meaning, medicinal ingredients (herbs), and key symptoms.
    - Do not nitpick on wording/phrasing; focus only on medical knowledge accuracy.
    - If the AI captures the main points but lacks minor details -> Fair/Good score.
    - If the AI gets the core nature wrong or fabricates information -> Low score.

    Return the result in the following single format:
    Score: [Score from 1 to 5]
    Reason: [One-sentence explanation]
    """)

    # --- Build Chains ---
    return {
        "faithfulness": faithfulness_prompt | judge_llm | StrOutputParser(),
        "relevance": relevance_prompt | judge_llm | StrOutputParser(),
        "correctness": correctness_prompt | judge_llm | StrOutputParser()
    }

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def extract_score(text):
    """
    Trích xuất điểm số từ chuỗi kết quả của LLM (VD: 'Score: 4' -> 4)
    """
    match = re.search(r'Score\s*:\s*(\d+)', text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def load_eval_data(file_path):
    """
    Load dữ liệu đánh giá từ file JSON.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Evaluation file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Loaded {len(data)} evaluation samples.")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load JSON: {e}")
        return []

# ==============================================================================
# 3. MAIN EVALUATION LOOP
# ==============================================================================
def run_evaluation(app_graph, eval_chains, output_file="eval_results.xlsx"):
    """
    Chạy hệ thống RAG trên tập dữ liệu test và chấm điểm.
    
    Args:
        app_graph: LangGraph app đã compile.
        eval_chains: Dict chứa các chains chấm điểm (từ build_eval_chains).
        output_file: Tên file Excel đầu ra.
    """
    eval_data_path = FILES["eval_json"]
    eval_data = load_eval_data(eval_data_path)
    
    if not eval_data:
        return

    results = []
    print(f"\n[PROCESS] Starting Evaluation on {len(eval_data)} samples...")

    for item in tqdm(eval_data):
        question = item['question']
        ground_truth = item.get('ground_truth_answer', '')

        # 1. Run RAG System
        try:
            inputs = {"question": question, "documents": [], "generation": ""}
            response = app_graph.invoke(inputs)
            
            generated_answer = response.get("generation", "")
            retrieved_docs = response.get("documents", [])
            
            # Format context text for faithfulness check
            # Handle case where documents might be list of Docs or Strings
            if retrieved_docs and hasattr(retrieved_docs[0], 'page_content'):
                context_text = "\n\n".join([d.page_content for d in retrieved_docs])
            else:
                context_text = "\n\n".join(str(d) for d in retrieved_docs)

        except Exception as e:
            print(f"[ERROR] RAG Execution failed for q='{question}': {e}")
            continue

        # 2. Run Evaluation (Judge)
        try:
            # A. Faithfulness
            faith_res = eval_chains["faithfulness"].invoke({
                "context": context_text, 
                "answer": generated_answer
            })
            
            # B. Relevance
            rel_res = eval_chains["relevance"].invoke({
                "question": question, 
                "answer": generated_answer
            })
            
            # C. Correctness
            corr_res = eval_chains["correctness"].invoke({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer
            })

            # 3. Store Results
            results.append({
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                "context_retrieved": context_text[:500] + "...", # Truncate for Excel
                "score_faithfulness": extract_score(faith_res),
                "score_relevance": extract_score(rel_res),
                "score_correctness": extract_score(corr_res),
                "reason_correctness": corr_res
            })

        except Exception as e:
            print(f"[ERROR] Scoring failed for q='{question}': {e}")

    # 4. Save and Report
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*40)
        print("EVALUATION REPORT")
        print("="*40)
        print(f"Faithfulness: {df_results['score_faithfulness'].mean():.2f} / 5")
        print(f"Relevance:    {df_results['score_relevance'].mean():.2f} / 5")
        print(f"Correctness:  {df_results['score_correctness'].mean():.2f} / 5")
        print("="*40)

        df_results.to_excel(output_file, index=False)
        print(f"[SUCCESS] Detailed report saved to: {output_file}")
    else:
        print("[WARNING] No results to save.")