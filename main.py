# main.py
from config import SIMILARITY_THRESHOLD, RETRIEVAL_K
from utils import load_embedding_model, load_llm_model
from modules.ingestion import create_vector_db
from modules.components import build_grader_chain, build_rewriter_chain, build_generator_chain
from modules.graph import CRAGWorkflow

def main():
    # 1. Load Models
    embed_model = load_embedding_model()
    llm = load_llm_model()
    
    # 2. Setup Vector DB
    db = create_vector_db(embed_model, force_rebuild=False)
    retriever = db.as_retriever(search_type="similarity_score_threshold", 
                                search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD})
    
    # 3. Setup Chains
    grader = build_grader_chain(llm)
    rewriter = build_rewriter_chain(llm)
    generator = build_generator_chain(llm)
    
    # 4. Build Graph
    crag_system = CRAGWorkflow(retriever, grader, rewriter, generator)
    app = crag_system.build()
    
    # 5. Run Inference
    query = "Triệu chứng và cách điều trị bệnh Trúng thử?"
    print(f"\nQuestion: {query}")
    inputs = {"question": query, "documents": [], "generation": ""}
    
    result = app.invoke(inputs)
    print(f"\nAnswer:\n{result['generation']}")

if __name__ == "__main__":
    main()