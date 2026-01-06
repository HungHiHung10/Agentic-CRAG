# modules/graph.py
from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from ddgs import DDGS
import numpy as np
from config import HYBRID_SCORE_THRESHOLD, SIMILARITY_THRESHOLD, RETRIEVAL_K
from modules.components import reranker_model

class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document] 

class CRAGWorkflow:
    def __init__(self, retriever, grader_chain, rewriter_chain, generator_chain):
        self.retriever = retriever
        self.grader = grader_chain
        self.rewriter = rewriter_chain
        self.generator = generator_chain

    def retrieve(self, state):
        print("\n[PROCESS] RETRIEVAL")
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs, "question": state["question"]}

    def grade_documents_hybrid(self, state):
        print("[PROCESS] HYBRID GRADING")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"

        if documents:
            for d in documents:
                llm_res = self.grader.invoke({"question": question, "document": d.page_content})
                llm_val = llm_res.get("binary_score", "no")
                
                # Calculate Score logic
                rerank_score = reranker_model.predict([(question, d.page_content)])[0]
                rerank_norm = 1 / (1 + np.exp(-rerank_score))
                llm_norm = 1.0 if llm_val == "yes" else 0.0
                final_score = (rerank_norm * 0.6) + (llm_norm * 0.4)
                
                if final_score >= HYBRID_SCORE_THRESHOLD:
                    filtered_docs.append(d)
        
        if not filtered_docs:
            web_search = "Yes"
            
        return {"documents": filtered_docs, "question": question, "web_search_needed": web_search}

    def rewrite_query(self, state):
        print("[PROCESS] REWRITE QUERY")
        new_q = self.rewriter.invoke({"question": state["question"]})
        return {"documents": state["documents"], "question": new_q}

    def web_search(state):
        """
        Web search using the new 'ddgs' library.
        """
        print("[PROCESS] WEB SEARCH (DuckDuckGo)")
        question = state["question"]

        documents = state.get("documents", [])
        if documents is None:
            documents = []

        web_results_content = ""

        try:
            with DDGS() as ddg:
                results = list(ddg.text(question, max_results=3))

                if results:
                    web_results_content = "\n\n".join(
                        [f"Title: {d.get('title', '')}\nLink: {d.get('href', '')}\nContent: {d.get('body', d.get('snippet', ''))}"
                        for d in results]
                    )
                else:
                    web_results_content = "[ERROR] Not Found"

        except Exception as e:
            print(f"[ERROR] {e}")
            web_results_content = f"[ERROR] {e}"

        web_doc = Document(
            page_content=web_results_content,
            metadata={"source": "duckduckgo_search"}
        )

        documents.append(web_doc)

        return {"documents": documents, "question": question}

    def generate_answer(self, state):
        print("[PROCESS] GENERATE")
        gen = self.generator.invoke({"context": state["documents"], "question": state["question"]})
        return {"generation": gen}

    def decide_to_generate(self, state):
        if state["web_search_needed"] == "Yes": return "rewrite_query"
        return "generate_answer"

    def build(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents_hybrid)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, 
                                       {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"})
        workflow.add_edge("rewrite_query", "web_search")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()