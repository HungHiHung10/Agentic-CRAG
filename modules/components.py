# modules/components.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from sentence_transformers import CrossEncoder
from operator import itemgetter
from config import RERANKER_MODEL_NAME

# --- RERANKER ---
print("[INIT] Loading Re-ranker...")
# reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device='cuda')
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device='cpu')

# --- CHAINS ---
def build_grader_chain(llm):
    def parse_grader_output(text: str):
        return {"binary_score": "yes" if "yes" in text.lower() else "no"}

    SYS_PROMPT = """You are a grader assessing relevance... (Copy full prompt)"""
    prompt = ChatPromptTemplate.from_messages([("system", SYS_PROMPT), ("human", "Doc:\n{document}\n\nQuestion: {question}")])
    return prompt | llm | StrOutputParser() | parse_grader_output

def build_rewriter_chain(llm):
    SYS_PROMPT = """Act as a question re-writer... (Copy full prompt)"""
    prompt = ChatPromptTemplate.from_messages([("system", SYS_PROMPT), ("human", "Question:\n{question}\n\nImproved question:")])
    return prompt | llm | StrOutputParser()

def build_generator_chain(llm):
    prompt_text = """You are an assistant... (Copy full prompt) ... Context: {context} Question: {question} Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": (itemgetter('context') | RunnableLambda(format_docs)), "question": itemgetter('question')}
        | prompt | llm | StrOutputParser()
    )

# --- TOOLS ---
def build_web_search_tool():
    ddg_search = DuckDuckGoSearchResults(backend="api", num_results=3)
    return Tool(name="web_search", func=ddg_search.invoke, description="Search internet")