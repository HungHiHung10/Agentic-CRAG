# Agentic Corrective RAG System with LangGraph


## Table of Contents
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Model and Tool Initialization](#model-and-tool-initialization)
- [Running the CRAG System](#running-the-crag-system)
- [System Evaluation](#system-evaluation)

<p align="left">
  <strong>Quickstart here </strong> 
  <a href="https://colab.research.google.com/drive/1-Hh52dIAnHE3QWzUYlKtvuf7IR0zksHY?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

## Installation

To run this notebook, install the following Python libraries in your Google Colab environment:

```python
%capture
!pip install -U langchain langchain-community langchain-core langgraph langchain-openai langchain-huggingface langchain-chroma chromadb sentence-transformers duckduckgo-search ddgs
```

```python
%capture
!pip install langchain-huggingface sentence-transformers langchain-community docx2txt gdown
```

## Data Setup
The system uses Vietnamese Traditional Medicine (YHCT) data stored as `.docx` files on Google Drive.

1.  **Mount Google Drive:**

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Data Files**: Ensure the following `.docx` files are available:
    -   `2010. Noi Khoa YHCT - GS Hoang Bao Chau. NXB Thoi Dai.docx`
    -   `bệnh ngũ quan.docx`
    -   `nhi-khoa-y-hoc-co-truyen.docx`
    -   `noi-khoa-y-hoc-co-truyen.docx`

3.  **HuggingFace Embedding Models Initialization**:

    ```python
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch

    model_name = "BAAI/bge-m3"
    print(f"[PROCESS] Loading Embedding model: {model_name}...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("[SUCCESS] Loaded Embedding Model into GPU!")
    ```

4.  **Document Chunking**: Run the provided cells to load and split documents into chunks:
    -   Cell for `2010. Noi Khoa YHCT - GS Hoang Bao Chau. NXB Thoi Dai.docx`
    -   Cell for `bệnh ngũ quan.docx`
    -   Cell for `nhi-khoa-y-hoc-co-truyen.docx`
    -   Cell for `noi-khoa-y-hoc-co-truyen.docx`

5.  **Vector Database Creation**: After chunking the documents, a ChromaDB vector database is created and persisted on disk for efficient context retrieval.  
The cell below under **Create a Vector DB and persist on disk** embeds all chunks and stores the vector index for reuse across runs.

6.  **Retriever Configuration**: A Similarity with Threshold Retriever is configured to retrieve relevant documents from the vector database.

    ```python
    similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                            search_kwargs={"k": 3,
                                                                           "score_threshold": 0.3})
    ```

## Model and Tool Initialization

1. **Load the Large Language Model (LLM)**:  
   The **Qwen2.5-7B-Instruct** model is loaded to perform text generation, document grading, and query rewriting tasks.


2. **Initialize LLM Chains**:  
   Initialize the LLM chains (**Grader**, **Rewriter**, **Generator**) to handle different stages of the CRAG system.
   - **Grader**: Evaluates the relevance of retrieved documents.
   - **Rewriter**: Rewrites user queries to improve retrieval quality. 
   - **Generator**: Generates answers based on the given context and question.

3. **Load Web Search Tool**:
   Use `DuckDuckGoSearchResults` to retrieve information from the web when local context is insufficient.

## Running the CRAG System

1. **Define Graph State**:  
   Define the `GraphState` structure to track the internal state of the agent

2. **Define Graph Nodes**:  
   The functions `retrieve`, `grade_documents`, `rewrite_query`, `web_search`, `generate_answer`, and `decide_to_generate` are defined as nodes in the graph.

3. **Build the Agent Graph**:  
   The graph is constructed using `langgraph` to orchestrate the agent’s workflow.

4. **Execute the Agent**:  
   Run the agent with a sample query.

    ```python
    query = "How is liver cirrhosis treated in Vietnamese Traditional Medicine?"

    inputs = {
        "question": query,
        "documents": [],
        "generation": ""
    }

    response = agentic_rag.invoke(inputs)
    display(Markdown(response['generation']))
    ```

## System Evaluation

The system's performance is evaluated using metrics like Faithfulness, Relevance, and Correctness against a set of predefined questions and ground truth answers. The evaluation process involves:

1.  **Preparing Judge Model**: An LLM (`Qwen/Qwen2.5-7B-Instruct`) is initialized to act as an evaluator based on specific prompts for faithfulness, relevance, and correctness.

2.  **Loading Evaluation Data:** A JSON file (`bệnh ngũ quan.json`) containing questions and ground truths is loaded.

3.  **Running Evaluation:**
    *   **Agentic CRAG System**: The agentic CRAG system processes each question, and its answer is then judged by the LLM (`Qwen/Qwen2.5-7B-Instruct`) based on the defined metrics.
    *   **CRAG (without web search)**: An evaluation for a simplified CRAG system is performed.
    *   **Naive RAG**: A baseline Naive RAG system is also evaluated for comparison.

4.  **Results:** The evaluation scores for each system are compiled into DataFrames and saved to Excel files (`agentic_crag_results.xlsx`, `crag_result.xlsx`, `naive_rag_results.xlsx`).
