# Intelligent Document Q&A Chatbot

An end-to-end Generative AI project that allows users to upload PDF documents, ask questions in natural language, and receive answers grounded in the uploaded documents using Retrieval-Augmented Generation (RAG).

## 1. Project Overview

The Intelligent Document Q&A Chatbot is designed to simplify information retrieval from PDF documents.

Instead of manually searching through lengthy documents, users can upload PDFs and ask questions. The system retrieves relevant document passages and uses a Large Language Model (LLM) to generate context-based answers.

The application also displays relevant document sources and page references to improve answer transparency.

## 2. Problem Statement

Extracting specific information from lengthy PDF documents can be time-consuming.

This project addresses that problem by developing a chatbot that:
- Accepts multiple PDF documents.
- Extracts and processes document content.
- Retrieves relevant information based on user questions.
- Generates answers using retrieved document context.
- Provides source references.
- Avoids generating answers when sufficient supporting information is unavailable.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| CrossEncoder | Reranking retrieved chunks |
| Transformers | Loading language models |
| Qwen | Answer generation and evaluation |
| Streamlit | Web application interface |
| Pandas | Evaluation data processing |
| NumPy | Numerical operations |

## 4. RAG Architecture

The application follows this workflow:

1. Upload PDF documents.
2. Extract text and retain page metadata.
3. Clean and preprocess extracted text.
4. Split documents into overlapping chunks.
5. Generate embeddings for document chunks.
6. Store and search embeddings using FAISS.
7. Accept a user question.
8. Rewrite the query when necessary.
9. Perform hybrid retrieval using vector search and BM25.
10. Combine retrieval results using Reciprocal Rank Fusion (RRF).
11. Rerank candidate chunks using CrossEncoder.
12. Apply relevance filtering.
13. Pass relevant context to the LLM.
14. Generate an answer grounded in retrieved context.
15. Display the answer with source references.

## 5. Implemented Features

- Multiple PDF upload and text extraction.
- Text cleaning and recursive chunking.
- Embedding generation.
- FAISS vector retrieval.
- BM25 keyword retrieval.
- Hybrid search using Reciprocal Rank Fusion.
- Metadata filtering.
- Query rewriting.
- CrossEncoder reranking.
- Relevance threshold filtering.
- Context-based answer generation.
- Not-found response when sufficient evidence is unavailable.
- Document and page references.
- Chat history in Streamlit.
- Retrieval and answer evaluation.
- Similarity scores and source information.

## 6. Project Structure

```text
document_qa_chatbot/
│
├── data/
│   └── evaluation/
│       ├── evaluation_results.csv
│       ├── judge_results.csv
│       ├── final_evaluation.csv
│       ├── evaluation_report.md
│       └── score_distribution.png
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── inspect_results.py
│   └── generate_report.py
│
├── src/
│   └── document_qa_chatbot/
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── answer_evaluation.py
│       │   └── evaluate.py
│       │
│       ├── __init__.py
│       ├── app.py
│       ├── ingestion.py
│       ├── retrieval.py
│       ├── generation.py
│       └── evaluation_controller.py
│
├── data/                  # Input PDF documents
├── requirements.txt
├── README.md
└── recall_evaluation.ipynb
```

## 7. Installation and Setup

### Prerequisites

- Python 3.12
- uv package manager
- Internet connection for downloading pretrained models

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd document_qa_chatbot
```

### Step 2: Install dependencies

```bash
uv pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run streamlit run src/document_qa_chatbot/app.py
```

The Streamlit application will provide a local URL in the terminal.

## 8. Evaluation

The system was evaluated using 20 predefined questions covering the contents of the test PDF documents.

### 8.1 Retrieval Evaluation

Recall@K measures whether relevant document chunks are retrieved within the top K results.

| Metric | Result |
|---|---:|
| Recall@1 | 26.83% |
| Recall@3 | 49.75% |
| Recall@5 | 54.75% |
| Recall@10 | 61.58% |

### 8.2 Answer Evaluation

An LLM-as-a-Judge approach using Qwen was used to evaluate generated answers.

Two evaluation dimensions were considered:

- Correctness: Whether the generated answer correctly addresses the question.
- Faithfulness: Whether the generated answer is supported by the retrieved document context.

| Evaluation Metric | Average Score |
|---|---:|
| Correctness | 3.20 / 5 |
| Faithfulness | 3.45 / 5 |

Total evaluated questions: 20.

Detailed evaluation results are available in `data/evaluation/`.

## 9. Limitations

- Answer quality depends on the relevance and completeness of retrieved document chunks.
- Query rewriting can occasionally change the intended meaning of technical terms.
- Retrieval may fail to identify some relevant passages.
- LLM-generated evaluation scores are estimates and may contain evaluator bias.
- PyPDF extracts text but does not perform OCR on scanned PDF documents.
- Model inference requires sufficient system memory and computational resources.

## 10. Future Improvements

- Improve retrieval quality and recall.
- Improve query rewriting for domain-specific terminology.
- Add support for scanned PDFs through OCR.
- Introduce human evaluation alongside automated evaluation.
- Improve answer generation and faithfulness.
- Optimize model inference performance.

## 11. Conclusion

The Intelligent Document Q&A Chatbot demonstrates an end-to-end Retrieval-Augmented Generation system with advanced retrieval techniques, LLM-based answer generation, source references, and evaluation.

The project combines semantic search, keyword retrieval, reranking, and language models to provide a practical solution for question answering over PDF documents.