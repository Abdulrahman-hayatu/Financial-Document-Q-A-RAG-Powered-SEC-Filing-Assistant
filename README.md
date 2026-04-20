# 📊 Financial Document Q&A — RAG-Powered SEC Filing Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Deployed](https://img.shields.io/badge/Deployed-Render-purple)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://financial-document-q-a-rag-powered-sec.onrender.com/docs)

A production-ready Retrieval-Augmented Generation (RAG) system
that enables natural language Q&A over SEC 10-K financial filings.
Pre-indexed on Apple Inc.'s 2023 Annual Report.

## Demo
![Demo GIF](assets/demo.gif)

**Live:** https://financial-document-q-a-rag-powered-sec.onrender.com/docs (free tier — ~60s warm-up)

## Architecture
```
User → Streamlit UI → FastAPI → LangChain RetrievalQA
                                        ↕
                               FAISS Vector Store
                                        ↕
                     HuggingFace Embeddings (MiniLM-L6-v2)
                                        ↕
                        LLM: llama-3.1-8b-instant (Groq)
```

## RAGAS Evaluation Results
| Metric | Score |
|---|---|
| Faithfulness | 0.97 |
| Answer Relevancy | 0.83 |
| Context Recall | 1.00 |
| Context Precision | 0.80 |

## Tech Stack
- **Orchestration:** LangChain 0.2
- **LLM:** llama-3.1-8b-instant via Groq Inference API
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Store:** FAISS (cosine similarity)
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Evaluation:** RAGAS
- **Deployment:** Render (free tier)

## Local Setup
```bash
git clone https://github.com/elhayatusman/Financial-Document-Q-A-RAG-Powered-SEC-Filing-Assistant.git
cd Financial-Document-Q-A-RAG-Powered-SEC-Filing-Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python -m app.ingest          # index a PDF
uvicorn app.api:app --port 8000  # start API
streamlit run frontend/ui.py    # start UI
```
