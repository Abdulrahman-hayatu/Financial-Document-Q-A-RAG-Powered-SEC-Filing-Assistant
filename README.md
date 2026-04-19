# Financial-Document-Q-A-RAG-Powered-SEC-Filing-Assistant
# 📊 Financial Document Q&A — RAG-Powered SEC Filing Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Deployed](https://img.shields.io/badge/Deployed-Render-purple)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://your-app.onrender.com)

A production-ready Retrieval-Augmented Generation (RAG) system
that enables natural language Q&A over SEC 10-K financial filings.
Pre-indexed on Apple Inc.'s 2023 Annual Report.

## Demo
![Demo GIF](assets/demo.gif)

**Live:** https://financial-rag-ui.onrender.com (free tier — ~60s warm-up)
**API docs:** https://financial-rag-api.onrender.com/docs

## Architecture
```
User → Streamlit UI → FastAPI → LangChain RetrievalQA
                                        ↕
                               FAISS Vector Store
                                        ↕
                     HuggingFace Embeddings (MiniLM-L6-v2)
                                        ↕
                        LLM: Mistral-7B-Instruct (HuggingFace)
```

## RAGAS Evaluation Results
| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.82 |
| Context Recall | 0.79 |
| Context Precision | 0.84 |

## Tech Stack
- **Orchestration:** LangChain 0.2
- **LLM:** Mistral-7B-Instruct via HuggingFace Inference API
- **Embeddings:** all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Store:** FAISS (cosine similarity)
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **Evaluation:** RAGAS
- **Deployment:** Render (free tier)

## Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/financial-rag.git
cd financial-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "HUGGINGFACEHUB_API_TOKEN=your_token_here" > .env
python -m app.ingest          # index a PDF
uvicorn app.api:app --port 8000  # start API
streamlit run frontend/ui.py    # start UI
```