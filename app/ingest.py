import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # loads HUGGINGFACEHUB_API_TOKEN from .env

# Configuration constants
PDF_PATH        = "data/sample_10k.pdf"
FAISS_INDEX_DIR = "faiss_index"
CHUNK_SIZE      = 800   # characters per chunk (not tokens — close enough for now)
CHUNK_OVERLAP   = 50    # overlap between consecutive chunks
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and split it into overlapping text chunks.

    Returns a list of LangChain Document objects, each containing:
      - page_content: the text of the chunk
      - metadata: source file, page number
    """
    # PyMuPDFLoader preserves page numbers in metadata automatically
    loader = PyMuPDFLoader(pdf_path)
    raw_docs = loader.load()
    print(f"[ingest] Loaded {len(raw_docs)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # RecursiveCharacter tries to split on paragraphs first,
        # then sentences, then words — preserving semantic boundaries
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ingest] Split into {len(chunks)} chunks")
    return chunks


def build_and_save_index(chunks: list) -> FAISS:
    """
    Convert text chunks to embeddings and store in a FAISS index.
    Saves the index to disk so it can be reloaded without re-embedding.
    """
    print("[ingest] Loading embedding model (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},    # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
        # normalize=True makes dot product equivalent to cosine similarity
    )

    print("[ingest] Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"[ingest] Index saved to ./{FAISS_INDEX_DIR}/")
    return vectorstore


def load_index() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # required for FAISS pkl files
    )
    print("[ingest] FAISS index loaded from disk")
    return vectorstore


if __name__ == "__main__":
    chunks = load_and_chunk_pdf(PDF_PATH)
    build_and_save_index(chunks)
    print("[ingest] Done. Run `python -m app.chain` to test retrieval.")