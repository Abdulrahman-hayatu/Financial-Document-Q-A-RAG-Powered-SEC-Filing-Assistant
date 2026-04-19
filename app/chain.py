import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from app.ingest import load_index
from dotenv import load_dotenv

load_dotenv()

# Prompt template
PROMPT_TEMPLATE = """You are a financial analyst assistant. Use ONLY the context
below to answer the question. If the answer is not in the context, say
"I could not find this information in the document." Do not speculate.

Context:
{context}

Question: {question}

Answer:"""


def build_chain():
    """
    Build and return a LangChain RetrievalQA chain.
    Call this once at app startup and reuse the returned chain object.
    """
    # Load FAISS index from disk
    vectorstore = load_index()

    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    # Instantiate the LLM via Groq Inference API
    llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.1,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY")
)

    # Create the prompt template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Assemble the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("[chain] RetrievalQA chain ready.")
    return chain


def ask(chain, question: str) -> dict:
    """
    Run a question through the retrieval chain.
    Returns a dict with 'answer' (str) and 'sources' (list of page numbers).
    """
    result = chain.invoke({"query": question})

    # Extract page numbers from source documents for citations
    sources = sorted(set(
        doc.metadata.get("page", "?") + 1
        for doc in result["source_documents"]
    ))

    return {
        "answer": result["result"].strip(),
        "sources": sources
    }


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chain = build_chain()
    response = ask(chain, "What was the total net revenue for the fiscal year?")
    print("\nAnswer:", response["answer"])
    print("Sources: Pages", response["sources"])