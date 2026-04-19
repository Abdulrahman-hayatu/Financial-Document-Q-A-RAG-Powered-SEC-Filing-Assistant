from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from app.chain import build_chain
from app.ingest import load_index
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Initialize the Groq LLM wrapper for RAGAS evaluation
groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,               # temperature=0 for deterministic judgements
    api_key=os.getenv("GROQ_API_KEY")
))
# Optional: set run_config for RAGAS to control retries and timeouts when calling the LLM
run_config={"max_retries": 2, "timeout": 60}

# Initialize the Hugging Face embeddings wrapper for RAGAS evaluation
ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
))

# Set the LLM and embeddings for each RAGAS metric
faithfulness.llm = groq_llm
context_recall.llm = groq_llm
context_precision.llm = groq_llm

# Reinitialise answer_relevancy with strictness=1 to force n=1 generation
# (Groq free tier rejects n>1 — strictness=1 maps directly to a single synthetic question)
answer_relevancy = AnswerRelevancy(
    llm=groq_llm,
    embeddings=ragas_embeddings,
    strictness=1
)

# Evaluation dataset - 5 questions with known answers based on the content of the sample PDF.
EVAL_DATASET = [
    {
        "question": "What was the total net revenue for fiscal year 2023?",
        "ground_truth": "Apple's total net revenue for fiscal year 2023 was $383.3 billion."
    },
    {
        "question": "What are the main risk factors related to competition?",
        "ground_truth": "The company faces intense competition in all markets, including from companies with greater resources, and product pricing pressures."
    },
    {
        "question": "How many full-time employees does the company have?",
        "ground_truth": "As of September 2023, Apple had approximately 161,000 full-time equivalent employees."
    },
    {
        "question": "What was the net income for fiscal year 2023?",
        "ground_truth": "Net income was $97.0 billion for fiscal year 2023."
    },
    {
        "question": "What geographic segments does the company report?",
        "ground_truth": "The company reports five geographic segments: Americas, Europe, Greater China, Japan, and Rest of Asia Pacific."
    },
]


def run_evaluation():
    chain = build_chain()
    vectorstore = load_index()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        question = item["question"]

        # Get retrieved context chunks
        retrieved_docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in retrieved_docs]

        # Get model answer
        result = chain.invoke({"query": question})
        answer = result["result"].strip()

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

        print(f"Q: {question}\nA: {answer}\n")
        time.sleep(3)    # pause between samples to stay under Groq rate limit

    # Create a Hugging Face Dataset for RAGAS evaluation
    eval_ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\n[eval] Running RAGAS evaluation...")
    from ragas.run_config import RunConfig

    results = evaluate(
        eval_ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        run_config=RunConfig(
            max_workers=1,       
            max_retries=2,
            timeout=120,
        )

    )

    print("\n═══ RAGAS Evaluation Results ═══")
    print(results)
    return results.to_pandas()


if __name__ == "__main__":
    df = run_evaluation()
    df.to_csv("eval_results.csv", index=False)
    print("[eval] Results saved to eval_results.csv")