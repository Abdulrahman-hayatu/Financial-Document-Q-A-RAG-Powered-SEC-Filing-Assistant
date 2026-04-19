import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000") # URL of the FastAPI backend

st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header and description
st.title("📊 Financial Document Q&A")
st.markdown("*Ask questions about SEC 10-K filings in natural language.*")
st.divider()

# Sidebar and document upload 
with st.sidebar:
    st.header("Upload Document")
    st.markdown("Upload a PDF 10-K filing to begin. The system will index it automatically.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="SEC 10-K filings work best. Download from sec.gov."
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary"):
            with st.spinner("Indexing document... this takes 30–90 seconds."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✓ Indexed {data['chunks']} chunks successfully.")
                    st.session_state["indexed"] = True
                else:
                    st.error(f"Error: {response.text}")

    st.divider()
    st.markdown("**Example questions:**")
    example_questions = [
        "What was the total net revenue??",
        "What are the main risk factors?",
        "How many employees does the company have?",
        "What products or segments does the company report?",
        "What was net income for the most recent fiscal year?"
    ]
    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_question"] = q

# Main — question and answer
prefill = st.session_state.get("prefill_question", "")
question = st.text_input(
    "Ask a question about the document",
    value=prefill,
    placeholder="e.g. What was Apple's revenue for fiscal year 2023?"
)

if st.button("Get Answer", type="primary", disabled=not question):
    st.session_state.pop("prefill_question", None)
    with st.spinner("Retrieving relevant passages and generating answer..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )

    if response.status_code == 200:
        data = response.json()
        st.markdown("### Answer")
        st.markdown(data["answer"])

        if data["sources"]:
            st.markdown(
                f"**Sources:** Pages {', '.join(str(p) for p in data['sources'])}"
            )

    elif response.status_code == 503:
        st.warning("No document indexed yet. Please upload a PDF using the sidebar.")
    else:
        st.error(f"API Error ({response.status_code}): {response.text}")