import streamlit as st
from rag_pipeline import process_pdf, retrieve_docs, answer_query, llm_model

st.title("Vishi's AI Lawyer Chatbot")

# -------- Step 1: Upload PDF --------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# -------- Step 2: User Query --------
user_query = st.text_area(
    "Enter your prompt:",
    height=150,
    placeholder="Ask anything from the uploaded PDF..."
)

if st.button("Ask Vishi's Chatbot"):

    # Validate PDF
    if not uploaded_file:
        st.error("Please upload a PDF first.")
        st.stop()

    # Validate question
    if not user_query.strip():
        st.error("Please enter a valid question.")
        st.stop()

    # Show on chat window
    st.chat_message("user").write(user_query)

    # -------- RAG Pipeline --------
    try:
        # 1. Process PDF → Build FAISS DB
        faiss_db = process_pdf(uploaded_file)

        # 2. Retrieve relevant chunks
        docs = retrieve_docs(user_query, faiss_db)

        if not docs:
            st.warning("No relevant information found in the document.")
            st.stop()

        # 3. Generate answer
        response = answer_query(docs, llm_model, user_query)

        # 4. Show answer
        st.chat_message("assistant").write(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
