from rag_pipeline import answer_query, retrieve_docs, llm_model

import streamlit as st
# Step 1: Setup upload PDF functionality
uploaded_file=st.file_uploader("Upload PDF", type='PDF', accept_multiple_files=False)

  
# Step 2: Chatbot Skeleton (Question and Answer)

user_query=st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")
ask_question=st.button("Ask from Vishi's Chatbot")
if ask_question:

  st.chat_message("user").write(user_query)

  # RAG Pipeline
  retrieved_docs=retrieve_docs(user_query)
  response=answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
  st.chat_message("AI Lawyer").write(response)
else:
  st.error("Kindly upload a valid file first.")
