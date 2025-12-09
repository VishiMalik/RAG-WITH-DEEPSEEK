from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step 1: Setup LLM using Groq
llm_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",   # <-- FIXED
    temperature=0.2
)

# Step 2: Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Step 3: Prompt Template
custom_prompt_template = """
Use ONLY the context to answer the user's question.
If the answer is not in the context, say "I don't know".

Question: {question}

Context:
{context}

Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    response = chain.invoke({"question": query, "context": context})

    # Return the clean text from the model response
    return response.content
