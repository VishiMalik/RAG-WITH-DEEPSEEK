from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load .env so GROQ_API_KEY is available
load_dotenv()

# Step1: Setup LLM (Use DeepSeek-distill with Groq)
llm_model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.2
)

# Step2: Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Step3: Answer Question
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know".
Don't create or assume info outside the context.

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

    # Extract only the answer text:
    return response.content
