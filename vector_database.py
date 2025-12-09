import os
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import GroqEmbeddings

# Directory where pdfs are saved
PDF_DIR = "pdfs/"

# ---------------- Step 1: Save uploaded PDF ----------------
def save_pdf(uploaded_file):
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ---------------- Step 2: Load PDFs ----------------
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# ---------------- Step 3: Split into chunks ----------------
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# ---------------- Step 4: Embeddings (Groq safe) ----------------
def get_embedding_model():
    return GroqEmbeddings(model="text-embedding-3-large")

# ---------------- Step 5: Build Vectorstore (FAISS) ----------------
def build_faiss_db(chunks):
    embeddings = get_embedding_model()
    db = FAISS.from_documents(chunks, embeddings)
    return db
