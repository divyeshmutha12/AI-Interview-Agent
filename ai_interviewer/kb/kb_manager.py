import os
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = "knowledge_bases"
os.makedirs(BASE_DIR, exist_ok=True)


# ---------- Document Loader ----------
def load_document(path):
    ext = path.lower().split(".")[-1]

    # PDF
    if ext == "pdf":
        loader = PyPDFLoader(path)
        return loader.load()

    # DOCX
    elif ext in ["docx", "doc"]:
        doc = Document(path)
        texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return [{"page_content": text, "metadata": {"source": path}} for text in texts]

    # TXT / Others
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{"page_content": text, "metadata": {"source": path}}]


# ---------- Create KB ----------
def create_kb(kb_name, role, uploaded_files):
    kb_dir = os.path.join(BASE_DIR, kb_name.replace(" ", "_").lower())
    os.makedirs(kb_dir, exist_ok=True)

    # Save docs locally
    saved_paths = []
    for file in uploaded_files:
        save_path = os.path.join(kb_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(save_path)

    # Load & combine all docs
    docs = []
    for path in saved_paths:
        docs.extend(load_document(path))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings + FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(os.path.join(kb_dir, "faiss_index"))

    # Save metadata
    with open(os.path.join(kb_dir, "metadata.json"), "w") as f:
        json.dump({"name": kb_name, "role": role}, f)

    return True


# # ---------- Search KB ----------
# def search_kb(kb_name, query, k=3):
#     kb_dir = os.path.join(BASE_DIR, kb_name)
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     vector_store = FAISS.load_local(os.path.join(kb_dir, "faiss_index"),
#                                     embeddings,
#                                     allow_dangerous_deserialization=True)
#     return vector_store.similarity_search(query, k=k)