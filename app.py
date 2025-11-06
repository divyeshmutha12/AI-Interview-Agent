# import os
# import json
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from docx import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings

# # Ensure base folder exists
# BASE_DIR = "knowledge_bases"
# os.makedirs(BASE_DIR, exist_ok=True)

# # -------------------- Helpers --------------------

# def create_kb(kb_name, role, uploaded_files):
#     kb_dir = os.path.join(BASE_DIR, kb_name.replace(" ", "_").lower())
#     os.makedirs(kb_dir, exist_ok=True)

#     docs_paths = []
#     for file in uploaded_files:
#         save_path = os.path.join(kb_dir, file.name)
#         with open(save_path, "wb") as f:
#             f.write(file.getbuffer())
#         docs_paths.append(save_path)

#     # Load & split docs
#     docs = []
#     for path in docs_paths:
#         loader = UnstructuredFileLoader(path)
#         docs.extend(loader.load())

#     splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)

#     # Embeddings + FAISS
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     vector_store.save_local(os.path.join(kb_dir, "faiss_index"))

#     # Metadata
#     metadata = {"name": kb_name, "role": role}
#     with open(os.path.join(kb_dir, "metadata.json"), "w") as f:
#         json.dump(metadata, f)

#     return True


# def search_kb(kb_name, query, k=3):
#     kb_dir = os.path.join(BASE_DIR, kb_name)
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.load_local(os.path.join(kb_dir, "faiss_index"),
#                                     embeddings,
#                                     allow_dangerous_deserialization=True)
#     return vector_store.similarity_search(query, k=k)


# # -------------------- UI --------------------

# st.title("📚 Knowledge Base Builder (Streamlit)")

# tab1, tab2 = st.tabs(["➕ Create KB", "🔍 Search KB"])

# # ---- Create KB ----
# with tab1:
#     st.header("Create a New Knowledge Base")
#     kb_name = st.text_input("Knowledge Base Name", placeholder="e.g., Java Backend KB")
#     role = st.text_input("Role Tag", placeholder="e.g., java_backend_interview")
#     uploaded_docs = st.file_uploader("Upload Documents", accept_multiple_files=True)

#     if st.button("Create Knowledge Base"):
#         if not kb_name or not role or not uploaded_docs:
#             st.warning("Please fill all fields and upload at least one document.")
#         else:
#             create_kb(kb_name, role, uploaded_docs)
#             st.success(f"✅ Knowledge Base '{kb_name}' created successfully!")

# # ---- Search KB ----
# with tab2:
#     st.header("Search Within a Knowledge Base")

#     kb_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

#     if len(kb_list) == 0:
#         st.info("No Knowledge Bases found. Create one in the first tab.")
#     else:
#         selected_kb = st.selectbox("Select Knowledge Base", kb_list)
#         query = st.text_input("Query Text")

#         if st.button("Search"):
#             if not query:
#                 st.warning("Enter text to search.")
#             else:
#                 results = search_kb(selected_kb, query)
#                 st.subheader("🔎 Results:")
#                 for r in results:
#                     st.write("📄 **Source**:", r.metadata.get("source", "Uploaded Document"))
#                     st.write(r.page_content)
#                     st.write("---")

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


# ---------- Search KB ----------
def search_kb(kb_name, query, k=3):
    kb_dir = os.path.join(BASE_DIR, kb_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.load_local(os.path.join(kb_dir, "faiss_index"),
                                    embeddings,
                                    allow_dangerous_deserialization=True)
    return vector_store.similarity_search(query, k=k)


# ---------- STREAMLIT UI ----------
st.title("📚 Knowledge Base Builder (Streamlit)")

tab1, tab2 = st.tabs(["➕ Create KB", "🔍 Search KB"])

# --- Create KB ---
with tab1:
    st.header("Create a New Knowledge Base")
    kb_name = st.text_input("Knowledge Base Name", placeholder="e.g., Java Backend KB")
    role = st.text_input("Role Tag", placeholder="e.g., java_backend_interview")
    uploaded_docs = st.file_uploader("Upload Documents", accept_multiple_files=True)

    if st.button("Create Knowledge Base"):
        if not kb_name or not role or not uploaded_docs:
            st.warning("Please fill all fields and upload at least one document.")
        else:
            create_kb(kb_name, role, uploaded_docs)
            st.success(f"✅ Knowledge Base '{kb_name}' created successfully!")


# --- Search KB ---
with tab2:
    st.header("Search Within a Knowledge Base")
    kb_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    if len(kb_list) == 0:
        st.info("No Knowledge Bases found. Create one in the first tab.")
    else:
        selected_kb = st.selectbox("Select Knowledge Base", kb_list)
        query = st.text_input("Search Query")

        if st.button("Search"):
            if not query:
                st.warning("Please enter search text.")
            else:
                results = search_kb(selected_kb, query)
                st.subheader("🔎 Results:")
                for r in results:
                    st.write("📄 **Source:**", r.metadata.get("source", "Unknown"))
                    st.write(r.page_content)
                    st.write("---")
