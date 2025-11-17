# import os
# from pathlib import Path
# import json
# from rank_bm25 import BM25Okapi
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document




# # KB storage directory
# BASE_KB_DIR = Path("stored_kbs")
# BASE_KB_DIR.mkdir(exist_ok=True)

# IN_MEMORY_KB = {}  # {kb_name: {...}}

# def build_kb(kb_name: str, kb_docs: list[str]):
#     """
#     Build and SAVE a hybrid KB (FAISS + BM25) under stored_kbs/<kb_name>/
#     Only text is processed — original files are NOT saved.
#     """
#     if not kb_docs:
#         return

#     kb_path = BASE_KB_DIR / kb_name
#     kb_path.mkdir(exist_ok=True)

#     docs = [Document(page_content=text) for text in kb_docs]

#     splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)

#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     faiss = FAISS.from_documents(chunks, embeddings)

#     # ✅ Save FAISS index on disk
#     faiss.save_local(str(kb_path), index_name="faiss_index")

#     tokens = [c.page_content.lower().split() for c in chunks]
#     bm25 = BM25Okapi(tokens)

#     # ✅ Save BM25 tokens for reload
#     with open(kb_path / "bm25_tokens.json", "w") as f:
#         json.dump(tokens, f)

#     # ✅ Save chunk contents and metadata for reconstruction
#     chunks_data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
#     with open(kb_path / "chunks.json", "w", encoding="utf-8") as f:
#         json.dump(chunks_data, f, ensure_ascii=False, indent=2)

#     IN_MEMORY_KB[kb_name] = {"chunks": chunks, "faiss": faiss, "bm25": bm25, "tokens": tokens}


# def load_kb(kb_name: str):
#     """Load KB from stored folder if not in memory."""
#     if kb_name in IN_MEMORY_KB:
#         return

#     kb_path = BASE_KB_DIR / kb_name
#     if not kb_path.exists():
#         return

#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     faiss = FAISS.load_local(str(kb_path), embeddings, index_name="faiss_index")

#     with open(kb_path / "bm25_tokens.json") as f:
#         tokens = json.load(f)

#     bm25 = BM25Okapi(tokens)

#     IN_MEMORY_KB[kb_name] = {"faiss": faiss, "bm25": bm25, "tokens": tokens}


# def hybrid_retrieve(kb_name: str, query: str, top_k=5):
#     load_kb(kb_name)
#     kb = IN_MEMORY_KB.get(kb_name)
#     if not kb:
#         return []

#     faiss = kb["faiss"]
#     bm25 = kb["bm25"]

#     dense = faiss.similarity_search(query, k=top_k)

#     scores = bm25.get_scores(query.lower().split())
#     ranked = sorted(zip(dense, scores), key=lambda x: x[1], reverse=True)
#     sparse = [doc for doc, _ in ranked[:top_k]]


#     merged = []
#     for d in dense + sparse:
#         if d not in merged:
#             merged.append(d)

#     return merged[:top_k]


import os
from pathlib import Path
import json
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

# KB storage directory
BASE_KB_DIR = Path("stored_kbs")
BASE_KB_DIR.mkdir(exist_ok=True)

IN_MEMORY_KB = {}  # {kb_name: {...}}

def build_kb(kb_name: str, kb_docs: list[str]):
    """
    Build and SAVE a hybrid KB (FAISS + BM25) under stored_kbs/<kb_name>/
    Only text is processed — original files are NOT saved.
    """
    if not kb_docs:
        return

    kb_path = BASE_KB_DIR / kb_name
    kb_path.mkdir(exist_ok=True)

    docs = [Document(page_content=text) for text in kb_docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss = FAISS.from_documents(chunks, embeddings)

    # ✅ Save FAISS index on disk
    faiss.save_local(str(kb_path), index_name="faiss_index")

    # ✅ Prepare BM25 data
    chunk_texts = [chunk.page_content for chunk in chunks]
    tokens = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokens)

    # ✅ Save BM25 tokens for reload
    with open(kb_path / "bm25_tokens.json", "w") as f:
        json.dump(tokens, f)

    # ✅ Save chunk contents and metadata for reconstruction
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(kb_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    # ✅ Store in memory with ALL needed components
    IN_MEMORY_KB[kb_name] = {
        "chunks": chunks,
        "chunk_texts": chunk_texts,  # Store text for BM25 scoring
        "faiss": faiss,
        "bm25": bm25,
        "tokens": tokens
    }

    logger.info(f"✅ KB '{kb_name}' built with {len(chunks)} chunks")

def load_kb(kb_name: str):
    """Load KB from stored folder if not in memory."""
    if kb_name in IN_MEMORY_KB:
        return

    kb_path = BASE_KB_DIR / kb_name
    if not kb_path.exists():
        logger.warning(f"KB path not found: {kb_path}")
        return

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Load FAISS
        faiss_index_path = kb_path / "faiss_index"
        if faiss_index_path.exists():
            faiss = FAISS.load_local(
                str(kb_path),
                embeddings,
                index_name="faiss_index",
                allow_dangerous_deserialization=True
            )
        else:
            logger.error(f"FAISS index not found for KB: {kb_name}")
            return

        # Load chunks for reconstruction
        chunks_path = kb_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            chunks = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in chunks_data]
            chunk_texts = [chunk.page_content for chunk in chunks]
        else:
            logger.error(f"Chunks file not found for KB: {kb_name}")
            return

        # Load BM25 tokens
        bm25_path = kb_path / "bm25_tokens.json"
        if bm25_path.exists():
            with open(bm25_path) as f:
                tokens = json.load(f)
            bm25 = BM25Okapi(tokens)
        else:
            logger.warning(f"BM25 tokens not found for KB: {kb_name}, creating empty BM25")
            bm25 = BM25Okapi([[]])
            tokens = [[]]

        # Store in memory
        IN_MEMORY_KB[kb_name] = {
            "chunks": chunks,
            "chunk_texts": chunk_texts,
            "faiss": faiss,
            "bm25": bm25,
            "tokens": tokens
        }

        logger.info(f"✅ KB '{kb_name}' loaded with {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"❌ Error loading KB '{kb_name}': {e}")
        return

def hybrid_retrieve(kb_name: str, query: str, top_k: int = 5) -> list:
    """
    Perform hybrid retrieval using both FAISS and BM25.

    Returns proper Document objects that can be used by the agent.
    """
    load_kb(kb_name)
    kb = IN_MEMORY_KB.get(kb_name)

    if not kb:
        logger.error(f"KB '{kb_name}' not found in memory")
        return []

    try:
        # 1. FAISS semantic search
        faiss_results = kb["faiss"].similarity_search(query, k=top_k)
        logger.info(f"FAISS found {len(faiss_results)} results")

        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        if kb["bm25"] and query_tokens:
            bm25_scores = kb["bm25"].get_scores(query_tokens)
            # Get top BM25 results
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            bm25_results = [kb["chunks"][i] for i in top_indices if bm25_scores[i] > 0]
            logger.info(f"BM25 found {len(bm25_results)} results")
        else:
            bm25_results = []
            logger.warning("BM25 not available or empty query")

        # 3. Combine and deduplicate results
        all_results = []
        seen_content = set()

        # Add FAISS results first (semantic relevance)
        for doc in faiss_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_results.append(doc)

        # Add BM25 results (keyword relevance)
        for doc in bm25_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_results.append(doc)

        # Return top_k results
        final_results = all_results[:top_k]
        logger.info(f"✅ Hybrid retrieval returned {len(final_results)} results for query: '{query}'")

        return final_results

    except Exception as e:
        logger.error(f"❌ Error in hybrid_retrieve for KB '{kb_name}': {e}")
        return []

def keyword_search_only(kb_name: str, keywords: str, top_k: int = 5) -> list:
    """
    Perform ONLY keyword search using BM25.
    Returns Document objects for specific keyword matching.
    """
    load_kb(kb_name)
    kb = IN_MEMORY_KB.get(kb_name)

    if not kb:
        logger.error(f"KB '{kb_name}' not found in memory")
        return []

    try:
        query_tokens = keywords.lower().split()
        if not kb["bm25"] or not query_tokens:
            logger.warning("BM25 not available or empty keywords")
            return []

        # Get BM25 scores
        bm25_scores = kb["bm25"].get_scores(query_tokens)

        # Get top results with positive scores
        scored_docs = [(i, bm25_scores[i]) for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_indices = [i for i, score in scored_docs[:top_k]]
        results = [kb["chunks"][i] for i in top_indices]

        logger.info(f"✅ Keyword search found {len(results)} results for keywords: '{keywords}'")
        return results

    except Exception as e:
        logger.error(f"❌ Error in keyword_search_only for KB '{kb_name}': {e}")
        return []

def vector_search_only(kb_name: str, query: str, top_k: int = 5) -> list:
    """
    Perform ONLY vector search using FAISS.
    Returns Document objects for semantic similarity.
    """
    load_kb(kb_name)
    kb = IN_MEMORY_KB.get(kb_name)

    if not kb:
        logger.error(f"KB '{kb_name}' not found in memory")
        return []

    try:
        results = kb["faiss"].similarity_search(query, k=top_k)
        logger.info(f"✅ Vector search found {len(results)} results for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"❌ Error in vector_search_only for KB '{kb_name}': {e}")
        return []