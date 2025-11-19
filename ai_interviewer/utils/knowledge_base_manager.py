"""
Knowledge Base Manager with FAISS Vector Store

This module manages a vector database of company-specific documents for intelligent
question generation. Documents are stored in FAISS and searched semantically.

Usage:
    # Initialize KB Manager
    kb_manager = KnowledgeBaseManager()

    # Add documents to KB
    kb_manager.add_documents_from_directory("kb_documents/ai_ml/")

    # Search KB
    results = kb_manager.search("RAG implementation best practices", k=5)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Document parsing
from .document_parser import DocumentParser

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages FAISS-based knowledge base for company documents.

    Features:
    - Add documents (PDF, DOCX, TXT) to knowledge base
    - Semantic search over documents
    - Organize documents by domain/topic
    - Persistent storage of vector index
    """

    def __init__(
        self,
        kb_dir: str = "knowledge_base",
        openai_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Knowledge Base Manager.

        Args:
            kb_dir: Directory to store FAISS index and metadata
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            embedding_model: OpenAI embedding model (uses EMBEDDING_MODEL env var if not provided, defaults to text-embedding-3-small)
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(exist_ok=True)

        self.index_path = self.kb_dir / "faiss_index"
        self.metadata_path = self.kb_dir / "metadata.pkl"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Get embedding model from parameter, env var, or default (best practice!)
        if embedding_model is None:
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # Get OpenAI API key from parameter or env var
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")

        # Initialize OpenAI embeddings
        logger.info(f"Initializing OpenAI embeddings: {embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
        logger.info("OpenAI embeddings initialized successfully")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize document parser
        self.doc_parser = DocumentParser()

        # Load or create vector store
        self.vector_store: Optional[FAISS] = None
        self.metadata: Dict[str, Any] = {}

        self._load_vector_store()

    def _load_vector_store(self):
        """Load existing FAISS index or create new one"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

                doc_count = self.metadata.get('document_count', 0)
                logger.info(f"FAISS index loaded: {doc_count} documents indexed")
            else:
                logger.info("No existing index found. Will create new index when documents are added.")
                self.metadata = {
                    'document_count': 0,
                    'domains': {},
                    'files_indexed': []
                }
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}. Creating new index.")
            self.vector_store = None
            self.metadata = {
                'document_count': 0,
                'domains': {},
                'files_indexed': []
            }

    def _save_vector_store(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.vector_store is not None:
                logger.info("Saving FAISS index...")
                self.vector_store.save_local(str(self.index_path))

                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)

                logger.info(f"FAISS index saved: {self.metadata['document_count']} documents")
            else:
                logger.warning("No vector store to save")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def add_document(
        self,
        file_path: str,
        domain: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a single document to the knowledge base.

        Args:
            file_path: Path to document file (PDF, DOCX, or TXT)
            domain: Domain/category (e.g., "ai_ml", "devops", "backend")
            metadata: Optional metadata to attach to document chunks

        Returns:
            Number of chunks added
        """
        try:
            file_path = Path(file_path)

            # Check if already indexed
            if str(file_path) in self.metadata.get('files_indexed', []):
                logger.warning(f"File already indexed: {file_path.name}")
                return 0

            logger.info(f"Adding document: {file_path.name} (domain: {domain})")

            # Parse document
            text = self.doc_parser.parse_file(str(file_path))

            if not text or len(text.strip()) < 10:
                logger.warning(f"Document is empty or too short: {file_path.name}")
                return 0

            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Document split into {len(chunks)} chunks")

            # Create Document objects with metadata
            doc_metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'domain': domain,
                'chunk_count': len(chunks)
            }

            if metadata:
                doc_metadata.update(metadata)

            documents = [
                Document(
                    page_content=chunk,
                    metadata={**doc_metadata, 'chunk_id': i}
                )
                for i, chunk in enumerate(chunks)
            ]

            # Add to vector store
            if self.vector_store is None:
                # Create new vector store
                logger.info("Creating new FAISS index...")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)

            # Update metadata
            self.metadata['document_count'] += 1
            self.metadata['files_indexed'].append(str(file_path))

            if domain not in self.metadata['domains']:
                self.metadata['domains'][domain] = []
            self.metadata['domains'][domain].append(str(file_path))

            # Save to disk
            self._save_vector_store()

            logger.info(f"Successfully added {len(chunks)} chunks from {file_path.name}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            raise

    def add_documents_from_directory(
        self,
        directory: str,
        domain: str = "general",
        recursive: bool = True
    ) -> int:
        """
        Add all documents from a directory to the knowledge base.

        Args:
            directory: Path to directory containing documents
            domain: Domain/category for all documents in this directory
            recursive: Whether to search subdirectories

        Returns:
            Total number of chunks added
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.info(f"Adding documents from directory: {directory}")

        # Supported extensions
        extensions = self.doc_parser.get_supported_formats()

        # Find all supported files
        files = []
        if recursive:
            for ext in extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(directory.glob(f"*{ext}"))

        logger.info(f"Found {len(files)} documents to index")

        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.add_document(file_path, domain=domain)
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Failed to add {file_path.name}: {e}")
                continue

        logger.info(f"Completed: {total_chunks} chunks added from {len(files)} files")
        return total_chunks

    def search(
        self,
        query: str,
        k: int = 5,
        domain: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            domain: Optional domain filter (e.g., "ai_ml")
            score_threshold: Optional minimum similarity score (0-1)

        Returns:
            List of search results with content and metadata
        """
        if self.vector_store is None:
            logger.warning("No documents in knowledge base. Returning empty results.")
            return []

        try:
            logger.info(f"Searching KB: '{query[:50]}...' (k={k}, domain={domain})")

            # Perform similarity search with scores
            if domain:
                # Filter by domain using metadata
                filter_dict = {'domain': domain}
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k * 2,  # Get more results for filtering
                    filter=filter_dict
                )
                # Take top k after filtering
                results = results[:k]
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)

            # Format results
            formatted_results = []
            for doc, score in results:
                # Convert distance to similarity (FAISS returns L2 distance)
                # Lower distance = higher similarity
                similarity = 1 / (1 + score)

                # Apply score threshold if specified
                if score_threshold and similarity < score_threshold:
                    continue

                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': round(similarity, 4),
                    'distance': round(score, 4)
                })

            logger.info(f"Found {len(formatted_results)} relevant chunks")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': self.metadata.get('document_count', 0),
            'total_chunks': self.vector_store.index.ntotal if self.vector_store else 0,
            'domains': list(self.metadata.get('domains', {}).keys()),
            'files_indexed': len(self.metadata.get('files_indexed', [])),
            'index_path': str(self.index_path),
            'embedding_model': self.embeddings.model
        }

    def clear_index(self):
        """Clear the entire knowledge base (use with caution!)"""
        logger.warning("Clearing knowledge base...")
        self.vector_store = None
        self.metadata = {
            'document_count': 0,
            'domains': {},
            'files_indexed': []
        }

        # Delete files
        if self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path, ignore_errors=True)

        if self.metadata_path.exists():
            self.metadata_path.unlink()

        logger.info("Knowledge base cleared")

    def remove_document(self, file_path: str) -> bool:
        """
        Remove a document from the knowledge base.

        Note: FAISS doesn't support deletion, so this rebuilds the index.
        """
        file_path = str(Path(file_path))

        if file_path not in self.metadata.get('files_indexed', []):
            logger.warning(f"File not in index: {file_path}")
            return False

        logger.info(f"Removing document: {file_path}")
        logger.warning("This requires rebuilding the entire index...")

        # This is expensive - need to rebuild index without this file
        # For now, just log and return
        logger.error("Document removal not yet implemented. Use clear_index() and re-add documents.")
        return False


# Convenience function for quick search
def search_knowledge_base(query: str, k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Quick utility to search the knowledge base.

    Args:
        query: Search query
        k: Number of results
        domain: Optional domain filter

    Returns:
        Search results
    """
    kb = KnowledgeBaseManager()
    return kb.search(query, k=k, domain=domain)
