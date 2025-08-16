"""
Document loader module for the Graph RAG Document Analyzer.

This module handles PDF document loading and text chunking using LangChain.
"""

from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Handles document loading and chunking operations."""
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def load_and_chunk_document(self, file_path: str) -> List[Document]:
        """
        Load a PDF document and split it into chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If PDF loading fails
        """
        try:
            # Load PDF using LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata for tracking
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "source_file": file_path,
                    "chunk_size": len(chunk.page_content)
                })
            
            return chunks
            
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load PDF {file_path}: {str(e)}")
    
    def get_document_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the loaded document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with document statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars // len(chunks),
            "source_pages": len(set(chunk.metadata.get("page", 0) for chunk in chunks))
        }


def load_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Convenience function to load and chunk a document.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.load_and_chunk_document(file_path)