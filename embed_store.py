"""
embed_store.py - Text Embedding and FAISS Vector Store

This module handles:
1. Converting text chunks into vector embeddings
2. Creating a FAISS index for fast similarity search
3. Storing embeddings for later retrieval

Dependencies:
- sentence-transformers: Free library for generating embeddings
- faiss-cpu: Facebook AI Similarity Search (CPU version, free)
"""

import os
import pickle
from typing import List, Tuple, Dict
import numpy as np


def create_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convert text chunks into vector embeddings using sentence-transformers.
    
    Embeddings are numerical representations of text that capture semantic meaning.
    Similar texts will have similar embeddings (close in vector space).
    
    Args:
        chunks: List of text chunks to embed
        model_name: Name of the sentence-transformer model (default: all-MiniLM-L6-v2)
                    This is a small, fast model that works well for general text.
        
    Returns:
        NumPy array of shape (num_chunks, embedding_dimension)
        Each row is an embedding vector for one chunk
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {model_name}")
        print("(First time may download the model, ~80MB)")
        
        # Load the pre-trained model
        model = SentenceTransformer(model_name)
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings for all chunks at once
        # This returns a numpy array where each row is an embedding
        embeddings = model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        print(f"(Each chunk is represented by a {embeddings.shape[1]}-dimensional vector)")
        
        return embeddings
        
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")


def create_faiss_index(embeddings: np.ndarray) -> 'faiss.Index':
    """
    Create a FAISS index for fast similarity search.
    
    FAISS (Facebook AI Similarity Search) allows us to quickly find
    the most similar vectors to a query vector using cosine similarity.
    
    Args:
        embeddings: NumPy array of embeddings (num_chunks, embedding_dim)
        
    Returns:
        FAISS index object ready for similarity search
    """
    try:
        import faiss
        
        # Get the dimension of embeddings
        dimension = embeddings.shape[1]
        
        print(f"Creating FAISS index with dimension: {dimension}")
        
        # Create a FAISS index using L2 (Euclidean) distance
        # We'll normalize vectors to use cosine similarity effectively
        index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity
        # Cosine similarity = dot product of normalized vectors
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to the index
        index.add(embeddings)
        
        print(f"FAISS index created with {index.ntotal} vectors")
        
        return index
        
    except ImportError:
        raise ImportError(
            "faiss-cpu is required. Install with: pip install faiss-cpu"
        )
    except Exception as e:
        raise Exception(f"Error creating FAISS index: {str(e)}")


def save_index_and_chunks(
    index: 'faiss.Index',
    chunks: List[str],
    metadata: List[Dict[str, str]] = None,
    index_path: str = "faiss_index.bin",
    chunks_path: str = "chunks.pkl",
    metadata_path: str = "metadata.pkl"
) -> None:
    """
    Save FAISS index, chunks, and metadata to disk for later use.
    
    This allows us to avoid re-processing documents every time.
    
    Args:
        index: FAISS index object
        chunks: List of text chunks
        metadata: List of metadata dictionaries (one per chunk)
        index_path: Where to save the FAISS index
        chunks_path: Where to save the chunks list
        metadata_path: Where to save the metadata list
    """
    try:
        import faiss
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        print(f"Saved FAISS index to: {index_path}")
        
        # Save chunks using pickle
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"Saved chunks to: {chunks_path}")
        
        # Save metadata if provided
        if metadata is not None:
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"Saved metadata to: {metadata_path}")
        
    except Exception as e:
        print(f"Warning: Could not save index/chunks/metadata: {str(e)}")


def load_index_and_chunks(
    index_path: str = "faiss_index.bin",
    chunks_path: str = "chunks.pkl",
    metadata_path: str = "metadata.pkl"
) -> Tuple['faiss.Index', List[str], List[Dict[str, str]]]:
    """
    Load previously saved FAISS index, chunks, and metadata from disk.
    
    Args:
        index_path: Path to saved FAISS index
        chunks_path: Path to saved chunks list
        metadata_path: Path to saved metadata list
        
    Returns:
        Tuple of (FAISS index, list of chunks, list of metadata)
        If metadata file doesn't exist, returns empty list for metadata
    """
    try:
        import faiss
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from: {index_path}")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from: {chunks_path}")
        
        # Load metadata if it exists
        metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"Loaded metadata for {len(metadata)} chunks from: {metadata_path}")
        else:
            # Create empty metadata if file doesn't exist (backward compatibility)
            metadata = [{"source": "unknown"} for _ in chunks]
            print(f"Warning: Metadata file not found. Using default metadata.")
        
        return index, chunks, metadata
        
    except Exception as e:
        raise Exception(f"Error loading index/chunks/metadata: {str(e)}")


def build_vector_store(
    chunks: List[str],
    metadata: List[Dict[str, str]] = None
) -> Tuple['faiss.Index', List[str], List[Dict[str, str]]]:
    """
    Main function: Create embeddings and FAISS index from chunks.
    This is the entry point for building the vector store.
    
    Args:
        chunks: List of text chunks from document ingestion
        metadata: List of metadata dictionaries (one per chunk, optional)
        
    Returns:
        Tuple of (FAISS index, original chunks list, metadata list)
        The index can be used to search for similar chunks
        If metadata is None, returns empty metadata list
    """
    # Step 1: Convert chunks to embeddings
    embeddings = create_embeddings(chunks)
    
    # Step 2: Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Step 3: Handle metadata (create default if not provided)
    if metadata is None:
        metadata = [{"source": "unknown"} for _ in chunks]
    
    return index, chunks, metadata


# Example usage (for testing)
if __name__ == "__main__":
    # This would typically be called after ingest.py
    # For testing, we'll create some dummy chunks
    test_chunks = [
        "This is a test chunk about artificial intelligence.",
        "Machine learning is a subset of AI.",
        "Natural language processing helps computers understand text."
    ]
    
    print("Building vector store from test chunks...")
    index, chunks, metadata = build_vector_store(test_chunks)
    print("Vector store built successfully!")
