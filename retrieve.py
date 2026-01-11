"""
retrieve.py - Query Embedding and Chunk Retrieval

This module handles:
1. Converting user queries into embeddings
2. Searching the FAISS index for most relevant chunks
3. Returning top-k most similar chunks

Dependencies:
- sentence-transformers: For query embedding (same model as embed_store.py)
- faiss: For similarity search
"""

import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from collections import Counter


def extract_document_hint(query: str, metadata: List[Dict[str, str]]) -> Optional[str]:
    """
    Detect document hints by keyword overlap between query and filenames.
    This version is robust to phrases like:
    'In the novel In Search of the Castaways'
    """

    if not metadata:
        return None

    # Normalize and tokenize query
    query_tokens = set(
        query.lower()
        .replace(".pdf", "")
        .replace("the", "")
        .replace("novel", "")
        .replace("in", "")
        .split()
    )

    # Collect unique source filenames
    sources = set(
        meta.get("source", "")
        for meta in metadata
        if meta.get("source") and meta.get("source") != "unknown"
    )

    for source in sources:
        source_tokens = set(
            source.lower()
            .replace(".pdf", "")
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )

        # If enough meaningful words overlap, treat as a match
        if len(query_tokens.intersection(source_tokens)) >= 2:
            return source

    return None



def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convert a user query into an embedding vector.
    
    Uses the same model as embed_store.py to ensure embeddings
    are in the same vector space and comparable.
    
    Args:
        query: User's question or search query
        model_name: Name of the sentence-transformer model
                    Must match the model used in embed_store.py
        
    Returns:
        NumPy array of shape (1, embedding_dimension)
        Single embedding vector for the query
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the same model used for chunk embeddings
        model = SentenceTransformer(model_name)
        
        # Generate embedding for the query
        # Note: encode returns shape (1, dim) for single string
        query_embedding = model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Ensure it's 2D (even for single query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        return query_embedding
        
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )
    except Exception as e:
        raise Exception(f"Error embedding query: {str(e)}")


def retrieve_chunks(
    query: str,
    index: 'faiss.Index',
    chunks: List[str],
    metadata: List[Dict[str, str]] = None,
    k: int = 3,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[List[Tuple[str, float]], List[Dict[str, str]], float]:
    """
    Retrieve the top-k most relevant chunks for a query.
    
    Supports document-aware retrieval: if query mentions a specific document,
    retrieval is filtered to that document only (with fallback to global search).
    
    Process:
    1. Extract document hint from query (if any)
    2. Convert query to embedding
    3. Search FAISS index for similar embeddings
    4. Filter results by document hint (if detected)
    5. Fall back to unfiltered results if filtered results are empty
    6. Return corresponding text chunks with similarity scores and metadata
    
    Args:
        query: User's search query
        index: FAISS index containing chunk embeddings
        chunks: List of original text chunks (must match index order)
        metadata: List of metadata dictionaries (one per chunk, optional)
        k: Number of top chunks to retrieve (default: 3)
        model_name: Embedding model name (must match embed_store.py)
        
    Returns:
        Tuple of:
        - retrieved_chunks: List of (chunk_text, similarity_score) tuples
        - retrieved_metadata: List of metadata dicts for retrieved chunks
        - avg_similarity: Average similarity score (for confidence calculation)
        Sorted by relevance (highest similarity first)
    """
    try:
        import faiss
        
        # Handle missing metadata (backward compatibility)
        if metadata is None:
            metadata = [{"source": "unknown"} for _ in chunks]
        
        # Step 1: Extract document hint from query (if any)
        document_hint = extract_document_hint(query, metadata)
        print(f"Document hint detected: {document_hint}")
        use_filtered = document_hint is not None
        
        if use_filtered:
            print(f"Document hint detected: '{document_hint}' - filtering retrieval...")
        
        # Step 2: Convert query to embedding
        query_embedding = embed_query(query, model_name)
        
        # Step 3: Normalize query embedding (same as chunks were normalized)
        faiss.normalize_L2(query_embedding)
        
        # Step 4: Search FAISS index for top-k similar vectors
        # We search for more results if filtering is needed (to ensure we get k results after filtering)
        search_k = k * 3 if use_filtered else k
        distances, indices = index.search(query_embedding, search_k)
        
        # Step 5: Extract corresponding chunks and metadata
        candidate_chunks = []
        candidate_metadata = []
        candidate_similarities = []
        
        # distances[0] and indices[0] because query_embedding has shape (1, dim)
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # idx is the position in the chunks list
            if idx < len(chunks):
                chunk_text = chunks[idx]
                # Convert distance to similarity score
                # For cosine similarity: similarity = 1 - distance (after normalization)
                similarity = 1 - distance
                
                # Get corresponding metadata
                chunk_metadata = metadata[idx] if idx < len(metadata) else {"source": "unknown"}
                
                # If filtering by document, only include chunks from that document
                if use_filtered:
                    chunk_source = chunk_metadata.get("source", "unknown")
                    if chunk_source != document_hint:
                        continue  # Skip chunks from other documents
                
                candidate_chunks.append((chunk_text, similarity))
                candidate_metadata.append(chunk_metadata)
                candidate_similarities.append(similarity)
                
                # Stop when we have enough results
                if len(candidate_chunks) >= k:
                    break
        
        # Step 6: Fallback to unfiltered results if filtered search returned nothing
        if use_filtered and not candidate_chunks:
            print(f"Filtered search returned no results. Falling back to global search...")
            # Search for more results to ensure we get k chunks
            fallback_k = min(k * 5, index.ntotal)  # Search up to 5x more, but not more than total chunks
            fallback_distances, fallback_indices = index.search(query_embedding, fallback_k)
            
            # Retry without filtering, using the expanded search results
            for i, (distance, idx) in enumerate(zip(fallback_distances[0], fallback_indices[0])):
                if idx < len(chunks) and len(candidate_chunks) < k:
                    chunk_text = chunks[idx]
                    similarity = 1 - distance
                    chunk_metadata = metadata[idx] if idx < len(metadata) else {"source": "unknown"}
                    candidate_chunks.append((chunk_text, similarity))
                    candidate_metadata.append(chunk_metadata)
                    candidate_similarities.append(similarity)
        
        # Step 7: Calculate average similarity (for confidence score)
        avg_similarity = sum(candidate_similarities) / len(candidate_similarities) if candidate_similarities else 0.0
        
        retrieval_type = "document-filtered" if use_filtered and candidate_chunks else "global"
        print(f"Retrieved {len(candidate_chunks)} chunks ({retrieval_type}) for query: '{query[:50]}...'")
        
        return candidate_chunks, candidate_metadata, avg_similarity
        
    except ImportError:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
    except Exception as e:
        raise Exception(f"Error retrieving chunks: {str(e)}")


def get_context_text(retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Combine retrieved chunks into a single context string.
    
    This context will be used by the LLM to generate answers.
    
    Args:
        retrieved_chunks: List of (chunk_text, similarity_score) tuples
        
    Returns:
        Single string containing all retrieved chunks, separated by newlines
    """
    # Extract just the text from tuples
    chunk_texts = [chunk for chunk, score in retrieved_chunks]
    
    # Combine with separator
    context = "\n\n".join(chunk_texts)
    
    return context


def get_most_frequent_source(retrieved_metadata: List[Dict[str, str]]) -> str:
    """
    Get the most frequently occurring source file from retrieved chunks.
    
    Args:
        retrieved_metadata: List of metadata dictionaries
        
    Returns:
        Most frequent source filename, or "unknown" if no metadata
    """
    if not retrieved_metadata:
        return "unknown"
    
    # Count occurrences of each source
    sources = [meta.get("source", "unknown") for meta in retrieved_metadata]
    source_counts = Counter(sources)
    
    # Return the most common source
    most_common = source_counts.most_common(1)
    if most_common:
        return most_common[0][0]
    return "unknown"


def calculate_confidence(avg_similarity: float) -> str:
    """
    Calculate confidence level based on average similarity score.
    
    Simple heuristic:
    - High: avg_similarity >= 0.75
    - Medium: 0.5 <= avg_similarity < 0.75
    - Low: avg_similarity < 0.5
    
    Args:
        avg_similarity: Average similarity score from retrieved chunks
        
    Returns:
        Confidence level: "High", "Medium", or "Low"
    """
    if avg_similarity >= 0.75:
        return "High"
    elif avg_similarity >= 0.5:
        return "Medium"
    else:
        return "Low"


# Example usage (for testing)
if __name__ == "__main__":
    # This would typically be called with a real index and chunks
    # For testing, we need to build them first
    print("This module requires a FAISS index and chunks.")
    print("Run embed_store.py first, then use retrieve_chunks() function.")
