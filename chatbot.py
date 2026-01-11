"""
chatbot.py - Main Terminal Chatbot Interface

This is the main entry point for the Explainable RAG Chatbot.
It orchestrates all components:
1. Document ingestion
2. Embedding and indexing
3. Query retrieval
4. Entity extraction and graph building
5. Answer generation

Usage:
    python chatbot.py

The chatbot will:
- Load documents from data/docs.pdf
- Build FAISS index (first time only)
- Run interactive Q&A loop
- Display answers with explanations
"""

import os
import sys
import json
from typing import List, Tuple, Dict
from glob import glob


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("  EXPLAINABLE RAG CHATBOT")
    print("  Using LLaMA 3 via Ollama")
    print("="*70 + "\n")


def print_separator():
    """Print visual separator."""
    print("\n" + "-"*70 + "\n")


def get_current_pdf_list(data_folder: str = "data") -> List[str]:
    """
    Get list of all PDF files currently in the data folder.
    
    Args:
        data_folder: Path to folder containing PDF files
        
    Returns:
        List of PDF filenames (sorted)
    """
    pdf_pattern = os.path.join(data_folder, "*.pdf")
    pdf_files = glob(pdf_pattern)
    pdf_filenames = sorted([os.path.basename(pdf) for pdf in pdf_files])
    return pdf_filenames


def get_indexed_pdf_list(indexed_files_path: str = "indexed_files.json") -> List[str]:
    """
    Get list of PDF files that were indexed previously.
    
    Args:
        indexed_files_path: Path to JSON file storing indexed files list
        
    Returns:
        List of PDF filenames from previous indexing, or empty list if file doesn't exist
    """
    if not os.path.exists(indexed_files_path):
        return []
    
    try:
        with open(indexed_files_path, 'r') as f:
            data = json.load(f)
            return data.get("indexed_files", [])
    except Exception as e:
        print(f"Warning: Could not read indexed files list: {e}")
        return []


def save_indexed_pdf_list(pdf_list: List[str], indexed_files_path: str = "indexed_files.json") -> None:
    """
    Save list of indexed PDF files to disk.
    
    Args:
        pdf_list: List of PDF filenames
        indexed_files_path: Path to JSON file to save to
    """
    try:
        data = {"indexed_files": pdf_list}
        with open(indexed_files_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved indexed files list to: {indexed_files_path}")
    except Exception as e:
        print(f"Warning: Could not save indexed files list: {e}")


def should_rebuild_index(data_folder: str = "data") -> bool:
    """
    Check if the index needs to be rebuilt based on PDF changes.
    
    Compares current PDF files in data/ folder with previously indexed files.
    
    Args:
        data_folder: Path to folder containing PDF files
        
    Returns:
        True if index should be rebuilt, False otherwise
    """
    current_pdfs = get_current_pdf_list(data_folder)
    indexed_pdfs = get_indexed_pdf_list()
    
    # Convert to sets for comparison (order doesn't matter)
    if set(current_pdfs) != set(indexed_pdfs):
        return True
    
    return False


def rebuild_index_files():
    """
    Delete old index files to force a rebuild.
    """
    index_files = ["faiss_index.bin", "chunks.pkl", "metadata.pkl", "document_graphs.pkl"]
    
    for file_path in index_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted old index file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")


def load_documents() -> Tuple[List[str], 'faiss.Index', List[Dict[str, str]], bool]:
    """
    Load documents and build/load vector store.
    Supports multiple PDFs from the data folder.
    Automatically rebuilds index if PDFs have changed.
    
    Returns:
        Tuple of (chunks, FAISS index, metadata list, was_rebuilt)
        was_rebuilt: True if index was rebuilt, False otherwise
    """
    from ingest import process_all_documents
    from embed_store import build_vector_store, load_index_and_chunks, save_index_and_chunks
    
    index_path = "faiss_index.bin"
    chunks_path = "chunks.pkl"
    metadata_path = "metadata.pkl"
    indexed_files_path = "indexed_files.json"
    
    # Step 0: Check if index needs to be rebuilt due to PDF changes
    current_pdfs = get_current_pdf_list("data")
    was_rebuilt = False
    
    if current_pdfs and should_rebuild_index("data"):
        print("PDF files have changed. Rebuilding index...")
        rebuild_index_files()
        was_rebuilt = True
    elif current_pdfs:
        # Save current list if it doesn't exist (first run)
        indexed_pdfs = get_indexed_pdf_list(indexed_files_path)
        if not indexed_pdfs:
            save_indexed_pdf_list(current_pdfs, indexed_files_path)
    
    # Check if index already exists (to avoid re-processing)
    if os.path.exists(index_path) and os.path.exists(chunks_path) and not was_rebuilt:
        print("Found existing FAISS index. Loading...")
        try:
            index, chunks, metadata = load_index_and_chunks(index_path, chunks_path, metadata_path)
            print("Index loaded successfully!")
            return chunks, index, metadata, False
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Will rebuild index...")
            was_rebuilt = True
    
    # Step 1: Load and chunk all PDFs from data folder
    print("Step 1: Loading and chunking PDF documents...")
    try:
        chunks, metadata = process_all_documents("data")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to process documents: {e}")
        sys.exit(1)
    
    if not chunks:
        print("ERROR: No chunks created from PDFs. Check if PDFs have readable text.")
        sys.exit(1)
    
    # Step 2: Create embeddings and FAISS index
    print("\nStep 2: Creating embeddings and FAISS index...")
    index, chunks, metadata = build_vector_store(chunks, metadata)
    
    # Save for future use
    try:
        save_index_and_chunks(index, chunks, metadata, index_path, chunks_path, metadata_path)
        # Save list of indexed PDFs
        save_indexed_pdf_list(current_pdfs, indexed_files_path)
    except Exception as e:
        print(f"Warning: Could not save index: {e}")
    
    return chunks, index, metadata, was_rebuilt


def initialize_spacy():
    """Initialize spaCy model for NER."""
    from entity_graph import load_spacy_model
    
    print("Loading spaCy model for entity extraction...")
    try:
        nlp = load_spacy_model()
        print("spaCy model ready!")
        return nlp
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTo fix this, run:")
        print("  python -m spacy download en_core_web_sm")
        sys.exit(1)


def check_ollama():
    """Check if Ollama is available."""
    from generate import check_ollama_available, ensure_llama3_model
    
    print("Checking Ollama connection...")
    if not check_ollama_available():
        print("ERROR: Ollama is not running or not installed.")
        print("\nTo fix this:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Download model: ollama pull llama3")
        sys.exit(1)
    
    print("Ollama is available!")
    ensure_llama3_model()


def process_query(
    query: str,
    index: 'faiss.Index',
    chunks: List[str],
    metadata: List[Dict[str, str]],
    nlp,
    k: int = 3
) -> Tuple[str, str, str, str, dict]:
    """
    Process a user query through the full pipeline.
    
    Args:
        query: User's question
        index: FAISS index
        chunks: List of text chunks
        metadata: List of metadata dictionaries
        nlp: spaCy model
        k: Number of chunks to retrieve
        
    Returns:
        Tuple of (answer, source, confidence, explanation_text, explanation_dict)
    """
    from retrieve import (
        retrieve_chunks, get_context_text, 
        get_most_frequent_source, calculate_confidence
    )
    from entity_graph import build_entity_graph, get_graph_explanation, format_explanation
    from generate import generate_answer_with_fallback
    
    # Step 1: Retrieve relevant chunks with metadata
    print(f"\nRetrieving top-{k} relevant chunks...")
    retrieved_chunks, retrieved_metadata, avg_similarity = retrieve_chunks(
        query, index, chunks, metadata, k=k
    )
    
    if not retrieved_chunks:
        return (
            "Not found in the provided documents.",
            "unknown",
            "Low",
            "No relevant chunks found.",
            {'entities': [], 'relations': []}
        )
    
    # Step 2: Extract context text
    context = get_context_text(retrieved_chunks)
    
    # Step 3: Get source attribution
    source = get_most_frequent_source(retrieved_metadata)
    
    # Step 4: Calculate confidence
    confidence = calculate_confidence(avg_similarity)
    
    # Step 5: Build entity graph
    print("Extracting entities and building knowledge graph...")
    graph = build_entity_graph(retrieved_chunks, nlp)
    explanation_dict = get_graph_explanation(graph)
    explanation_text = format_explanation(explanation_dict)
    
    # Step 6: Generate answer
    print("Generating answer using LLaMA 3...")
    answer = generate_answer_with_fallback(query, context)
    
    return answer, source, confidence, explanation_text, explanation_dict


def build_and_save_document_graphs(chunks: List[str], metadata: List[Dict[str, str]], nlp):
    """
    Build document-level knowledge graphs and save them.
    Only called when index is rebuilt.
    """
    from document_graph import build_all_document_graphs, save_document_graphs
    
    print("\nBuilding document-level knowledge graphs...")
    try:
        document_graphs = build_all_document_graphs(chunks, metadata, nlp)
        save_document_graphs(document_graphs)
        print("Document graphs built and saved successfully!")
        return document_graphs
    except Exception as e:
        print(f"Warning: Could not build document graphs: {e}")
        return {}


def handle_show_graph_command(pdf_name: str, document_graphs: Dict[str, Dict]):
    """
    Handle the /show_graph command to display a document's knowledge graph.
    
    Args:
        pdf_name: PDF filename or partial name to search for
        document_graphs: Dictionary of all document graphs
        
    Returns:
        Formatted graph display string, or error message
    """
    from document_graph import format_document_graph
    
    # Normalize input (case-insensitive, remove spaces)
    pdf_name_lower = pdf_name.lower().strip()
    
    # Try exact match first
    matching_keys = [key for key in document_graphs.keys() if pdf_name_lower in key.lower()]
    
    if not matching_keys:
        return f"Document '{pdf_name}' not found in indexed files."
    
    # Use first match (or exact match if available)
    if pdf_name in document_graphs:
        selected_key = pdf_name
    else:
        selected_key = matching_keys[0]
    
    graph_data = document_graphs[selected_key]
    
    output = [f"\nDocument Graph: {selected_key}"]
    output.append("="*70)
    output.append(format_document_graph(graph_data, max_entities=15, max_relations=10))
    
    return "\n".join(output)


def main():
    """Main chatbot loop."""
    print_banner()
    
    # Initialize all components
    print("Initializing chatbot components...\n")
    
    # Initialize spaCy first (needed for document graph building)
    nlp = initialize_spacy()
    print_separator()
    
    # Load documents and build index (checks for PDF changes automatically)
    chunks, index, metadata, index_was_rebuilt = load_documents()
    
    # Check if we need to rebuild document graphs
    from document_graph import load_document_graphs
    document_graphs = load_document_graphs()
    
    # Rebuild document graphs if:
    # 1. They don't exist, OR
    # 2. Index was just rebuilt (PDFs changed)
    if not document_graphs or index_was_rebuilt:
        if index_was_rebuilt:
            print("Rebuilding document graphs due to index rebuild...")
        document_graphs = build_and_save_document_graphs(chunks, metadata, nlp)
    else:
        # Verify that document graphs match current PDFs
        current_pdfs = get_current_pdf_list("data")
        graph_sources = set(document_graphs.keys())
        current_pdf_set = set(current_pdfs)
        
        # If PDFs were added/removed, rebuild graphs
        if graph_sources != current_pdf_set:
            print("Document graphs are out of sync. Rebuilding...")
            document_graphs = build_and_save_document_graphs(chunks, metadata, nlp)
    
    print_separator()
    
    # Check Ollama
    check_ollama()
    print_separator()
    
    print("Chatbot ready! Type your questions below.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type '/show_graph <pdf_name>' to view a document's knowledge graph.\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            query = input("You: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("\nGoodbye!")
                break
            
            # Check for special commands
            if query.startswith('/show_graph'):
                # Extract PDF name from command
                parts = query.split(' ', 1)
                if len(parts) > 1:
                    pdf_name = parts[1]
                    result = handle_show_graph_command(pdf_name, document_graphs)
                    print(result)
                else:
                    print("Usage: /show_graph <pdf_filename>")
                print_separator()
                continue
            
            # Process regular query
            answer, source, confidence, explanation, _ = process_query(
                query, index, chunks, metadata, nlp
            )
            
            # Display results
            print_separator()
            print("Answer:")
            print(answer)
            print_separator()
            print("Source:")
            print(source)
            print_separator()
            print("Confidence:")
            print(confidence)
            print_separator()
            print("Explanation:")
            print(explanation)
            print_separator()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
