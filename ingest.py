"""
ingest.py - PDF Document Loading and Text Chunking

This module handles:
1. Loading PDF documents from the data folder
2. Extracting text from PDF pages
3. Splitting text into manageable chunks (~300 words each)

Dependencies:
- PyPDF2: Free library for PDF text extraction
"""

import os
from typing import List, Tuple, Dict
from glob import glob


def load_pdf(pdf_path: str) -> str:
    """
    Load a PDF file and extract all text content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complete text content from the PDF as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read
    """
    try:
        # Import PyPDF2 for PDF reading
        from PyPDF2 import PdfReader
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open and read the PDF
        reader = PdfReader(pdf_path)
        full_text = ""
        
        # Extract text from each page
        print(f"Loading PDF: {pdf_path}")
        print(f"Total pages: {len(reader.pages)}")
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            full_text += page_text + "\n"
            print(f"Processed page {page_num}/{len(reader.pages)}")
        
        print(f"Extracted {len(full_text)} characters of text")
        return full_text
        
    except ImportError:
        raise ImportError("PyPDF2 is required. Install it with: pip install PyPDF2")
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of approximately 'chunk_size' words.
    Uses overlap to maintain context between chunks.
    
    Args:
        text: The full text to chunk
        chunk_size: Target number of words per chunk (default: 300)
        overlap: Number of words to overlap between chunks (default: 50)
        
    Returns:
        List of text chunks, each containing approximately chunk_size words
    """
    # Split text into words (simple whitespace split)
    words = text.split()
    
    # If text is shorter than chunk_size, return as single chunk
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start_idx = 0
    
    # Create chunks with overlap
    while start_idx < len(words):
        # Calculate end index for this chunk
        end_idx = start_idx + chunk_size
        
        # Extract words for this chunk
        chunk_words = words[start_idx:end_idx]
        
        # Join words back into text
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Move start index forward, accounting for overlap
        # This ensures chunks share some context
        start_idx = end_idx - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start_idx += 1
    
    print(f"Created {len(chunks)} text chunks (target size: {chunk_size} words)")
    return chunks


def process_document(pdf_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Load a single PDF and return chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Tuple of (chunks, metadata_list)
        Each metadata dict contains: {"source": filename}
    """
    # Step 1: Load PDF and extract text
    full_text = load_pdf(pdf_path)
    
    # Step 2: Split into chunks
    chunks = chunk_text(full_text)
    
    # Step 3: Create metadata for each chunk
    # Extract just the filename from the path
    filename = os.path.basename(pdf_path)
    metadata_list = [{"source": filename} for _ in chunks]
    
    return chunks, metadata_list


def process_all_documents(data_folder: str = "data") -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Main function: Load ALL PDFs from data folder and return chunks with metadata.
    This is the entry point for multi-document ingestion.
    
    Args:
        data_folder: Path to folder containing PDF files (default: "data")
        
    Returns:
        Tuple of (all_chunks, all_metadata)
        - all_chunks: List of all text chunks from all PDFs
        - all_metadata: List of metadata dicts, one per chunk
        Each metadata dict contains: {"source": PDF_filename}
    """
    # Find all PDF files in the data folder
    pdf_pattern = os.path.join(data_folder, "*.pdf")
    pdf_files = glob(pdf_pattern)
    
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_folder}' folder. "
            f"Please place at least one PDF file in the '{data_folder}' directory."
        )
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{data_folder}' folder")
    
    all_chunks = []
    all_metadata = []
    
    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        try:
            chunks, metadata_list = process_document(pdf_path)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata_list)
            print(f"  Added {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  Warning: Failed to process {pdf_path}: {e}")
            continue
    
    if not all_chunks:
        raise Exception("No chunks were created from any PDF files. Check if PDFs contain readable text.")
    
    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} PDF file(s)")
    
    return all_chunks, all_metadata


# Example usage (for testing)
if __name__ == "__main__":
    # Test the multi-document ingestion pipeline
    try:
        chunks, metadata = process_all_documents("data")
        print(f"\nFirst chunk preview (first 200 chars):")
        print(chunks[0][:200] + "...")
        print(f"\nFirst chunk source: {metadata[0]['source']}")
    except Exception as e:
        print(f"Error: {e}")
