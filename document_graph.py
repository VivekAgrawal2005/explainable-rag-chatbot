"""
document_graph.py - Document-Level Knowledge Graph Generation

This module handles:
1. Building knowledge graphs for entire documents (one per PDF)
2. Storing document graphs for reuse across queries
3. Generating graphs only when documents change

Graph Structure:
- Nodes: Named entities (PERSON, ORG, GPE, LOC, FAC)
- Edges: Sentence-level co-occurrence relationships

Dependencies:
- spacy: For named entity recognition
- networkx: For graph data structure
"""

import pickle
import os
from typing import List, Dict, Tuple
import networkx as nx


def build_document_graph(
    chunks: List[str],
    metadata: List[Dict[str, str]],
    source_file: str,
    nlp
) -> Dict:
    """
    Build a knowledge graph for a single document from all its chunks.
    
    This creates a document-wide entity relationship graph by:
    1. Collecting all chunks belonging to the document
    2. Extracting entities using the same rules as query-level graphs
    3. Building relationships based on sentence-level co-occurrence
    
    Args:
        chunks: List of all text chunks
        metadata: List of metadata dictionaries (one per chunk)
        source_file: PDF filename to build graph for
        nlp: Loaded spaCy model
        
    Returns:
        Dictionary containing graph data:
        {
            "nodes": [(entity_name, entity_type), ...],
            "edges": [(source, target, weight), ...]
        }
    """
    from entity_graph import extract_entities, normalize_entity_name
    
    # Collect all chunks belonging to this document
    document_chunks = []
    for chunk, meta in zip(chunks, metadata):
        if meta.get("source") == source_file:
            document_chunks.append(chunk)
    
    if not document_chunks:
        return {"nodes": [], "edges": []}
    
    print(f"Building document graph for '{source_file}' ({len(document_chunks)} chunks)...")
    
    # Combine all chunks from this document
    document_text = " ".join(document_chunks)
    
    # Extract all entities from the document
    all_entities = []
    entity_counts = {}
    entity_labels = {}
    
    # Process document in chunks to handle large documents
    for chunk_text in document_chunks:
        # Extract entities from this chunk
        chunk_entities = extract_entities(chunk_text, nlp)
        
        # Track unique entities
        chunk_entity_set = set()
        for entity_text, entity_label in chunk_entities:
            normalized = normalize_entity_name(entity_text)
            chunk_entity_set.add((normalized, entity_label))
            entity_labels[normalized] = entity_label
        
        # Update counts
        for entity_text, _ in chunk_entity_set:
            entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
        
        all_entities.extend(chunk_entity_set)
    
    # Select top entities (up to 15 for document-level graphs)
    # Sort by frequency
    sorted_entities = sorted(
        entity_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    selected_entities = {entity: entity_labels[entity] for entity, _ in sorted_entities}
    
    # Build graph with selected entities
    graph = nx.Graph()
    
    # Add nodes
    for entity_text, entity_label in selected_entities.items():
        graph.add_node(entity_text, entity_type=entity_label)
    
    # Build edges based on sentence-level co-occurrence
    for chunk_text in document_chunks:
        # Process chunk into sentences
        doc = nlp(chunk_text)
        
        for sentence in doc.sents:
            # Extract entities from this sentence
            sentence_entities = extract_entities(sentence.text, nlp)
            
            # Normalize and filter to selected entities only
            sentence_entity_set = set()
            for entity_text, _ in sentence_entities:
                normalized = normalize_entity_name(entity_text)
                if normalized in selected_entities:
                    sentence_entity_set.add(normalized)
            
            # Create edges between entities in the same sentence
            sentence_entities_list = list(sentence_entity_set)
            for i, ent1 in enumerate(sentence_entities_list):
                for ent2 in sentence_entities_list[i+1:]:
                    if ent1 != ent2:
                        # Add edge (or increment weight)
                        if graph.has_edge(ent1, ent2):
                            graph[ent1][ent2]['weight'] += 1
                        else:
                            graph.add_edge(ent1, ent2, weight=1)
    
    # Convert graph to serializable format
    nodes = []
    for node in graph.nodes():
        entity_type = graph.nodes[node].get('entity_type', 'UNKNOWN')
        nodes.append((node, entity_type))
    
    edges = []
    for source, target, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        edges.append((source, target, weight))
    
    print(f"  Graph built: {len(nodes)} entities, {len(edges)} relationships")
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def build_all_document_graphs(
    chunks: List[str],
    metadata: List[Dict[str, str]],
    nlp
) -> Dict[str, Dict]:
    """
    Build knowledge graphs for all documents.
    
    Args:
        chunks: List of all text chunks
        metadata: List of metadata dictionaries
        nlp: Loaded spaCy model
        
    Returns:
        Dictionary mapping PDF filenames to their graph data:
        {
            "filename1.pdf": {"nodes": [...], "edges": [...]},
            "filename2.pdf": {"nodes": [...], "edges": [...]},
            ...
        }
    """
    # Get unique source files
    unique_sources = set()
    for meta in metadata:
        source = meta.get("source", "")
        if source and source != "unknown":
            unique_sources.add(source)
    
    document_graphs = {}
    
    for source_file in unique_sources:
        try:
            graph_data = build_document_graph(chunks, metadata, source_file, nlp)
            document_graphs[source_file] = graph_data
        except Exception as e:
            print(f"Warning: Failed to build graph for {source_file}: {e}")
            continue
    
    return document_graphs


def save_document_graphs(
    document_graphs: Dict[str, Dict],
    file_path: str = "document_graphs.pkl"
) -> None:
    """
    Save document graphs to disk.
    
    Args:
        document_graphs: Dictionary of document graphs
        file_path: Path to save the graphs
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(document_graphs, f)
        print(f"Saved document graphs to: {file_path}")
    except Exception as e:
        print(f"Warning: Could not save document graphs: {e}")


def load_document_graphs(
    file_path: str = "document_graphs.pkl"
) -> Dict[str, Dict]:
    """
    Load document graphs from disk.
    
    Args:
        file_path: Path to load the graphs from
        
    Returns:
        Dictionary of document graphs, or empty dict if file doesn't exist
    """
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'rb') as f:
            document_graphs = pickle.load(f)
        print(f"Loaded document graphs from: {file_path}")
        return document_graphs
    except Exception as e:
        print(f"Warning: Could not load document graphs: {e}")
        return {}


def format_document_graph(graph_data: Dict, max_entities: int = 15, max_relations: int = 10) -> str:
    """
    Format document graph data into human-readable text.
    
    Args:
        graph_data: Graph data dictionary with "nodes" and "edges"
        max_entities: Maximum number of entities to show
        max_relations: Maximum number of relations to show
        
    Returns:
        Formatted string ready for display
    """
    output = []
    
    # Format entities
    nodes = graph_data.get("nodes", [])
    output.append("Entities:")
    if nodes:
        # Sort by entity type, then name
        sorted_nodes = sorted(nodes[:max_entities], key=lambda x: (x[1], x[0]))
        for entity_text, entity_type in sorted_nodes:
            output.append(f"- {entity_text} ({entity_type})")
    else:
        output.append("(No entities found)")
    
    # Format relations
    edges = graph_data.get("edges", [])
    output.append("\nRelations:")
    if edges:
        # Sort by weight (descending) and limit
        sorted_edges = sorted(edges, key=lambda x: x[2] if len(x) > 2 else 1, reverse=True)
        for edge in sorted_edges[:max_relations]:
            if len(edge) >= 2:
                source, target = edge[0], edge[1]
                output.append(f"- {source} -> associated_with -> {target}")
    else:
        output.append("(No relations found)")
    
    return "\n".join(output)


# Example usage (for testing)
if __name__ == "__main__":
    print("This module is used by chatbot.py to build document-level graphs.")
    print("Run 'python chatbot.py' to use this functionality.")
