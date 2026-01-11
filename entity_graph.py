"""
entity_graph.py - Named Entity Recognition and Knowledge Graph

This module handles:
1. Extracting named entities from retrieved text chunks
2. Building a simple knowledge graph (nodes = entities, edges = relations)
3. Visualizing relationships between entities

Dependencies:
- spacy: Natural language processing library
- networkx: Graph data structure library
"""

from typing import List, Dict, Set, Tuple
import networkx as nx


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load spaCy language model for named entity recognition (NER).
    
    Note: First time requires downloading the model:
    python -m spacy download en_core_web_sm
    
    Args:
        model_name: Name of spaCy model to load
        
    Returns:
        Loaded spaCy nlp model
    """
    try:
        import spacy
        
        print(f"Loading spaCy model: {model_name}")
        print("(First time: run 'python -m spacy download en_core_web_sm')")
        
        nlp = spacy.load(model_name)
        print("spaCy model loaded successfully")
        
        return nlp
        
    except OSError:
        raise OSError(
            f"spaCy model '{model_name}' not found. "
            f"Download it with: python -m spacy download {model_name}"
        )
    except ImportError:
        raise ImportError("spacy is required. Install with: pip install spacy")
    except Exception as e:
        raise Exception(f"Error loading spaCy model: {str(e)}")


def extract_entities(text: str, nlp) -> List[Tuple[str, str]]:
    """
    Extract named entities from text using spaCy NER.
    
    Filters entities to keep only meaningful ones:
    - PERSON: People's names
    - ORG: Organizations
    - GPE: Countries, cities (geopolitical entities)
    - LOC: Locations
    - FAC: Facilities
    
    Ignores: CARDINAL, DATE, TIME, QUANTITY, PERCENT, MONEY, ORDINAL
    
    Args:
        text: Text to extract entities from
        nlp: Loaded spaCy model
        
    Returns:
        List of tuples: [(entity_text, entity_label), ...]
        Example: [("Apple Inc", "ORG"), ("John Smith", "PERSON")]
    """
    # Allowed entity labels (only meaningful entities)
    ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC"}
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract entities
    entities = []
    seen = set()  # Avoid duplicates
    
    for ent in doc.ents:
        entity_label = ent.label_
        
        # Filter: Only keep allowed labels
        if entity_label not in ALLOWED_LABELS:
            continue
        
        # Normalize entity text (strip whitespace)
        entity_text = ent.text.strip()
        
        # Filter: Remove entities shorter than 3 characters
        if len(entity_text) < 3:
            continue
        
        # Create unique key to avoid duplicates (case-insensitive)
        entity_key = (entity_text.lower(), entity_label)
        
        if entity_key not in seen:
            entities.append((entity_text, entity_label))
            seen.add(entity_key)
    
    return entities


def normalize_entity_name(entity_text: str) -> str:
    """
    Normalize entity names by removing common titles and prefixes.
    
    Examples:
    - "M. Edmond Dantès" -> "Edmond Dantès"
    - "Monsieur Dantès" -> "Dantès"
    - "the United States" -> "United States"
    
    Args:
        entity_text: Original entity text
        
    Returns:
        Normalized entity text
    """
    # Common titles and prefixes to remove
    titles = ["M.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Monsieur", "Madame", "the ", "The "]
    
    normalized = entity_text.strip()
    
    # Remove titles from the beginning
    for title in titles:
        if normalized.startswith(title):
            normalized = normalized[len(title):].strip()
    
    return normalized


def merge_duplicate_entities(entities: List[Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    """
    Merge duplicate entity mentions (case-insensitive, handles partial matches).
    
    Examples:
    - "Edmond", "Dantès", "Edmond Dantès" -> "Edmond Dantès"
    - "Apple", "Apple Inc" -> "Apple Inc" (prefer longer, more complete)
    
    Args:
        entities: List of (entity_text, entity_label) tuples
        
    Returns:
        Dictionary mapping normalized keys to (entity_text, entity_label)
        Prefers longer, more complete entity names
    """
    # Dictionary to store merged entities
    # Key: normalized lowercase name, Value: (best_entity_text, entity_label)
    merged = {}
    
    for entity_text, entity_label in entities:
        # Normalize the entity name
        normalized = normalize_entity_name(entity_text)
        key = normalized.lower()
        
        if key not in merged:
            # First occurrence of this entity
            merged[key] = (normalized, entity_label)
        else:
            # Entity already exists - prefer longer, more complete name
            existing_text, existing_label = merged[key]
            if len(normalized) > len(existing_text):
                merged[key] = (normalized, entity_label)
            # Also check if current entity contains the existing one
            elif normalized.lower() in existing_text.lower():
                # Current is subset of existing, keep existing
                pass
            elif existing_text.lower() in normalized.lower():
                # Existing is subset of current, use current
                merged[key] = (normalized, entity_label)
    
    return merged


def select_top_entities(
    entity_counts: Dict[str, int],
    entity_labels: Dict[str, str],
    max_entities: int = 7
) -> List[Tuple[str, str]]:
    """
    Select top N entities based on frequency and importance.
    
    Strategy:
    1. Always include the most frequent PERSON entity
    2. Prefer entities that appear in multiple chunks
    3. Limit to max_entities total
    
    Args:
        entity_counts: Dictionary mapping entity names to occurrence counts
        entity_labels: Dictionary mapping entity names to labels
        max_entities: Maximum number of entities to select (default: 7)
        
    Returns:
        List of (entity_text, entity_label) tuples, sorted by importance
    """
    # Separate entities by type
    person_entities = []
    other_entities = []
    
    for entity_text, count in entity_counts.items():
        label = entity_labels.get(entity_text, "UNKNOWN")
        if label == "PERSON":
            person_entities.append((entity_text, label, count))
        else:
            other_entities.append((entity_text, label, count))
    
    # Sort by frequency (descending)
    person_entities.sort(key=lambda x: x[2], reverse=True)
    other_entities.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    
    # Always include the most frequent PERSON if available
    if person_entities:
        selected.append((person_entities[0][0], person_entities[0][1]))
    
    # Add remaining entities (prioritize those in multiple chunks)
    remaining_slots = max_entities - len(selected)
    
    # Combine remaining persons and others, sorted by frequency
    all_remaining = person_entities[1:] + other_entities
    all_remaining.sort(key=lambda x: x[2], reverse=True)
    
    # Add entities that appear in multiple chunks first
    multi_chunk = [(e, l, c) for e, l, c in all_remaining if c > 1]
    single_chunk = [(e, l, c) for e, l, c in all_remaining if c == 1]
    
    for entity_text, label, _ in multi_chunk + single_chunk:
        if len(selected) >= max_entities:
            break
        if (entity_text, label) not in selected:
            selected.append((entity_text, label))
    
    return selected


def build_entity_graph(
    retrieved_chunks: List[Tuple[str, float]],
    nlp,
    max_entities: int = 7
) -> nx.Graph:
    """
    Build a simplified knowledge graph from entities found in retrieved chunks.
    
    Graph structure:
    - Nodes: Top N named entities (filtered and normalized)
    - Edges: Sentence-level co-occurrence (entities in same sentence)
    
    Args:
        retrieved_chunks: List of (chunk_text, similarity_score) tuples
        nlp: Loaded spaCy model
        max_entities: Maximum number of entities to include (default: 7)
        
    Returns:
        NetworkX graph with entities as nodes and relations as edges
    """
    # Step 1: Extract entities from each chunk separately
    all_entities = []
    entity_counts = {}  # Track how many chunks each entity appears in
    entity_labels = {}  # Store entity labels
    
    for chunk_text, _ in retrieved_chunks:
        # Extract entities from this chunk
        chunk_entities = extract_entities(chunk_text, nlp)
        
        # Track unique entities in this chunk
        chunk_entity_set = set()
        
        for entity_text, entity_label in chunk_entities:
            # Normalize entity name
            normalized = normalize_entity_name(entity_text)
            chunk_entity_set.add((normalized, entity_label))
            entity_labels[normalized] = entity_label
        
        # Update counts (entity appears in this chunk)
        for entity_text, _ in chunk_entity_set:
            entity_counts[entity_text] = entity_counts.get(entity_text, 0) + 1
        
        all_entities.extend(chunk_entity_set)
    
    # Step 2: Merge duplicate entities
    merged_entities = merge_duplicate_entities(all_entities)
    
    # Update counts and labels for merged entities
    merged_counts = {}
    merged_labels = {}
    
    # First, map all original entities to their merged versions
    entity_to_merged = {}
    for entity_text, entity_label in all_entities:
        key = entity_text.lower()
        if key in merged_entities:
            merged_text, merged_label = merged_entities[key]
            entity_to_merged[entity_text] = merged_text
            merged_labels[merged_text] = merged_label
    
    # Now aggregate counts for merged entities
    for entity_text, count in entity_counts.items():
        # Find the merged version of this entity
        if entity_text in entity_to_merged:
            merged_text = entity_to_merged[entity_text]
            merged_counts[merged_text] = merged_counts.get(merged_text, 0) + count
        else:
            # Entity wasn't in all_entities (shouldn't happen, but handle it)
            merged_counts[entity_text] = count
            if entity_text not in merged_labels:
                merged_labels[entity_text] = entity_labels.get(entity_text, "UNKNOWN")
    
    # Step 3: Select top entities
    selected_entities = select_top_entities(merged_counts, merged_labels, max_entities)
    
    print(f"Selected {len(selected_entities)} entities from {len(merged_entities)} candidates")
    
    # Step 4: Create graph with selected entities only
    graph = nx.Graph()
    
    # Add nodes
    selected_entity_set = {entity_text for entity_text, _ in selected_entities}
    for entity_text, entity_label in selected_entities:
        graph.add_node(entity_text, entity_type=entity_label)
    
    # Step 5: Build edges based on sentence-level co-occurrence
    for chunk_text, _ in retrieved_chunks:
        # Process chunk into sentences
        doc = nlp(chunk_text)
        
        for sentence in doc.sents:
            # Extract entities from this sentence
            sentence_entities = extract_entities(sentence.text, nlp)
            
            # Normalize and filter to selected entities only
            sentence_entity_set = set()
            for entity_text, _ in sentence_entities:
                normalized = normalize_entity_name(entity_text)
                if normalized in selected_entity_set:
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
                            graph.add_edge(
                                ent1,
                                ent2,
                                relation="associated_with",
                                weight=1
                            )
    
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    return graph


def get_graph_explanation(graph: nx.Graph, max_relations: int = 10) -> Dict[str, List]:
    """
    Extract human-readable explanation from the knowledge graph.
    
    Returns entities and relationships in a structured format
    for display to the user.
    
    Args:
        graph: NetworkX knowledge graph
        max_relations: Maximum number of relations to include (default: 10)
        
    Returns:
        Dictionary with 'entities' and 'relations' lists
    """
    explanation = {
        'entities': [],
        'relations': []
    }
    
    # Extract entities with their types
    for node in graph.nodes():
        entity_type = graph.nodes[node].get('entity_type', 'UNKNOWN')
        explanation['entities'].append((node, entity_type))
    
    # Extract relations (edges), sorted by weight (most frequent first)
    relations = []
    seen_edges = set()  # Avoid duplicate edges (undirected graph)
    
    for source, target, data in graph.edges(data=True):
        # Create canonical edge representation (avoid duplicates)
        edge_key = tuple(sorted([source, target]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        
        relation_type = data.get('relation', 'associated_with')
        weight = data.get('weight', 1)
        
        relations.append((source, relation_type, target, weight))
    
    # Sort by weight (descending) and limit to max_relations
    relations.sort(key=lambda x: x[3], reverse=True)
    explanation['relations'] = relations[:max_relations]
    
    return explanation


def format_explanation(explanation: Dict[str, List]) -> str:
    """
    Format explanation dictionary into human-readable text.
    
    Output format:
    Entities:
    - Entity Name (Type)
    
    Relations:
    - Entity A -> associated_with -> Entity B
    
    Args:
        explanation: Dictionary from get_graph_explanation()
        
    Returns:
        Formatted string ready for display
    """
    output = []
    
    # Format entities
    output.append("Entities:")
    if explanation['entities']:
        for entity_text, entity_type in explanation['entities']:
            output.append(f"- {entity_text} ({entity_type})")
    else:
        output.append("(No entities found)")
    
    # Format relations
    output.append("\nRelations:")
    if explanation['relations']:
        for source, relation, target, weight in explanation['relations']:
            # Use consistent relation label
            output.append(f"- {source} -> associated_with -> {target}")
    else:
        output.append("(No relations found)")
    
    return "\n".join(output)


# Example usage (for testing)
if __name__ == "__main__":
    # Test entity extraction
    print("Testing entity extraction...")
    
    try:
        nlp = load_spacy_model()
        
        test_text = "Apple Inc. is a technology company founded by Steve Jobs. It is based in Cupertino, California."
        entities = extract_entities(test_text, nlp)
        
        print(f"\nExtracted entities from test text:")
        for entity, label in entities:
            print(f"  - {entity} ({label})")
            
    except Exception as e:
        print(f"Error: {e}")
