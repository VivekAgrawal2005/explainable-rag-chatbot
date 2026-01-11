# Explainable RAG Chatbot

A complete Retrieval-Augmented Generation (RAG) chatbot with explainability features, built using only free and open-source tools.

## Features

- **Document Processing**: Loads and chunks PDF documents
- **Vector Search**: Uses FAISS for fast similarity search
- **Entity Extraction**: Identifies named entities (people, organizations, locations, etc.)
- **Knowledge Graph**: Builds relationships between entities
- **Local LLM**: Uses LLaMA 3 via Ollama (completely offline)
- **Explainability**: Shows which entities and relationships were used to generate answers

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Start with: `ollama serve`
   - Download model: `ollama pull llama3`

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy language model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Place your PDF document:**
   - Put a PDF file named `docs.pdf` in the `data/` folder

## Usage

Run the chatbot:

```bash
python chatbot.py
```

The chatbot will:

1. Load and process your PDF (first time only)
2. Build a FAISS index for fast retrieval
3. Start an interactive Q&A session

Type your questions and press Enter. Type `quit` or `exit` to stop.

## Project Structure

```
explainable_rag_chatbot/
├── data/
│   └── docs.pdf          # Your PDF document (place here)
├── ingest.py             # PDF loading and chunking
├── embed_store.py        # Embedding generation and FAISS index
├── retrieve.py           # Query retrieval
├── entity_graph.py       # Named entity recognition and graph building
├── generate.py           # Answer generation using Ollama
├── chatbot.py            # Main terminal interface
└── requirements.txt      # Python dependencies
```

## How It Works

### Pipeline Overview

1. **Document Ingestion** (`ingest.py`)

   - Loads PDF from `data/docs.pdf`
   - Extracts text from all pages
   - Splits text into chunks (~300 words each)

2. **Embedding & Indexing** (`embed_store.py`)

   - Converts text chunks into vector embeddings using sentence-transformers
   - Creates a FAISS index for fast similarity search
   - Saves index for future use (avoids re-processing)

3. **Query Retrieval** (`retrieve.py`)

   - Converts user query into embedding
   - Searches FAISS index for top-k most similar chunks
   - Returns relevant context for answer generation

4. **Entity Extraction** (`entity_graph.py`)

   - Uses spaCy to extract named entities from retrieved chunks
   - Builds a knowledge graph (nodes = entities, edges = relationships)
   - Identifies how entities relate to each other

5. **Answer Generation** (`generate.py`)

   - Sends query + context to Ollama (LLaMA 3)
   - Prompts model to answer only from provided context
   - Returns answer or "Not found" if information is missing

6. **Display Results** (`chatbot.py`)
   - Shows the generated answer
   - Displays entities found and their relationships
   - Provides clear explanation of reasoning

### Explainability Features

The chatbot is "explainable" because it shows:

1. **Source Transparency**: You can see which chunks were retrieved
2. **Entity Awareness**: Lists all named entities found in relevant text
3. **Relationship Mapping**: Shows how entities connect to each other
4. **Context Limitation**: Answers are restricted to provided documents only

This helps users understand:

- **What** information was used (entities)
- **How** information connects (relationships)
- **Why** the answer is relevant (retrieved chunks)

## Example Output

```
You: What is artificial intelligence?

Answer:
Artificial intelligence (AI) is the simulation of human intelligence by machines...

Explanation:
Entities:
  - Artificial Intelligence (ORG)
  - Machine Learning (PRODUCT)
  - John McCarthy (PERSON)

Relations:
  - Artificial Intelligence -> related_to -> Machine Learning
  - John McCarthy -> related_to -> Artificial Intelligence
```

## Troubleshooting

**Ollama not found:**

- Install Ollama from https://ollama.ai
- Run `ollama serve` in a separate terminal
- Download model: `ollama pull llama3`

**spaCy model not found:**

- Run: `python -m spacy download en_core_web_sm`

**PDF not loading:**

- Ensure PDF is named `docs.pdf` and placed in `data/` folder
- Check that PDF contains readable text (not just images)

**Slow performance:**

- First run processes documents (takes time)
- Subsequent runs use cached index (faster)
- Consider using `llama3:8b` for faster inference

## License

This project uses only free and open-source libraries. All dependencies are listed in `requirements.txt`.
