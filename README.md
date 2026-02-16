# 🧠 Explainable Multi-Document RAG Chatbot with Knowledge Graphs

## 📌 Project Overview

This project implements an **Explainable Retrieval-Augmented Generation (RAG)** system that answers user questions strictly based on uploaded PDF documents.

Unlike conventional RAG systems that may hallucinate or provide opaque answers, this system emphasizes:

- **Document grounding**
- **Transparency**
- **Trustworthiness**
- **Safe refusal when evidence is insufficient**

The system supports multiple PDF documents, automatically rebuilds its retrieval index when documents change, and provides knowledge graph–based explanations at both query and document levels.

---

## ✨ Key Features

### 📂 Multi-Document Support
- Ingests multiple PDFs from the `data/` directory
- Automatically rebuilds the index when PDFs are added or removed

### 🧩 Intelligent Chunking & Embeddings
- Text chunking for improved retrieval
- Semantic embeddings using:
all-MiniLM-L6-v2 (Sentence Transformers)


### ⚡ Efficient Similarity Search
- FAISS-based vector indexing for fast retrieval

### 📄 Document-Aware Retrieval
- Prevents cross-document mixing
- Allows document-specific query scoping

### 🤖 Local Answer Generation
- Uses **LLaMA 3 via Ollama**
- Answers are strictly constrained to retrieved content

### 🛑 Safe Refusal Mechanism
- Refuses to answer if sufficient evidence is not found
- Prevents hallucinations

### 📊 Confidence Scoring
- Based on retrieval similarity scores

### 🔎 Explainability Features
- Query-level entity graphs
- Persistent document-level knowledge graphs
- Source attribution
- Confidence scores

### 💻 Lightweight Interface
- Terminal-based interaction
- Minimal setup complexity

---

## 🧠 System Architecture (High-Level)

The system consists of four major components:

---

### 1️⃣ Document Ingestion & Indexing

- PDF documents are parsed using **PyPDF2**
- Text is split into chunks
- Chunks are converted into embeddings
- Stored inside a FAISS vector index
- Index is automatically rebuilt when document changes are detected

---

### 2️⃣ Knowledge Graph Generation

- Named Entity Recognition (NER) using **spaCy**
- Persistent document-level knowledge graphs
- Query-level graphs dynamically generated from retrieved context
- Graph construction using **NetworkX**

---

### 3️⃣ Retrieval & Answer Generation

- User query → converted into embedding
- Similar chunks retrieved from FAISS
- Retrieval scoped to specific document (if mentioned)
- Answer generated using:
LLaMA 3 (via Ollama)

- Generation constrained strictly to retrieved context

---

### 4️⃣ Explainability & Output

For every query, the system provides:

- ✅ Final Answer (or safe refusal)
- 📄 Source Document
- 📊 Confidence Score
- 🕸 Entity-based Explanation Graph

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Named Entity Recognition | spaCy |
| Knowledge Graph | NetworkX |
| LLM | LLaMA 3 (via Ollama) |
| PDF Parsing | PyPDF2 |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-github-repo-link>
cd explainable_rag_chatbot
```
2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
4️⃣ Install Ollama & LLaMA 3
Download Ollama:
```bash
https://ollama.ai
```
Pull the model:
```bash
ollama pull llama3
```
▶️ How to Run the Project
Step 1
Place your PDF files inside the:
```bash
data/
```

Step 2
Start the chatbot:
```bash
python chatbot.py
```

💬 Example Usage
🔹 Ask a Question
Who are the companions of Lord Glenarvan in In Search of the Castaways?
🔹 View Document-Level Knowledge Graph
/show_graph In search of the castaways.pdf
🔹 Exit
exit

## 🔍 Explainability Design
The system ensures transparency through:

### 🕸 Query-Level Entity Graphs
Shows entities and relationships relevant to the answer

Built dynamically from retrieved chunks

### 📚 Document-Level Knowledge Graphs
Persistent entity network across entire document

Helps users understand document structure

### 📄 Source Attribution
Clearly indicates which document was used

### 📊 Confidence Scores
Reflect retrieval strength

Provide reliability estimation

This enables users to verify:

What the system answered

Why it answered that way

How confident the system is

## ⚠️ Limitations
Knowledge graphs rely on entity co-occurrence (not deep semantic relations)

Index rebuilding is full (not incremental)

Terminal-based interaction only (no graphical UI)

## 🚀 Future Work
- Incremental indexing
- Advanced relation extraction
- Lightweight graph visualisation
- Web-based interface
- Improved semantic graph construction

## 📄 License
This project was developed as part of a hackathon submission for academic and educational purposes.

## 🏁 Final Note
This project demonstrates that trustworthy and explainable RAG systems can be built using:
- Simple architectures
- Efficient vector search
- Transparent entity reasoning
- Strict document grounding

Without relying on:
- Complex reasoning pipelines
- External knowledge bases
- Black-box hallucination-prone systems
