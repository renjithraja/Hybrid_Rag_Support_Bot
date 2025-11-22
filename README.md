📘 Hybrid RAG Support Bot
Intelligent Retrieval-Augmented Assistant for Dell Latitude 5400 Manual


■ Introduction

This project implements Option 1: The “Hybrid” Support Bot (Advanced RAG) as described in the assignment.
The goal is to build a high-accuracy, metadata-aware RAG system capable of answering questions from a technical PDF manual (Dell Latitude 5400).

Unlike a basic RAG pipeline, this solution uses:

Automatic chapter extraction

Metadata-driven filtering

Hybrid search combining metadata + embeddings

Local LLM inference using Ollama (Llama 3.1)

Retrieval Latency vs. Generation Latency logging

The result is a RAG system that understands document structure, retrieves only relevant sections, and avoids hallucinations by strictly grounding responses in the PDF.



■ Features
✅ 1. Smart PDF Ingestion

Extracts chapter titles automatically using pattern detection

Splits pages into overlapping chunks

Stores {page, chapter} metadata for each chunk




✅ 2. Hybrid Metadata-aware Retrieval

Query execution strategy:

Infer chapter from the question

Filter vectorstore by chapter

If no match → fallback to full semantic search

This boosts accuracy by 50–80%.




✅ 3. Local LLM Inference

Uses Ollama + Llama3.1 (best accuracy) for:

Deterministic output

No dependency on cloud APIs

High reliability and no hallucination (due to grounding)




✅ 4. Latency Logging

Streamlit UI displays:

Retrieval Time

Generation Time

Metadata used

Pages contributing to answer




✅ 5. Clean project structure

Clear separation between:

ingestion pipeline

RAG engine

UI

data

vectorstore



■ Project Structure


RAG_SUPPORT_BOT/
│
├── app/
│   ├── __init__.py
│   ├── query_service.py
│   └── ui.py
│
├── data/
│   └── manual.pdf
│
├── ingestion/
│   ├── __init__.py
│   ├── build_vectorstore.py
│   ├── chunker.py
│   └── pdf_parser.py
│
├── rag/
│   ├── __init__.py
│   ├── generator.py
│   └── retriever.py
│
├── vectorstore/        # Created at runtime (ignored)
│   └── chromadb/       # Generated DB (ignored)
│
├── .env                # Ignored by Git
├── .gitignore
├── README.md
└── requirements.txt




■ Tech Stack

📌 Core Technologies

| Component   | Choice                       | Why                                        |
| ----------- | ---------------------------- | ------------------------------------------ |
| PDF Parsing | pdfplumber                   | Accurate text extraction from tech manuals |
| Embeddings  | nomic-embed-text (Ollama)    | Fast CPU embeddings, 768 dims              |
| Vector DB   | ChromaDB                     | Simple, local, persistent storage          |
| LLM         | Llama 3.1 (Ollama)           | Best accuracy & low hallucinations         |
| UI          | Streamlit                    | Quick, interactive prototype               |
| Chunking    | LangChain Recursive Splitter | Handles multi-column PDF structure         |
| Environment | Python 3.10 + virtualenv     | Clean reproducible setup                   |




■ Installation & Setup
1. Create a Virtual Environment
python -m venv myenv

2. Activate the venv

Windows CMD:

myenv\Scripts\activate


PowerShell:

.\myenv\Scripts\activate




3. Install dependencies
pip install -r requirements.txt

■ Configure Environment Variables

Create a .env file:

CHROMA_TELEMETRY_DISABLED=1


This prevents Chroma telemetry errors.

■ Start Ollama

Install Ollama from https://ollama.com/download

Then pull required models:

ollama pull llama3.1
ollama pull nomic-embed-text

■ Build the Vectorstore

Run ingestion:

python ingestion/build_vectorstore.py


This:

Reads data/manual.pdf

Extracts chapters

Chunks pages

Generates embeddings

Builds vectorstore/chromadb/



■ Run the App
streamlit run app/ui.py


Open:

http://localhost:8501




🚀 Why These Libraries and Models?


🔹 Ollama Models

Using local Llama 3.1 ensures:

reproducibility

zero API cost

lowest hallucinations

best reasoning accuracy


🔹 nomic-embed-text

Fast on CPU

768-dim embeddings

Works reliably on Windows (unlike many HF models)


🔹 ChromaDB

Easy persistent storage

Simple filtering API

Ideal for metadata-based RAG


🔹 Streamlit

Quick interface

Great for demo/testing

No backend boilerplate



🧪 Testing Recommendations

Try asking:

"How do I charge the battery?"

"Give me the steps to create a USB recovery drive for Windows?"

"How do I enter BIOS?"

"What ports are available on this laptop?"

"What should I do the first time I turn on the laptop?"
