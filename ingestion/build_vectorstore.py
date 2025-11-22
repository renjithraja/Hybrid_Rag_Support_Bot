import os
import re
import pdfplumber
import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_chapters_from_page(text):
    """
    Extract section titles from a page by detecting the pattern used in the
    Dell Latitude 5400 manual:

        <section number>
        <section title>

    Example found in this PDF:
        1
        Set up your computer

    The function scans each page for this two-line pattern and returns the
    titles found. Only non-empty titles longer than a few characters are used.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chapters = []

    for i in range(len(lines) - 1):
        # First line: just a number (e.g., "1", "2", "3")
        if re.fullmatch(r"\d+", lines[i]):
            title = lines[i + 1].strip()
            if len(title) > 3:
                chapters.append(title)

    return chapters


def embed(text):
    """
    Generate a 768-dimensional embedding using Ollama's nomic-embed-text.
    This must match the embedding dimension used during retrieval.
    """
    resp = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return resp["embeddings"][0]


def main():
    pdf_path = "data/manual.pdf"
    print("Parsing PDF...")

    # Extract plain text for every page
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]

    print(f"Parsed {len(pages)} pages")

    # -----------------------
    # CHAPTER DETECTION PASS
    # -----------------------
    # We run a full scan BEFORE chunking so each page inherits its chapter.
    print("Detecting chapters...")
    page_chapters = []
    last_chapter = "UNKNOWN"

    for text in pages:
        detected = extract_chapters_from_page(text)

        # If a chapter header appears on this page, update the active chapter
        if detected:
            last_chapter = detected[0]

        # Carry chapter forward to all pages until a new one is detected
        page_chapters.append(last_chapter)

    # -----------------------
    # CHUNKING PASS
    # -----------------------
    # Chunking after chapter detection ensures every chunk gets proper metadata.
    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    metadata = []
    embeddings = []

    for page_num, text in enumerate(pages, start=1):
        chapter = page_chapters[page_num - 1]

        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadata.append({"page": page_num, "chapter": chapter})
            embeddings.append(embed(chunk))

    print("Total chunks:", len(chunks))

    # -----------------------
    # BUILD LOCAL VECTORSTORE
    # -----------------------
    os.makedirs("vectorstore/chromadb", exist_ok=True)
    client = chromadb.PersistentClient("vectorstore/chromadb")

    # Rebuild from scratch each run
    try:
        client.delete_collection("rag_manual")
    except:
        pass

    collection = client.create_collection(
        name="rag_manual",
        metadata={"hnsw:space": "cosine"}
    )

    print("Adding to Chroma...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Insert documents, embeddings, and metadata together
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata
    )

    print("DONE — Vectorstore built successfully.")


if __name__ == "__main__":
    main()
