import chromadb
import ollama


class Retriever:
    """
    Handles all retrieval logic for the RAG system.

    This component is responsible for:
      - Converting user queries into embeddings
      - Optionally routing queries to the correct document chapter using metadata
      - Falling back to a full-document vector search when needed

    Keeping retrieval isolated in this file makes the system modular:
    the generator, UI, and ingestion pipeline never need to know about
    embedding details or vector DB specifics.
    """

    def __init__(self, persist_dir="vectorstore/chromadb"):
        # Load the Chroma persistent database created during ingestion.
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("rag_manual")

        # Embedding model must match what was used to build the vectorstore.
        self.embedding_model = "nomic-embed-text"

    def embed(self, text: str):
        """
        Convert a piece of text into a vector using Ollama's embedding endpoint.
        Chroma expects embedding vectors of length 768 when using this model.
        """
        resp = ollama.embed(
            model=self.embedding_model,
            input=text
        )
        return resp["embeddings"][0]

    def detect_chapter(self, question: str):
        """
        Lightweight heuristic to route queries toward the correct chapter.

        This improves retrieval accuracy by narrowing the search space
        when users ask section-specific questions.
        The mapping keys use lowercase keywords extracted from the question,
        and map them to the chapter titles actually found inside the PDF.
        """
        q = question.lower()

        mapping = {
            "set up": "Set up your computer",
            "first time": "Set up your computer",

            "recovery": "Create a USB recovery drive for Windows",
            "usb": "Create a USB recovery drive for Windows",

            "ports": "Chassis overview",
            "keyboard": "Chassis overview",
            "touchpad": "Chassis overview",

            "system setup": "System setup",
            "bios": "System setup",
        }

        # Return the first matching mapped chapter
        for keyword, chapter in mapping.items():
            if keyword in q:
                return chapter

        # If nothing matches, fallback to full vector search
        return None

    def search(self, question: str, n_results=4):
        """
        Perform hybrid retrieval:
           1. Try metadata-filtered vector search (chapter-specific)
           2. If no match or filter error → revert to global search

        This is the core of the "Hybrid RAG" requirement:
        reducing hallucination and improving accuracy by scoping retrieval.
        """
        q_emb = self.embed(question)
        chapter = self.detect_chapter(question)

        # 1. Metadata-aware search
        if chapter:
            try:
                filtered = self.collection.query(
                    query_embeddings=[q_emb],
                    where={"chapter": {"$eq": chapter}},   # Chroma only supports $eq
                    n_results=n_results
                )

                # If we successfully retrieved chapter-matching context
                if filtered["documents"][0]:
                    return filtered

            except Exception as e:
                # If filtering fails, just proceed to fallback search
                print("Metadata filter failed:", e)

        # 2. Fallback: search entire document
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results
        )
