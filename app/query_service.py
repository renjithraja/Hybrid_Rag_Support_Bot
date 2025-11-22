import time
from rag.retriever import Retriever
from rag.generator import LlamaGenerator


class QueryService:
    """
    High-level orchestrator that connects:
    - the retriever (vector search)
    - the generator (LLM answer synthesis)

    This class defines the RAG workflow:
        1. Embed + retrieve chunks
        2. Build a grounded prompt
        3. Generate answer using LLM
        4. Return answer + latency metrics
    """

    def __init__(self):
        # Instantiate retrieval and generation components.
        # Keeping them as long-lived objects avoids reloading models repeatedly.
        self.retriever = Retriever()
        self.generator = LlamaGenerator()

    def answer(self, question: str):
        """
        Executes the end-to-end RAG query pipeline.

        Steps:
        1. Retrieve relevant chunks based on vector + metadata search.
        2. Construct a constrained prompt so LLM does not hallucinate.
        3. Generate a grounded answer using Llama.
        4. Return answer + runtime diagnostics.
        """

        # --- 1. Retrieve supporting context ---
        t0 = time.time()
        results = self.retriever.search(question)
        retrieval_time = time.time() - t0

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        # If retrieval yields no passages, do not generate.
        if not docs:
            return {
                "answer": "I don't know.",
                "retrieval_time": retrieval_time,
                "generation_time": 0,
                "metadata": {}
            }

        # Merge retrieved passages into a single context block.
        # Keeping long context is fine — Llama handles it well.
        context = "\n\n".join(docs)

        # Strict prompt to avoid hallucinations.
        # The LLM is instructed to ONLY answer from the given context.
        prompt = f"""
Answer ONLY using the provided manual context.
If the answer is not present, reply exactly: "I don't know."

Context:
{context}

Question: {question}

Answer:
"""

        # --- 2. Generate the final answer using Llama ---
        answer, gen_time = self.generator.generate(prompt)

        # Safety check — if model returned empty output.
        if not answer.strip():
            answer = "I don't know."

        # Return full diagnostic package back to UI.
        return {
            "answer": answer.strip(),
            "retrieval_time": retrieval_time,
            "generation_time": gen_time,
            "metadata": metas
        }
