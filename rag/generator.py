import time
import ollama


class LlamaGenerator:
    """
    Wrapper around a local Llama model served through Ollama.

    This class isolates the generation logic so the rest of the RAG pipeline
    does not need to know anything about the model backend. If you later swap
    the model (quantized version, larger model, etc.), only this file needs to
    change.
    """

    def __init__(self):
        # Using the full-precision Llama 3.1 model for maximum answer quality.
        # This name must match the model pulled into Ollama.
        self.model = "llama3.1"

    def generate(self, prompt: str):
        """
        Perform a blocking text generation call.

        The function measures generation latency explicitly because the
        assignment requires logging both retrieval time and model generation
        time. This also makes it easier to compare different model versions
        (full vs. quantized).
        """
        t0 = time.time()

        # Send the prompt to the running Ollama server.
        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )

        gen_time = time.time() - t0

        # "response" always contains a "response" field with the model output.
        return response["response"], gen_time
