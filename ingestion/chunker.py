from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """
    Responsible for converting parsed PDF pages into vector-friendly text chunks.

    Why this class exists:
    ----------------------
    - Raw PDF text usually contains multiple columns, spacing issues,
      and sections that do not align well with vector embedding boundaries.
    - If we chunk too aggressively, the LLM loses context.
    - If we chunk too large, retrieval becomes imprecise.

    This class centralizes the chunking strategy so it can be tuned in one place
    without touching ingestion and retrieval logic.

    Current strategy:
      - Larger chunk size (1200 chars) helps preserve multi-column manual content.
      - Overlap (200 chars) ensures sentences are not cut abruptly.
      - Custom separators make the splitter prefer paragraph and line boundaries.
    """

    def __init__(self):
        # Configure the chunking behavior.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". "],
        )

    def create_chunks(self, parsed_pages):
        """
        Convert a list of parsed PDF pages into structured chunks.

        Arguments:
            parsed_pages: List of dicts in the format:
                {
                    "page": <page_number>,
                    "chapter": <chapter_name>,
                    "text": <raw_extracted_text>
                }

        Returns:
            A flat list of chunk dictionaries structured as:
                {
                    "text": <chunk_text>,
                    "page": <page_number>,
                    "chapter": <chapter_name>
                }

        Each chunk is already enriched with metadata so ingestion can directly
        embed and store it in Chroma without additional processing.
        """
        chunks = []

        for page in parsed_pages:
            text = page["text"]
            split_chunks = self.splitter.split_text(text)

            for chunk in split_chunks:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"],
                        "chapter": page["chapter"],
                    }
                )

        return chunks
