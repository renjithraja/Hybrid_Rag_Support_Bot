import pdfplumber


class PDFParser:
    """
    PDF parser using pdfplumber.
    Handles multi-column text, accurate raw extraction,
    and robust section (chapter) detection for Dell manuals.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

        # Section headings extracted from Dell Latitude manual
        self.known_sections = [
            "SET UP YOUR COMPUTER",
            "CREATE A USB RECOVERY DRIVE FOR WINDOWS",
            "CHASSIS OVERVIEW",
            "TECHNICAL SPECIFICATIONS",
            "SOFTWARE",
            "SYSTEM SETUP",
            "GETTING HELP",
        ]

    def detect_heading(self, lines):
        """
        Detect heading by:
        - ALL CAPS short lines
        - Known section name match
        - Position (top of page)
        """
        for line in lines[:6]:
            text = line.strip()

            # ALL CAPS detection
            if text.isupper() and 3 <= len(text.split()) <= 8:
                return text

            # Keyword / known header match
            for sec in self.known_sections:
                if sec.lower() in text.lower():
                    return sec

        return None

    def extract(self):
        """
        Extract pages and metadata.
        Returns list of:
        { page: int, chapter: str, text: str }
        """
        results = []
        current_chapter = "Unknown"

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                if not text:
                    continue

                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if not lines:
                    continue

                # Detect chapter heading
                heading = self.detect_heading(lines)
                if heading:
                    current_chapter = heading

                results.append(
                    {
                        "page": page_no,
                        "chapter": current_chapter,
                        "text": text,
                    }
                )

        return results
