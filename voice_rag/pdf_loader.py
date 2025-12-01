from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from .utils import info, warn, die
from .config import CONFIG
import re

# Optional libraries
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    fitz = None
    HAS_FITZ = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK_SENT = True
except Exception:
    HAS_NLTK_SENT = False


# -----------------------------
# PDF Repair Helpers
# -----------------------------
def _repair_pdf_with_fitz(src: Path, out: Path) -> bool:
    """
    Attempt to repair a PDF using PyMuPDF.

    Returns:
        True if repair produced a viable PDF; False otherwise.
    """
    if not HAS_FITZ:
        warn("PyMuPDF (fitz) not available; cannot repair PDF.")
        return False

    try:
        doc = fitz.open(str(src))
        doc.save(str(out), garbage=4, deflate=True, incremental=False)
        doc.close()

        if out.exists() and out.stat().st_size > 1024:
            info(f"Repaired PDF written: {out}")
            return True

        warn(f"Repaired PDF is too small or missing: {out}")
        return False

    except Exception as e:
        warn(f"PDF repair failed for {src}: {e}")
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        return False


# -----------------------------
# PDF / TXT Loader + Chunker
# -----------------------------
@dataclass
class PDFTextLoader:
    """
    Load text from PDF or TXT files, split into sentences, and produce
    overlapping text chunks suitable for embedding/indexing.
    """
    pdf_paths: List[Path]
    chunk_size: int = 900
    chunk_overlap: int = 200

    # -----------------------------
    # Text Extraction
    # -----------------------------
    def extract_text(self, path: Path) -> str:
        """
        Extract text from a .txt or .pdf file.

        Returns:
            Raw extracted text. Returns an empty string on failure.
        """
        info(f"Extracting text: {path}")

        # Text file read
        if path.suffix.lower() == ".txt":
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                warn(f"Failed to read text file {path}: {e}")
                return ""

        # PDF read requires pypdf
        if PdfReader is None:
            die("pypdf is required to parse PDFs. Install with `pip install pypdf`.")

        def _try_read(p: Path) -> Optional[str]:
            """Try extracting text from a PDF via pypdf."""
            try:
                reader = PdfReader(str(p))
                if getattr(reader, "is_encrypted", False):
                    try:
                        reader.decrypt("")
                    except Exception:
                        info(f"Encrypted PDF cannot be auto-decrypted: {p}")
                        return ""

                pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception as e:
                        warn(f"Page {i} parse error in {p}: {e}")
                        pages.append("")

                return "\n\n".join(pages).strip()

            except Exception:
                raise

        # First attempt
        try:
            text = _try_read(path)
            if text and text.strip():
                return text
            if text == "":
                info(f"No text found in {path} (possible image-only PDF).")
                return ""
        except Exception as e:
            info(f"pypdf initial read failed for {path}: {e}")

        # Repair attempt
        if HAS_FITZ:
            repaired = path.with_name(f"{path.stem}._repaired{path.suffix}")
            info(f"Attempting repair: {path} → {repaired}")

            if _repair_pdf_with_fitz(path, repaired):
                try:
                    repaired_text = _try_read(repaired)
                    if repaired_text and repaired_text.strip():
                        if not CONFIG.get("save_repaired_pdf", False):
                            try:
                                repaired.unlink()
                            except Exception:
                                pass
                        return repaired_text

                    info(f"Repaired PDF had no extractable text: {repaired}")

                except Exception as e:
                    info(f"Failed to parse repaired PDF {repaired}: {e}")
            else:
                info("PDF repair failed.")
        else:
            warn("PyMuPDF not available; skipping repair step.")

        info(f"Skipping PDF after failed parse/repair: {path}")
        return ""

    # -----------------------------
    # Sentence Splitting
    # -----------------------------
    def _sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK if available; otherwise use regex.
        """
        if HAS_NLTK_SENT:
            try:
                return sent_tokenize(text)
            except Exception:
                pass

        # Regex fallback
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in parts if s.strip()]

    # -----------------------------
    # Text Chunking
    # -----------------------------
    def simple_chunker(self, text: str) -> List[str]:
        """
        Chunk text by sentence boundaries into overlapping windows.

        Returns:
            A list of chunk strings.
        """
        if not text:
            return []

        sentences = self._sentence_split(text)
        if not sentences:
            return []

        chunks = []
        cur = []
        cur_len = 0
        max_len = self.chunk_size
        overlap = self.chunk_overlap

        for s in sentences:
            slen = len(s.split())

            if cur_len + slen <= max_len:
                cur.append(s)
                cur_len += slen
                continue

            # Emit full chunk
            chunks.append(" ".join(cur).strip())

            # Prepare next chunk start with overlap
            if overlap > 0:
                tail_words = " ".join(cur).split()[-overlap:]
                cur = [" ".join(tail_words), s]
                cur_len = len(" ".join(cur).split())
            else:
                cur = [s]
                cur_len = slen

        # Final chunk
        if cur:
            chunks.append(" ".join(cur).strip())

        return chunks

    # -----------------------------
    # Load and Chunk All Files
    # -----------------------------
    def load_and_chunk(self) -> Tuple[List[str], List[dict]]:
        """
        Load all provided files, extract text, chunk, and produce metadata.

        Returns:
            (chunks, metadatas)
        """
        all_chunks = []
        metadatas = []

        for p in self.pdf_paths:
            text = self.extract_text(p)
            if not text or not text.strip():
                info(f"No text extracted from {p}; skipping.")
                continue

            chunks = self.simple_chunker(text)

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadatas.append({"source": p.name, "chunk_index": idx})

            info(f"Produced {len(chunks)} chunks from {p.name}")

        info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks, metadatas