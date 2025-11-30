from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from .utils import info, warn, die
from .config import CONFIG
import re, os

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
# PDF repair + Text extraction
# -----------------------------
def _repair_pdf_with_fitz(src: Path, out: Path) -> bool:
    """Attempt to repair a PDF using PyMuPDF (fitz)."""
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
        warn(f"Repaired PDF is suspiciously small or missing: {out}")
        return False
    except Exception as e:
        warn(f"fitz repair failed for {src}: {e}")
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        return False


@dataclass
class PDFTextLoader:
    pdf_paths: List[Path]
    chunk_size: int = 900
    chunk_overlap: int = 200

    # -----------------------------
    # Extract text from PDF or TXT
    # -----------------------------
    def extract_text(self, path: Path) -> str:
        info(f"Extracting text: {path}")

        if path.suffix.lower() == ".txt":
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                warn(f"Failed to read text file {path}: {e}")
                return ""

        if PdfReader is None:
            die("pypdf is required to parse PDFs. Install with `pip install pypdf`.")

        def _try_read(p: Path) -> Optional[str]:
            """Attempt reading PDF text with pypdf."""
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
                        warn(f"Page {i} parse error for {p}: {e}")
                        pages.append("")
                return "\n\n".join(pages).strip()
            except Exception as e:
                raise

        # Attempt direct read
        try:
            t = _try_read(path)
            if t and t.strip():
                return t
            if t == "":
                info(f"No text found in {path} (maybe image-only).")
                return ""
        except Exception as e:
            info(f"pypdf initial read failed for {path}: {e}")

        # Attempt repair
        if HAS_FITZ:
            repaired = path.with_name(f"{path.stem}._repaired{path.suffix}")
            info(f"Attempting repair: {path} -> {repaired}")
            if _repair_pdf_with_fitz(path, repaired):
                try:
                    t2 = _try_read(repaired)
                    if t2 and t2.strip():
                        if not CONFIG.get("save_repaired_pdf", False):
                            try:
                                repaired.unlink()
                            except Exception:
                                pass
                        return t2
                    info(f"Repaired PDF had no text: {repaired}")
                except Exception as e:
                    info(f"pypdf read failed on repaired PDF {repaired}: {e}")
            else:
                info("fitz failed to repair PDF.")
        else:
            warn("fitz not available; skipping repair.")

        info(f"Skipping PDF after failed parse/repair: {path}")
        return ""

    # -----------------------------
    # Sentence splitting
    # -----------------------------
    def _sentence_split(self, text: str) -> List[str]:
        """Split text into sentences using NLTK if available, otherwise regex."""
        try:
            if HAS_NLTK_SENT:
                return sent_tokenize(text)
        except Exception:
            pass
        # Fallback regex
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if s.strip()]

    # -----------------------------
    # Chunk text into overlapping segments
    # -----------------------------
    def simple_chunker(self, text: str) -> List[str]:
        if not text:
            return []

        sentences = self._sentence_split(text)
        if not sentences:
            return []

        chunks = []
        cur = []
        cur_len = 0
        cs = self.chunk_size
        ov = self.chunk_overlap

        for s in sentences:
            slen = len(s.split())
            if cur_len + slen <= cs:
                cur.append(s)
                cur_len += slen
            else:
                chunks.append(" ".join(cur).strip())
                # Start new chunk with overlap
                if ov > 0:
                    tail_words = " ".join(cur).split()[-ov:]
                    cur = [" ".join(tail_words), s]
                    cur_len = len(" ".join(cur).split())
                else:
                    cur = [s]
                    cur_len = slen

        if cur:
            chunks.append(" ".join(cur).strip())
        return chunks

    # -----------------------------
    # Load all PDFs/TXT and chunk
    # -----------------------------
    def load_and_chunk(self) -> Tuple[List[str], List[dict]]:
        all_chunks = []
        metadatas = []

        for p in self.pdf_paths:
            txt = self.extract_text(p)
            if not txt or not txt.strip():
                info(f"No text for {p}; skipping")
                continue

            chunks = self.simple_chunker(txt)
            for idx, c in enumerate(chunks):
                all_chunks.append(c)
                metadatas.append({"source": p.name, "chunk_index": idx})

            info(f"Produced {len(chunks)} chunks from {p.name}")

        info(f"Total chunks: {len(all_chunks)}")
        return all_chunks, metadatas
