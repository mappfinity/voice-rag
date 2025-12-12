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

    Uses hierarchical text splitting with configurable chunk sizes and overlap.
    """
    pdf_paths: List[Path]
    chunk_size: int = None  # Will use CONFIG default if None
    chunk_overlap: int = None  # Will use CONFIG default if None
    min_chunk_size: int = None  # Will use CONFIG default if None
    separators: List[str] = None  # Will use CONFIG default if None

    def __post_init__(self):
        """Initialize chunking parameters from config if not provided."""
        if self.chunk_size is None:
            self.chunk_size = CONFIG.get("chunk_size", 512)
        if self.chunk_overlap is None:
            self.chunk_overlap = CONFIG.get("chunk_overlap", 128)
        if self.min_chunk_size is None:
            self.min_chunk_size = CONFIG.get("min_chunk_size", 100)
        if self.separators is None:
            self.separators = CONFIG.get("separators", ["\n\n", "\n", ". ", " ", ""])

        # Validate parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        if self.min_chunk_size > self.chunk_size:
            warn(f"min_chunk_size ({self.min_chunk_size}) > chunk_size ({self.chunk_size}), "
                 f"adjusting min_chunk_size to {self.chunk_size // 2}")
            self.min_chunk_size = self.chunk_size // 2

        info(f"PDFTextLoader initialized: chunk_size={self.chunk_size}, "
             f"overlap={self.chunk_overlap}, min_size={self.min_chunk_size}")

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
    # Hierarchical Text Splitting
    # -----------------------------
    def _split_text_hierarchical(self, text: str) -> List[str]:
        """
        Split text using hierarchical separators (paragraphs, then sentences, etc.).

        This approach tries to split on natural boundaries in order of preference:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Sentence boundaries (". ")
        4. Spaces
        5. Character level (last resort)

        Returns:
            List of text segments
        """
        if not text or not text.strip():
            return []

        segments = [text]

        # Apply each separator hierarchically
        for separator in self.separators:
            if not separator:  # Empty separator means character-level split
                continue

            new_segments = []
            for segment in segments:
                if len(segment.split()) <= self.chunk_size:
                    # Segment is already small enough
                    new_segments.append(segment)
                else:
                    # Split on this separator
                    parts = segment.split(separator)
                    for i, part in enumerate(parts):
                        if part.strip():
                            # Re-add separator except for last part
                            if i < len(parts) - 1 and separator:
                                new_segments.append(part + separator)
                            else:
                                new_segments.append(part)

            segments = new_segments

        # Filter out empty segments and strip whitespace
        return [s.strip() for s in segments if s.strip()]

    # -----------------------------
    # Sentence Splitting (fallback)
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
    # Text Chunking with Overlap
    # -----------------------------
    def simple_chunker(self, text: str) -> List[str]:
        """
        Chunk text using hierarchical splitting with overlapping windows.

        Process:
        1. Split text hierarchically (paragraphs → sentences → words)
        2. Combine segments into chunks of target size
        3. Add overlap between chunks for context continuity
        4. Filter chunks below minimum size

        Returns:
            A list of chunk strings.
        """
        if not text or not text.strip():
            return []

        # First, try hierarchical splitting
        segments = self._split_text_hierarchical(text)

        # If hierarchical splitting didn't work well, fall back to sentence splitting
        if not segments or (len(segments) == 1 and len(segments[0].split()) > self.chunk_size * 2):
            sentences = self._sentence_split(text)
            if sentences:
                segments = sentences

        if not segments:
            return []

        chunks = []
        cur = []
        cur_len = 0  # Track length in words

        for segment in segments:
            seg_len = len(segment.split())

            # If segment alone exceeds chunk_size, split it further
            if seg_len > self.chunk_size:
                # If we have accumulated content, emit it first
                if cur:
                    chunk_text = " ".join(cur).strip()
                    if len(chunk_text.split()) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    cur = []
                    cur_len = 0

                # Split long segment by words
                words = segment.split()
                for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_words).strip()
                    if len(chunk_words) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                continue

            # Check if adding this segment would exceed chunk_size
            if cur_len + seg_len <= self.chunk_size:
                cur.append(segment)
                cur_len += seg_len
            else:
                # Emit current chunk
                chunk_text = " ".join(cur).strip()
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and cur:
                    # Take last N words for overlap
                    overlap_text = " ".join(cur)
                    overlap_words = overlap_text.split()[-self.chunk_overlap:]
                    cur = [" ".join(overlap_words), segment]
                    cur_len = len(" ".join(cur).split())
                else:
                    cur = [segment]
                    cur_len = seg_len

        # Emit final chunk
        if cur:
            chunk_text = " ".join(cur).strip()
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append(chunk_text)

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

            # Log chunk statistics
            if chunks:
                avg_chunk_size = sum(len(c.split()) for c in chunks) / len(chunks)
                info(f"Produced {len(chunks)} chunks from {p.name} "
                     f"(avg size: {avg_chunk_size:.0f} words)")
            else:
                warn(f"No valid chunks produced from {p.name}")

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = {
                    "source": p.name,
                    "chunk_index": idx,
                    "chunk_size": len(chunk.split()),
                    "total_chunks": len(chunks)
                }
                metadatas.append(meta)

        info(f"Total chunks produced: {len(all_chunks)}")

        if all_chunks:
            avg_size = sum(len(c.split()) for c in all_chunks) / len(all_chunks)
            min_size = min(len(c.split()) for c in all_chunks)
            max_size = max(len(c.split()) for c in all_chunks)
            info(f"Chunk size stats: avg={avg_size:.0f}, min={min_size}, max={max_size} words")

        return all_chunks, metadatas