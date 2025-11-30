"""Simple CLI entrypoints for building/launching the index and running the UI/interactive prompt."""

import shutil
import json
from pathlib import Path
from typing import List, Optional, Set

from voice_rag.pdf_loader import PDFTextLoader
from voice_rag.embeddings import LocalEmbeddingIndex
from voice_rag.ollama import OllamaLocal
from voice_rag.stt import LocalSTT
from voice_rag.tts import LocalTTS
from voice_rag.agent import LocalRAGAgent
from voice_rag.config import CONFIG
from voice_rag.utils import die, info, warn

PROCESSED_TRACK_FILENAME = "processed_files.json"

# -----------------------------
# Internal helpers
# -----------------------------
def _load_processed_files(chroma_dir: Path) -> Set[str]:
    track_path = chroma_dir / PROCESSED_TRACK_FILENAME
    if not track_path.exists():
        return set()
    try:
        data = json.loads(track_path.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    except Exception as e:
        warn(f"Failed to read processed list: {e}")
        return set()


def _save_processed_files(chroma_dir: Path, processed: Set[str]):
    track_path = chroma_dir / PROCESSED_TRACK_FILENAME
    try:
        track_path.write_text(json.dumps({"processed": sorted(list(processed))}, ensure_ascii=False),
                              encoding="utf-8")
    except Exception as e:
        warn(f"Could not write processed list: {e}")


def _list_files_in_docs(docs_dir: Path) -> List[Path]:
    exts = {".pdf", ".txt"}
    return [p for p in sorted(docs_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _deterministic_ids_for_chunks(metas: List[dict]) -> List[str]:
    """Generate deterministic IDs based on source file and chunk index."""
    return [f"{m['source']}_{m['chunk_index']}" for m in metas]


# -----------------------------
# Index building
# -----------------------------
def setup_index_from_docs(
        docs_dir: str = CONFIG["docs_dir"],
        reindex: bool = False,
        chunk_size: int = 900,
        chunk_overlap: int = 200
) -> LocalRAGAgent:
    docs_path = Path(docs_dir)
    chroma_path = Path(CONFIG["chroma_dir"])

    if not docs_path.exists():
        die(f"Docs directory not found: {docs_path}")

    emb_index = LocalEmbeddingIndex()

    if reindex and chroma_path.exists():
        info("Reindex requested: clearing chroma directory.")
        try:
            shutil.rmtree(chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warn(f"Could not clear chroma dir: {e}")

    chroma_path.mkdir(parents=True, exist_ok=True)
    emb_index.init_chroma()

    processed = _load_processed_files(chroma_path)
    candidate_files = _list_files_in_docs(docs_path)
    new_files = [p for p in candidate_files if p.name not in processed]
    info(f"Found {len(candidate_files)} files, {len(new_files)} new to process.")

    if new_files:
        loader = PDFTextLoader(pdf_paths=new_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks, metas = loader.load_and_chunk()
        if chunks:
            ids = _deterministic_ids_for_chunks(metas)
            emb_index.add_texts(chunks, metadatas=metas, ids=ids)
        for p in new_files:
            processed.add(p.name)
        _save_processed_files(chroma_path, processed)
    else:
        info("No new documents to index.")

    # Initialize agent components
    ollama = OllamaLocal()
    stt = LocalSTT()
    tts = LocalTTS()
    agent = LocalRAGAgent(ollama=ollama, stt=stt, tts=tts, index=emb_index)
    return agent


def setup_index(pdf_filepaths: Optional[List[str]] = None, reindex: bool = False) -> LocalRAGAgent:
    """
    Setup index from a list of PDF/TXT files, or fallback to docs_dir.
    """
    if pdf_filepaths:
        pdf_paths = [Path(p) for p in pdf_filepaths]
        for p in pdf_paths:
            if not p.exists():
                die(f"File not found: {p}")

        loader = PDFTextLoader(pdf_paths=pdf_paths)
        chunks, metas = loader.load_and_chunk()

        emb_index = LocalEmbeddingIndex()
        if reindex and Path(CONFIG["chroma_dir"]).exists():
            try:
                shutil.rmtree(CONFIG["chroma_dir"])
            except Exception as e:
                warn(f"Could not delete DB dir: {e}")

        emb_index.init_chroma()

        if chunks:
            ids = _deterministic_ids_for_chunks(metas)
            emb_index.add_texts(chunks, metadatas=metas, ids=ids)

        ollama = OllamaLocal()
        stt = LocalSTT()
        tts = LocalTTS()
        return LocalRAGAgent(ollama=ollama, stt=stt, tts=tts, index=emb_index)

    else:
        return setup_index_from_docs(reindex=reindex)
