"""
CLI entrypoints for building a local RAG index and running interactive chat.

Features:
- PDF/TXT loading + chunking
- Incremental or full reindexing
- Chroma-based embedding store
- Local LLM + STT + TTS agent initialization
- Hotkey-driven multiline and audio recording chat
"""

from typing import List, Optional, Set
from pathlib import Path
import json
import shutil

from voice_rag.hotkeys import HotkeyManager
from voice_rag.history import ChatHistory
from voice_rag.utils import die, info, warn
from voice_rag.config import CONFIG
from voice_rag.agent import LocalRAGAgent
from voice_rag.tts import LocalTTS
from voice_rag.stt import LocalSTT
from voice_rag.ollama import OllamaLocal
from voice_rag.embeddings import LocalEmbeddingIndex
from voice_rag.pdf_loader import PDFTextLoader

PROCESSED_TRACK_FILENAME = "processed_files.json"

# Global history + hotkeys
history = ChatHistory()
hotkeys = HotkeyManager()


# ---------------------------------------------------------------------------
# Multiline helper
# ---------------------------------------------------------------------------
def read_multiline() -> str:
    """Read multiline input until a single '.' line."""
    info("Multiline mode. End with '.'")
    lines = []
    while True:
        l = input("> ")
        if l.strip() == ".":
            return "\n".join(lines)
        lines.append(l)


# ---------------------------------------------------------------------------
# Interactive Chat
# ---------------------------------------------------------------------------
def interactive_chat(agent: LocalRAGAgent):
    """Interactive console chat with hotkeys and optional TTS."""
    speaking = True

    def hotkey_multiline():
        text = read_multiline()
        handle_input(text)

    def hotkey_record():
        print("\n[Hotkey] Recording…")
        ans, rag_sources, voice_path = agent.record_and_answer(speak=speaking)
        print(f"Agent> {ans}")
        history.add_interaction(
            user_prompt="[Hotkey Record]",
            bot_response=ans,
            rag_sources=rag_sources,
            voice_output_path=voice_path,
        )

    # Register hotkeys
    hotkeys.bind("ctrl+j", hotkey_multiline)
    hotkeys.bind("ctrl+r", hotkey_record)
    hotkeys.start()

    info("Interactive chat. Commands: record, file <path>, speakon, speakoff, :ml, exit.")
    info("Hotkeys: Ctrl+J multiline, Ctrl+R record audio")

    def handle_input(user: str):
        nonlocal speaking
        ans = None
        rag_sources = []
        voice_path = None

        parts = user.split()
        cmd = parts[0].lower() if parts else ""

        if cmd in ("exit", "quit"):
            exit()

        if cmd == "speakon":
            speaking = True
            info("TTS enabled.")
            return

        if cmd == "speakoff":
            speaking = False
            info("TTS disabled.")
            return

        if cmd == "record":
            ans, rag_sources, voice_path = agent.record_and_answer(speak=speaking)

        elif cmd == "file" and len(parts) > 1:
            path = user.split(" ", 1)[1].strip()
            ans, rag_sources, voice_path = agent.answer_voice_file(path, speak=speaking)

        else:
            ans, rag_sources, voice_path = agent.answer_text(user, speak=speaking)

        print(f"Agent> {ans}")
        history.add_interaction(
            user_prompt=user,
            bot_response=ans,
            rag_sources=rag_sources,
            voice_output_path=voice_path,
        )

    # REPL
    while True:
        try:
            user = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user:
            continue
        if user == ":ml":
            user = read_multiline()

        handle_input(user)


# ---------------------------------------------------------------------------
# Helpers: processed file tracking
# ---------------------------------------------------------------------------
def _load_processed_files(chroma_dir: Path) -> Set[str]:
    """Return the set of already-indexed files."""
    path = chroma_dir / PROCESSED_TRACK_FILENAME
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    except Exception as e:
        warn(f"Failed to read processed list: {e}")
        return set()


def _save_processed_files(chroma_dir: Path, processed: Set[str]):
    """Persist the processed file list."""
    path = chroma_dir / PROCESSED_TRACK_FILENAME
    try:
        path.write_text(
            json.dumps({"processed": sorted(list(processed))}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        warn(f"Could not write processed list: {e}")


def _list_files_in_docs(docs_dir: Path) -> List[Path]:
    """Return all PDF/TXT files in the docs directory."""
    exts = {".pdf", ".txt"}
    return [p for p in sorted(docs_dir.iterdir()) if p.suffix.lower() in exts]


def _deterministic_ids_for_chunks(metas: List[dict]) -> List[str]:
    """Stable IDs based on <source>_<chunk_index>."""
    return [f"{m['source']}_{m['chunk_index']}" for m in metas]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def setup_index_from_docs(
        docs_dir: str = CONFIG["docs_dir"],
        reindex: bool = False,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
) -> LocalRAGAgent:
    """
    Build or update the Chroma index from a docs directory.

    - Incremental indexing via processed_files.json
    - reindex=True clears and rebuilds the embedding store
    """
    docs_path = Path(docs_dir)
    chroma_path = Path(CONFIG["chroma_dir"])

    if not docs_path.exists():
        die(f"Docs directory not found: {docs_path}")

    emb_index = LocalEmbeddingIndex()

    if reindex and chroma_path.exists():
        info("Reindex requested: resetting chroma directory.")
        try:
            shutil.rmtree(chroma_path)
        except Exception as e:
            warn(f"Could not clear chroma dir: {e}")

    chroma_path.mkdir(parents=True, exist_ok=True)
    emb_index.init_chroma()

    processed = _load_processed_files(chroma_path)
    candidates = _list_files_in_docs(docs_path)
    new_files = [p for p in candidates if p.name not in processed]

    info(f"{len(candidates)} files found, {len(new_files)} new.")

    if new_files:
        loader = PDFTextLoader(
            pdf_paths=new_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks, metas = loader.load_and_chunk()
        if chunks:
            ids = _deterministic_ids_for_chunks(metas)
            emb_index.add_texts(chunks, metadatas=metas, ids=ids)

        for p in new_files:
            processed.add(p.name)
        _save_processed_files(chroma_path, processed)
    else:
        info("No new documents to index.")

    ollama = OllamaLocal()
    stt = LocalSTT()
    tts = LocalTTS()

    return LocalRAGAgent(ollama=ollama, stt=stt, tts=tts, index=emb_index)


def setup_index(
        pdf_filepaths: Optional[List[str]] = None,
        reindex: bool = False,
) -> LocalRAGAgent:
    """
    Build an index from the provided files or the default docs directory.

    This bypasses incremental tracking when specific files are passed.
    """
    if pdf_filepaths:
        pdf_paths = [Path(p) for p in pdf_filepaths]
        for p in pdf_paths:
            if not p.exists():
                die(f"File not found: {p}")

        loader = PDFTextLoader(pdf_paths=pdf_paths)
        chunks, metas = loader.load_and_chunk()

        emb_index = LocalEmbeddingIndex()
        chroma_dir = Path(CONFIG["chroma_dir"])
        if reindex and chroma_dir.exists():
            try:
                shutil.rmtree(chroma_dir)
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

    return setup_index_from_docs(reindex=reindex)
