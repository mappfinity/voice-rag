from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict, Optional
from .reranker import ReRanker
from .utils import info, warn
from .config import CONFIG

import uuid
import os

from voice_rag.embeddings import LocalEmbeddingIndex
from voice_rag.agent_helpers import save_numpy_to_wav, record_to_numpy


# ============================================================================
# Local RAG Agent
# ============================================================================
@dataclass
class LocalRAGAgent:
    """
    A fully local Retrieval-Augmented Generation agent.

    Components:
    - Local LLM (Ollama)
    - Local STT
    - Local TTS
    - Local embedding index
    - Optional reranker

    Designed for offline reliability with simple, predictable behavior.
    """
    ollama: Any
    stt: Any
    tts: Any
    index: LocalEmbeddingIndex

    history: List[Tuple[str, str]] = field(default_factory=list)

    system_prompt: str = (
        "You are a highly reliable, detail-oriented assistant.\n\n"
        "Always begin by analyzing the provided CONTEXTS. Treat them as the source "
        "of truth and avoid speculation.\n\n"
        "Summarize and synthesize relevant content in your own words. Use internal "
        "knowledge only for clarification.\n\n"
        "Avoid claims about product features unless explicitly supported.\n\n"
        "Produce clean, structured output suitable for text-to-speech.\n"
    )

    reranker: Optional[ReRanker] = None

    # ----------------------------------------------------------------------
    # Initialize reranker if not supplied
    # ----------------------------------------------------------------------
    def __post_init__(self):
        if self.reranker is None:
            self.reranker = ReRanker(
                embed_fn=self.index.embed_texts,
                ollama=self.ollama
            )

    # ----------------------------------------------------------------------
    # Full prompt builder (debug / verbose)
    # ----------------------------------------------------------------------
    def build_prompt(self, user_query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Build a verbose prompt that includes full context chunks.
        Used mainly for debugging or manual analysis.
        """
        parts = []
        for i, d in enumerate(docs):
            text = d.get("text", "")
            meta = d.get("meta", {}) or {}

            header = (
                f"<<CONTEXT_{i} source={meta.get('source', '?')} "
                f"chunk={meta.get('chunk_index', '?')}>>\n"
            )
            parts.append(header + text)

        ctx = "\n\n".join(parts) if parts else "<<NO_CONTEXT>>"

        return (
            f"{self.system_prompt}\n"
            f"CONTEXTS:\n{ctx}\n\n"
            f"User: {user_query}\n"
            f"Answer:"
        )

    # ----------------------------------------------------------------------
    # Minimal cleanup for safe TTS consumption
    # ----------------------------------------------------------------------
    def clean_for_tts(self, text: str, *args, **kwargs) -> str:
        """
        Strip potentially problematic characters for TTS engines.
        Conservative filtering only: markdown markers, emojis, non-ASCII, etc.
        """
        import re

        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r"[*#`]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        return text.strip()

    # ----------------------------------------------------------------------
    # Compress long contexts to fit within LLM input limits
    # ----------------------------------------------------------------------
    def _compress_contexts_for_llm(
            self,
            docs: List[Dict[str, Any]],
            max_chars_per_chunk: Optional[int] = None
    ) -> str:
        """
        Create truncated previews for each retrieved chunk.

        Helps maintain prompt size stability by showing only head/tail portions
        of long passages.
        """
        if max_chars_per_chunk is None:
            max_chars_per_chunk = CONFIG.get("context_chunk_preview_chars", 1024)

        parts = []

        for i, d in enumerate(docs):
            txt = d.get("text", "") or ""
            meta = d.get("meta", {}) or {}

            # Head+tail truncation if needed
            if len(txt) > max_chars_per_chunk:
                head_len = int(max_chars_per_chunk * 0.6)
                tail_len = int(max_chars_per_chunk * 0.4)

                head = txt[:head_len]
                tail = txt[-tail_len:]
                preview = f"{head}\n... (truncated) ...\n{tail}"
            else:
                preview = txt

            header = (
                f"<<CONTEXT source={meta.get('source','?')} "
                f"chunk={meta.get('chunk_index','?')}>>\n"
            )
            parts.append(header + preview)

        return "\n\n".join(parts) if parts else "<<NO_CONTEXT>>"

    # ----------------------------------------------------------------------
    # Text-based RAG answer
    # ----------------------------------------------------------------------
    def answer_text(
            self,
            user_query: str,
            top_k: int = 4,
            speak: bool = False
    ) -> str:
        """
        Core RAG pipeline:
        1. Retrieve candidates
        2. Rerank
        3. Compress context
        4. Query LLM
        5. Optional TTS
        """
        if not user_query or not user_query.strip():
            return ""

        user_query = user_query.strip()
        info(f"Query: {user_query[:120]}")

        # Retrieval bounds
        top_k = max(1, min(top_k, CONFIG.get("max_retrieval_topk", 8)))

        # Initial retrieval
        try:
            retrieved = self.index.query(
                user_query,
                top_k=CONFIG.get("initial_retrieval_k", 16)
            )
        except Exception as e:
            warn(f"Initial retrieval failed: {e}")
            retrieved = []

        # Format for reranker
        candidates = []
        for i, item in enumerate(retrieved):
            try:
                doc, meta, dist = item
            except Exception:
                warn(f"Malformed retrieval item at index {i}: {item}")
                continue

            candidates.append({
                "id": meta.get(
                    "id",
                    f"{meta.get('source','?')}_{meta.get('chunk_index', i)}"
                ),
                "text": doc,
                "meta": meta,
                "embedding": None
            })

        # Reranking (MMR by default)
        try:
            reranked = self.reranker.rerank(
                user_query,
                candidates,
                top_k=top_k,
                use_mmr=True
            )
        except Exception as e:
            warn(f"Reranking failed: {e}")
            reranked = candidates[:top_k]

        # Build compressed prompt
        ctx_for_llm = self._compress_contexts_for_llm(
            reranked,
            max_chars_per_chunk=CONFIG.get("context_chunk_preview_chars", 1024)
        )

        prompt = (
            f"{self.system_prompt}\n\n"
            f"CONTEXTS:\n{ctx_for_llm}\n\n"
            f"User: {user_query}\n"
            f"Answer:"
        )

        # LLM generation
        try:
            resp = self.ollama.generate(
                prompt,
                max_tokens=CONFIG.get("llm_max_tokens", 1200)
            ).strip()
        except Exception as e:
            warn(f"LLM generation failed: {e}")
            resp = "Sorry, I couldn't generate a response."

        # History logging
        self.history.append((user_query, resp))

        # Optional TTS
        if speak and resp:
            safe = self.clean_for_tts(resp)
            try:
                self.tts.speak(safe)
            except Exception as e:
                warn(f"TTS speak failed: {e}")

        return resp

    # ----------------------------------------------------------------------
    # RAG answer from prerecorded audio
    # ----------------------------------------------------------------------
    def answer_voice_file(
            self,
            audio_path: str,
            top_k: int = 4,
            speak: bool = True
    ) -> str:
        """
        STT → RAG pipeline for existing audio files.
        """
        if not audio_path:
            return "Invalid audio path."

        try:
            text = self.stt.transcribe_file(audio_path)
        except Exception as e:
            warn(f"STT transcription error: {e}")
            return "Could not transcribe audio."

        if not text:
            return "Could not transcribe audio."

        return self.answer_text(text, top_k=top_k, speak=speak)

    # ----------------------------------------------------------------------
    # Record microphone audio and answer
    # ----------------------------------------------------------------------
    def record_and_answer(
            self,
            duration: Optional[int] = None,
            top_k: int = 4,
            speak: bool = True
    ) -> str:
        """
        Record → WAV → STT → RAG pipeline.
        Uses temp files to maintain consistent STT behavior.
        """
        duration = duration or CONFIG.get("record_seconds_default", 5)

        # Capture microphone input
        try:
            audio, fs = record_to_numpy(duration_seconds=duration)
        except Exception as e:
            warn(f"Recording failed: {e}")
            return "Could not record audio."

        tmp_path = f"tmp_{uuid.uuid4().hex}.wav"

        # Persist temp file for STT
        try:
            save_numpy_to_wav(audio, tmp_path, fs=fs)
        except Exception as e:
            warn(f"Saving temporary WAV failed: {e}")
            return "Could not process recorded audio."

        # Convert to text → RAG
        try:
            return self.answer_voice_file(tmp_path, top_k=top_k, speak=speak)
        finally:
            # Clean temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass
