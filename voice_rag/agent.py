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
    A fully local retrieval-augmented generation agent with STT, TTS, Ollama LLM,
    and optional reranking. All design is optimized for clarity and robustness.
    """
    ollama: Any
    stt: Any
    tts: Any
    index: LocalEmbeddingIndex

    history: List[Tuple[str, str]] = field(default_factory=list)

    system_prompt: str = (
        "You are a highly reliable, detail-oriented assistant.\n\n"
        "Always begin by thoroughly analyzing the provided CONTEXTS. Treat them as "
        "the primary source of truth. Do not assume, infer, or speculate beyond what "
        "the CONTEXTS explicitly state.\n\n"
        "When relevant information exists in the CONTEXTS, summarize, synthesize, and "
        "explain it in your own words. Do not copy long passages verbatim unless quoting "
        "is essential.\n\n"
        "Use internal knowledge only to clarify or organize, never to assert unverified facts. "
        "Label unsupported claims as uncertain or unknown.\n\n"
        "Avoid making statements about product features or availability unless explicitly "
        "supported by the CONTEXTS.\n\n"
        "Produce clear, well-structured explanations suitable for text-to-speech. "
        "Avoid raw JSON, internal markers, or metadata.\n"
        "Finally, you're final answer MUST always be in FRENCH.\n"
    )

    reranker: Optional[ReRanker] = None

    # ----------------------------------------------------------------------
    # Post-init: setup reranker if needed
    # ----------------------------------------------------------------------
    def __post_init__(self):
        if self.reranker is None:
            self.reranker = ReRanker(
                embed_fn=self.index.embed_texts,
                ollama=self.ollama
            )

    # ----------------------------------------------------------------------
    # Prompt builder
    # ----------------------------------------------------------------------
    def build_prompt(self, user_query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Assemble a full prompt including context blocks and system prompt.
        Uses raw (uncompressed) text — typically for debugging / analysis.
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
    # Clean text for TTS safety
    # ----------------------------------------------------------------------
    def clean_for_tts(self, text: str, *args, **kwargs) -> str:
        """
        Performs extremely conservative cleanup to reduce TTS crashes.
        Removes markdown, emojis, IPA, and non-ASCII.
        """
        import re

        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r"[*#`]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Strip non-ASCII
        return text.strip()

    # ----------------------------------------------------------------------
    # Compress contexts for LLM (for length control)
    # ----------------------------------------------------------------------
    def _compress_contexts_for_llm(
            self,
            docs: List[Dict[str, Any]],
            max_chars_per_chunk: Optional[int] = None
    ) -> str:
        """
        Produces trimmed preview of each context chunk.
        Prevents LLM runaway input size.
        """
        if max_chars_per_chunk is None:
            max_chars_per_chunk = CONFIG.get("context_chunk_preview_chars", 1024)

        parts = []

        for i, d in enumerate(docs):
            txt = d.get("text", "") or ""
            meta = d.get("meta", {}) or {}

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
    # Answer from text
    # ----------------------------------------------------------------------
    def answer_text(
            self,
            user_query: str,
            top_k: int = 4,
            speak: bool = False
    ) -> str:
        """
        Main RAG text answering routine.
        - Retrieves relevant passages
        - Reranks
        - Builds compressed prompt
        - Queries LLM
        - Optionally TTS speaks the answer
        """
        if not user_query or not user_query.strip():
            return ""

        user_query = user_query.strip()
        info(f"Query: {user_query[:120]}")

        # Retrieval upper bounds
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

        # Convert to reranker candidate format
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

        # Reranking
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

        # Build compressed contexts
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

        # Save to history
        self.history.append((user_query, resp))

        # Optional speaking
        if speak and resp:
            safe = self.clean_for_tts(resp)
            try:
                self.tts.speak(safe)
            except Exception as e:
                warn(f"TTS speak failed: {e}")

        return resp

    # ----------------------------------------------------------------------
    # Answer from audio file
    # ----------------------------------------------------------------------
    def answer_voice_file(
            self,
            audio_path: str,
            top_k: int = 4,
            speak: bool = True
    ) -> str:
        """
        STT → RAG pipeline for prerecorded audio files.
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
    # Record audio and answer
    # ----------------------------------------------------------------------
    def record_and_answer(
            self,
            duration: Optional[int] = None,
            top_k: int = 4,
            speak: bool = True
    ) -> str:
        """
        Record microphone audio → save temporary WAV → STT → RAG.
        """
        duration = duration or CONFIG.get("record_seconds_default", 5)

        # Record
        try:
            audio, fs = record_to_numpy(duration_seconds=duration)
        except Exception as e:
            warn(f"Recording failed: {e}")
            return "Could not record audio."

        tmp_path = f"tmp_{uuid.uuid4().hex}.wav"

        # Save to temp file
        try:
            save_numpy_to_wav(audio, tmp_path, fs=fs)
        except Exception as e:
            warn(f"Saving temporary WAV failed: {e}")
            return "Could not process recorded audio."

        # Use STT → RAG pipeline
        try:
            return self.answer_voice_file(tmp_path, top_k=top_k, speak=speak)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
