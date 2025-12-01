from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional
import uuid
import json
import re

# =============================================================================
# Local RAG Agent
# =============================================================================
from .config import CONFIG
from .utils import warn, info
from voice_rag.embeddings import LocalEmbeddingIndex
from voice_rag.reranker import ReRanker


# Ensure output directory exists
OUTPUT_DIR = Path(CONFIG.get("output_dir", "output")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LocalRAGAgent:
    """
    Local Retrieval-Augmented Generation agent.

    Responsibilities:
    - Retrieve + rerank text chunks
    - Compose LLM prompt from compressed contexts
    - Handle STT/TTS
    - Persist chat history + audio artifacts

    All operations remain fully local.
    """

    ollama: Any
    stt: Any
    tts: Any
    index: LocalEmbeddingIndex
    reranker: Optional[ReRanker] = None

    # (user_prompt, bot_response)
    history: List[Tuple[str, str]] = field(default_factory=list)

    system_prompt: str = (
        "You are a reliable, detail-oriented assistant.\n"
        "Analyze provided CONTEXTS as the primary source of truth.\n"
        "Synthesize relevant content concisely and avoid speculation.\n"
        "Produce clean, structured text suitable for TTS.\n"
    )

    def __post_init__(self):
        """Initialize a default reranker if none was provided."""
        if self.reranker is None:
            self.reranker = ReRanker(
                embed_fn=self.index.embed_texts,
                ollama=self.ollama,
            )

    # ---------------------------------------------------------------------
    # TTS-safe cleanup
    # ---------------------------------------------------------------------
    def clean_for_tts(self, text: str) -> str:
        """Normalize text for TTS: strip markdown, collapse whitespace, remove non-ASCII."""
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r"[*#`]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        return text.strip()

    # ---------------------------------------------------------------------
    # Context compressor for LLM prompt
    # ---------------------------------------------------------------------
    def _compress_contexts_for_llm(
            self,
            docs: List[Dict[str, Any]],
            max_chars_per_chunk: Optional[int] = None,
    ) -> str:
        """Return a compact preview of selected context chunks."""
        max_chars = max_chars_per_chunk or CONFIG.get("context_chunk_preview_chars", 1024)
        parts = []

        for d in docs:
            txt = str(d.get("text", "") or "").strip()
            meta = d.get("meta", {}) or {}

            source = str(meta.get("source", "unknown"))
            chunk_idx = meta.get("chunk_index", "?")

            if len(txt) > max_chars:
                head = int(max_chars * 0.6)
                tail = int(max_chars * 0.4)
                preview = f"{txt[:head]}\n... (truncated) ...\n{txt[-tail:]}"
            else:
                preview = txt

            parts.append(f"<<CONTEXT source={source} chunk={chunk_idx}>>\n{preview}")

        return "\n\n".join(parts) if parts else "<<NO_CONTEXT>>"

    # ---------------------------------------------------------------------
    # Main text-based RAG flow
    # ---------------------------------------------------------------------
    def answer_text(
            self,
            user_query: str,
            top_k: int = 4,
            speak: bool = False,
    ) -> Tuple[str, List[str], Optional[str]]:
        """
        Produce an LLM answer using retrieved + reranked context.
        Optionally synthesize audio.

        Returns:
            response_text, rag_sources, voice_path
        """
        if not user_query or not user_query.strip():
            return "", [], None

        user_query = user_query.strip()
        info(f"Query: {user_query[:120]}")

        top_k = max(1, min(top_k, CONFIG.get("max_retrieval_topk", 8)))

        # Initial retrieval
        try:
            retrieved = self.index.query(
                user_query,
                top_k=CONFIG.get("initial_retrieval_k", 16),
            )
        except Exception as e:
            warn(f"Initial retrieval failed: {e}")
            retrieved = []

        # Normalize retrieved items
        candidates = []
        rag_sources = []

        for i, item in enumerate(retrieved):
            try:
                doc, meta, dist = item
            except Exception:
                warn(f"Malformed retrieval item: {item}")
                continue

            candidates.append(
                {
                    "id": meta.get(
                        "id",
                        f"{meta.get('source','?')}_{meta.get('chunk_index', i)}",
                    ),
                    "text": doc,
                    "meta": meta,
                    "embedding": None,
                }
            )
            rag_sources.append((doc, meta, dist))

        # Rerank
        try:
            reranked = self.reranker.rerank(
                user_query,
                candidates,
                top_k=top_k,
                use_mmr=True,
            )
        except Exception as e:
            warn(f"Reranking failed: {e}")
            reranked = candidates[:top_k]

        # Generate prompt
        ctx_for_llm = self._compress_contexts_for_llm(reranked)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"CONTEXTS:\n{ctx_for_llm}\n\n"
            f"User: {user_query}\nAnswer:"
        )

        # LLM call
        try:
            resp = self.ollama.generate(
                prompt,
                max_tokens=CONFIG.get("llm_max_tokens", 1200),
            ).strip()
        except Exception as e:
            warn(f"LLM generation failed: {e}")
            resp = "Sorry, I could not generate a response."

        # Save history
        self.history.append((user_query, resp))
        try:
            with open(OUTPUT_DIR / "chat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            warn(f"Failed to save chat history: {e}")

        # Optional TTS
        voice_path = None
        if speak and resp:
            safe = self.clean_for_tts(resp)
            try:
                filename = f"tts_{uuid.uuid4().hex}.wav"
                out_path = OUTPUT_DIR / filename

                # Speed optimizations (if supported)
                if hasattr(self.tts, "griffin_lim_iters"):
                    self.tts.griffin_lim_iters = 15
                if hasattr(self.tts, "do_trim_silence"):
                    self.tts.do_trim_silence = True

                self.tts.speak_to_file(safe, str(out_path))
                voice_path = str(out_path.relative_to(Path.cwd()))
            except Exception as e:
                warn(f"TTS generation failed: {e}")

        return resp, rag_sources, voice_path

    # ---------------------------------------------------------------------
    # Answer from audio file
    # ---------------------------------------------------------------------
    def answer_voice_file(
            self,
            audio_path: str,
            top_k: int = 4,
            speak: bool = True,
    ) -> Tuple[str, List[str], Optional[str]]:
        """
        Transcribe an audio file, then run the normal text-based RAG flow.
        """
        if not audio_path:
            return "Invalid audio path.", [], None

        try:
            text = self.stt.transcribe_file(audio_path)
        except Exception as e:
            warn(f"STT error: {e}")
            return "Could not transcribe audio.", [], None

        if not text:
            return "Could not transcribe audio.", [], None

        return self.answer_text(text, top_k=top_k, speak=speak)

    # ---------------------------------------------------------------------
    # Record microphone → answer
    # ---------------------------------------------------------------------
    def record_and_answer(
            self,
            duration: Optional[int] = None,
            top_k: int = 4,
            speak: bool = True,
    ) -> Tuple[str, List[str], Optional[str]]:
        """
        Record temporary audio, transcribe, answer, then clean up.
        """
        duration = duration or CONFIG.get("record_seconds_default", 5)

        try:
            from voice_rag.agent_helpers import record_to_numpy, save_numpy_to_wav

            audio, fs = record_to_numpy(duration_seconds=duration)
            tmp_path = OUTPUT_DIR / f"tmp_{uuid.uuid4().hex}.wav"
            save_numpy_to_wav(audio, str(tmp_path), fs=fs)
        except Exception as e:
            warn(f"Recording failed: {e}")
            return "Could not record audio.", [], None

        try:
            return self.answer_voice_file(str(tmp_path), top_k=top_k, speak=speak)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
