from pathlib import Path
from typing import Any, Dict

# =============================================================================
# Global Configuration
# =============================================================================
# Central config dictionary for the Voice-RAG system. All components (STT, TTS,
# LLM, embeddings, retrieval, I/O dirs) pull defaults from here.

CONFIG: Dict[str, Any] = {
    # --------------------------
    # Whisper STT
    # --------------------------
    "whisper_model": "small",
    "whisper_compute_type": "int8",
    "whisper_beam_size": 1,
    "whisper_language": "en",
    "whisper_temperature": (0.0, 0.2, 0.4),

    # --------------------------
    # Embeddings / Vector Index
    # --------------------------
    "embedding_model": "voice_rag/bge-small-en",
    "chroma_dir": "voice_rag/chroma_db",
    "docs_dir": "voice_rag/docs",
    "embedding_batch_size": 32,
    "embedding_num_workers": 0,

    # --------------------------
    # LLM (Ollama)
    # --------------------------
    "ollama_http": "http://localhost:11434",
    "ollama_model": "mistral:7b", # "qwen2.5:3b",
    "llm_max_tokens": 2400,

    # --------------------------
    # TTS
    # --------------------------
    "coqui_model": "tts_models/en/ljspeech/fast_pitch",

    # --------------------------
    # Recording
    # --------------------------
    "record_seconds_default": 15,
    "sample_rate": 16000,
    "mic_device_id": None,
    "mic_channels": 1,

    # --------------------------
    # Document Handling
    # --------------------------
    "save_repaired_pdf": False,

    # --------------------------
    # Reranking
    # --------------------------
    "reranker_use_cross_encoder": True,
    "reranker_cross_max_tokens": 48,
    "reranker_cross_batch_size": 4,
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mmr_lambda": 0.6,


    # --------------------------
    # Retrieval
    # --------------------------
    "max_retrieval_topk": 8,
    "initial_retrieval_k": 16,

    # --------------------------
    # Prompt Management
    # --------------------------
    "context_chunk_preview_chars": 1024,

    # --------------------------
    # General Output
    # --------------------------
    "output_dir": "output",
}

# =============================================================================
# Directory Initialization
# =============================================================================

def _ensure_dir(path_value: Any, key: str) -> Path:
    """Resolve a filesystem path from CONFIG, ensure the directory exists, and
    return it as a Path. Raises RuntimeError on failure for clear surfacing."""
    try:
        p = Path(path_value).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize CONFIG['{key}'] directory ({path_value}): {e}"
        )

# Normalize directory paths early so downstream modules receive only Path objs.
CONFIG["chroma_dir"] = _ensure_dir(CONFIG["chroma_dir"], "chroma_dir")
CONFIG["docs_dir"] = _ensure_dir(CONFIG["docs_dir"], "docs_dir")
CONFIG["output_dir"] = _ensure_dir(CONFIG["output_dir"], "output_dir")