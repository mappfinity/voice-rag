from pathlib import Path
from typing import Any, Dict


# ============================================================================
# Configuration Dictionary
# ============================================================================
# All default settings for models, paths, audio, retrieval, and generation.
# This file is intentionally lightweight, but validated and robust.
# ============================================================================

CONFIG: Dict[str, Any] = {
    # --------------------------
    # Whisper STT
    # --------------------------
    "whisper_model": "small",
    "whisper_compute_type": "int8",     # CPU-friendly quantization
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
    "ollama_model": "qwen2.5-coder:7b-instruct",
    "llm_max_tokens": 1200,

    # --------------------------
    # TTS
    # --------------------------
    "coqui_model": "tts_models/en/ljspeech/fast_pitch", # "tts_models/fr/css10/vits"

    # --------------------------
    # Microphone / Audio Recording
    # --------------------------
    "record_seconds_default": 10,
    "sample_rate": 16000,
    "mic_device_id": None,
    "mic_channels": 1,

    # --------------------------
    # PDF / Document Handling
    # --------------------------
    "save_repaired_pdf": False,

    # --------------------------
    # Reranking
    # --------------------------
    "reranker_use_cross_encoder": False,
    "reranker_cross_max_tokens": 48,
    "reranker_cross_batch_size": 4,
    "mmr_lambda": 0.6,

    # --------------------------
    # Retrieval
    # --------------------------
    "max_retrieval_topk": 8,
    "initial_retrieval_k": 16,

    # --------------------------
    # Prompt Input Size Control
    # --------------------------
    "context_chunk_preview_chars": 1024,
}


# ============================================================================
# Directory Initialization (robust)
# ============================================================================

def _ensure_dir(path_value: Any, key: str) -> Path:
    """
    Convert config path to a safe Path object and ensure directory exists.
    Provides clear warnings if invalid.
    """
    try:
        p = Path(path_value).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize directory for CONFIG['{key}'] = {path_value}: {e}"
        )


# Normalize and ensure directories exist
CONFIG["chroma_dir"] = _ensure_dir(CONFIG["chroma_dir"], "chroma_dir")
CONFIG["docs_dir"]   = _ensure_dir(CONFIG["docs_dir"], "docs_dir")
