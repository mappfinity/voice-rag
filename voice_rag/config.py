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
    "whisper_model": "small",  # Upgraded from 'small' for better accuracy with technical terms
    "whisper_compute_type": "int8",
    "whisper_beam_size": 3,  # Increased from 1 for better transcription quality
    "whisper_language": "en",
    "whisper_temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # Extended range for better fallback
    "whisper_vad_filter": True,  # Add Voice Activity Detection to reduce hallucinations
    "whisper_initial_prompt": "Technical discussion about machine learning, data science, AI engineering, and programming.",  # Bias toward technical vocabulary

    # --------------------------
    # Embeddings / Vector Index
    # --------------------------
    "embedding_model": "voice_rag/bge-small-en",  # Consider upgrading to bge-base-en-v1.5 for better quality
    "chroma_dir": "voice_rag/chroma_db",
    "docs_dir": "voice_rag/docs",
    "embedding_batch_size": 64,  # Increased from 32 for better throughput
    "embedding_num_workers": 2,  # Changed from 0 to enable parallel processing (adjust based on CPU cores)
    "embedding_normalize": True,  # Normalize embeddings for better cosine similarity
    "embedding_show_progress": True,  # Show progress for large document sets

    # --------------------------
    # LLM (Ollama)
    # --------------------------
    "ollama_http": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",   # "mistral:7b" default; alternative: "qwen2.5:7b" for technical tasks
    "llm_max_tokens": 3072,  # Increased from 2400 for more comprehensive responses
    "llm_context_window": 32768,  # Track model's context limit for better context management
    "llm_timeout": 360,  # Add timeout to prevent hanging requests
    "llm_streaming": True,  # Enable streaming for real-time response display

    # --------------------------
    # TTS
    # --------------------------
    "coqui_model": "tts_models/en/vctk/fast_pitch",  # Upgraded for better multi-speaker quality
    "tts_speed": 1.0,  # Speech speed multiplier
    "tts_sample_rate": 22050,  # Standard TTS sample rate

    # --------------------------
    # Recording
    # --------------------------
    "record_seconds_default": 20,  # Increased from 15 for longer technical explanations
    "record_seconds_max": 60,  # Add maximum limit
    "sample_rate": 16000,
    "mic_device_id": None,
    "mic_channels": 1,
    "audio_format": "wav",  # Explicit format specification

    # --------------------------
    # Document Handling
    # --------------------------
    "save_repaired_pdf": False,
    "chunk_size": 512,  # Add explicit chunk size for document splitting
    "chunk_overlap": 128,  # 25% overlap for better context preservation
    "min_chunk_size": 100,  # Minimum viable chunk size
    "separators": ["\n\n", "\n", ". ", " ", ""],  # Hierarchical text splitting

    # --------------------------
    # Reranking
    # --------------------------
    "reranker_use_cross_encoder": True,
    "reranker_cross_max_tokens": 96,  # Doubled from 48 for better context in reranking
    "reranker_cross_batch_size": 8,  # Increased from 4 for better throughput
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Upgraded to L-12 for better reranking quality
    "mmr_lambda": 0.7,  # Increased from 0.6 for better diversity vs relevance balance
    "mmr_use": True,  # Explicit flag to enable/disable MMR

    # --------------------------
    # Retrieval
    # --------------------------
    "max_retrieval_topk": 5,  # Reduced from 8 to focus on most relevant chunks
    "initial_retrieval_k": 20,  # Increased from 16 for better reranking pool
    "retrieval_score_threshold": 0.3,  # Add minimum similarity threshold
    "retrieval_strategy": "hybrid",  # Options: "semantic", "hybrid", "mmr"
    "keyword_weight": 0.3,  # For hybrid retrieval: balance between semantic (0.7) and keyword (0.3)

    # --------------------------
    # Prompt Management
    # --------------------------
    "context_chunk_preview_chars": 2048,  # Doubled from 1024 for more complete context
    "max_context_chunks": 5,  # Maximum chunks to include in LLM context
    "context_compression": False,  # Enable LLMLingua-style compression if needed
    "system_prompt_file": None,  # Optional: path to external system prompt file

    # --------------------------
    # General Output
    # --------------------------
    "output_dir": "output",
    "log_level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    "save_conversations": True,  # Save conversation history
    "conversation_history_limit": 10,  # Number of turns to keep in memory

    # --------------------------
    # Ollama LLM Generation Settings
    # --------------------------
    "temperature": 0.1,  # Reduced from 0.2 for more deterministic technical responses
    "top_p": 0.95,  # Increased from 0.9 for slightly more diversity
    "top_k": 50,  # Increased from 40 for better vocabulary coverage
    "repeat_penalty": 1.1,  # Prevent repetitive responses
    "num_predict": -1,  # -1 means use model default, or set explicit limit
    "stop_sequences": ["</s>", "[DONE]", "\n\nUser:", "\n\nHuman:"],  # Prevent runaway generation

    # --------------------------
    # Performance & Caching
    # --------------------------
    "enable_cache": True,  # Cache embeddings and LLM responses
    "cache_ttl": 3600,  # Cache time-to-live in seconds (1 hour)
    "max_cache_size_mb": 500,  # Maximum cache size in MB

    # --------------------------
    # Monitoring & Debugging
    # --------------------------
    "track_metrics": True,  # Track retrieval quality, latency, etc.
    "verbose_retrieval": False,  # Log detailed retrieval scores
    "verbose_llm": False,  # Log LLM token usage and timing
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

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config() -> None:
    """Validate configuration parameters for common issues."""
    # Validate retrieval parameters
    if CONFIG["max_retrieval_topk"] > CONFIG["initial_retrieval_k"]:
        raise ValueError(
            f"max_retrieval_topk ({CONFIG['max_retrieval_topk']}) cannot exceed "
            f"initial_retrieval_k ({CONFIG['initial_retrieval_k']})"
        )

    # Validate chunk parameters
    if CONFIG["chunk_overlap"] >= CONFIG["chunk_size"]:
        raise ValueError(
            f"chunk_overlap ({CONFIG['chunk_overlap']}) must be less than "
            f"chunk_size ({CONFIG['chunk_size']})"
        )

    # Validate temperature
    if not 0.0 <= CONFIG["temperature"] <= 2.0:
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {CONFIG['temperature']}")

    # Validate MMR lambda
    if not 0.0 <= CONFIG["mmr_lambda"] <= 1.0:
        raise ValueError(f"mmr_lambda must be between 0.0 and 1.0, got {CONFIG['mmr_lambda']}")

# Run validation on import
validate_config()