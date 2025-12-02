# Local Voice-RAG

Local Voice-RAG is a **fully offline Retrieval-Augmented Generation (RAG) assistant** optimized for CPU. It supports:

- **Text & voice queries**
- **PDF/TXT indexing**
- **Semantic search**
- **Speech-to-Text (STT)**
- **Text-to-Speech (TTS)**
- **Interactive CLI & Gradio UI**

Powered by:

- **Ollama** (LLM)
- **Faster-Whisper** (STT)
- **Coqui TTS**
- **ChromaDB** (vector store)

---

## Features

### Core

- Offline **PDF/TXT ingestion** with chunking
- Local **embedding generation** for semantic search
- **RAG-augmented answers** using local LLM
- **TTS & STT support**
- **CPU-optimized** Gradio UI
- Full **chat history** storage and export

### New

- Interactive **CLI REPL**
    - Multiline input (`:ml` or `Ctrl+J`)
    - Audio recording (`record` or `Ctrl+R`)
    - TTS toggle (`speakon` / `speakoff`)
    - File-based audio queries (`file <wav|mp3>`)
- **Incremental indexing**: only index new documents
- **Hybrid UI + CLI mode**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/voice-rag.git
cd voice-rag
```

### 2. Create Python environment

**Recommended (Conda)**

```bash
conda create -n voicerag python=3.10 -y
conda activate voicerag
```

**Or Python venv**

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional (if not included in `requirements.txt`):

```bash
pip install faster-whisper chromadb sentence-transformers TTS pypdf pymupdf gradio
```

Ensure **Ollama** is installed and running:

```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## Configuration

All configuration lives in `voice_rag/config.py`. Example:

```python
CONFIG = {
    # --------------------------
    # Whisper STT
    # --------------------------
    "whisper_model": "small",                    # Efficient Whisper ASR model offering great speed and accuracy
    "whisper_compute_type": "int8",              # INT8 compute for fast, lightweight inference
    "whisper_beam_size": 1,                      # Single-beam decoding for low-latency transcription
    "whisper_language": "en",                    # Optimized for high-quality English transcription
    "whisper_temperature": (0.0, 0.2, 0.4),      # Multiple temperatures for flexible decoding behavior

    # --------------------------
    # Embeddings / Vector Index
    # --------------------------
    "embedding_model": "voice_rag/bge-small-en", # Compact, high-performance embedding model
    "chroma_dir": "voice_rag/chroma_db",         # Path to the Chroma vector store for fast retrieval
    "docs_dir": "voice_rag/docs",                # Folder containing documents to index and search
    "embedding_batch_size": 32,                  # Solid batch size for efficient embedding generation
    "embedding_num_workers": 0,                  # Single-process embedding for predictable performance

    # --------------------------
    # LLM (Ollama)
    # --------------------------
    "ollama_http": "http://localhost:11434",     # Local Ollama API endpoint for quick LLM access
    "ollama_model": "qwen2.5:7b",                # Strong mid-size Qwen model providing rich responses
    "llm_max_tokens": 1200,                      # Generous token limit for detailed outputs

    # --------------------------
    # TTS
    # --------------------------
    "coqui_model": "tts_models/en/ljspeech/fast_pitch", # Fast, natural English TTS for smooth playback

    # --------------------------
    # Recording
    # --------------------------
    "record_seconds_default": 15,                # Convenient default recording duration
    "sample_rate": 16000,                        # Standard telephony-grade sample rate for STT compatibility
    "mic_device_id": None,                       # Auto-select microphone for user-friendly setup
    "mic_channels": 1,                           # Mono input for clean and simple audio capture

    # --------------------------
    # Document Handling
    # --------------------------
    "save_repaired_pdf": False,                  # Optional PDF repair disabled for faster processing

    # --------------------------
    # Reranking
    # --------------------------
    "reranker_use_cross_encoder": False,         # Lightweight reranking without cross-encoder overhead
    "reranker_cross_max_tokens": 48,             # Efficient token cap for compact cross-encoding
    "reranker_cross_batch_size": 4,              # Small batch size for predictable CPU use
    "mmr_lambda": 0.6,                           # Balanced MMR tuning for diverse yet relevant results

    # --------------------------
    # Retrieval
    # --------------------------
    "max_retrieval_topk": 8,                     # Tight top-K limit for high-precision retrieval
    "initial_retrieval_k": 16,                   # Broader initial search for robust recall

    # --------------------------
    # Prompt Management
    # --------------------------
    "context_chunk_preview_chars": 1024,         # Handy preview window for context chunk inspection

    # --------------------------
    # General Output
    # --------------------------
    "output_dir": "output",                      # Central output directory for all generated artifacts
}



```

---

## Usage

Run the main entrypoint:

```bash
python run.py [options]
```

### 1. Build or rebuild index

```bash
python run.py --reindex
```

### 2. Index specific PDFs

```bash
python run.py --pdfs doc1.pdf doc2.pdf --reindex
```

### 3. Launch Gradio UI

```bash
python run.py --ui
```
Visit: [http://127.0.0.1:7861](http://127.0.0.1:7861)

### 4. Interactive CLI Chat

```bash
python run.py --chat
```

#### Commands

| Command | Description |
|---------|-------------|
| `record` | Record audio → STT → RAG → LLM |
| `file <path>` | Use an audio file as query |
| `speakon` / `speakoff` | Enable/disable TTS |
| `:ml` | Multiline input |
| `exit` / `quit` | Exit REPL |

#### Hotkeys

| Hotkey | Action |
|--------|--------|
| Ctrl+J | Multiline query |
| Ctrl+R | Record audio |

### 5. Hybrid Mode (UI + CLI)

```bash
python run.py --ui --chat
```
Allows interacting with both **UI** and **CLI** simultaneously.

---

## Chat History

- Stored in `output/chat_history.json`
- Each entry contains:
    - User text or STT transcript
    - Assistant response
    - RAG source snippets (top-k)
    - TTS audio output path
    - Timestamp

Export plaintext history:

```bash
python -m voice_rag.history.export_txt
```

---

## Folder Structure

```
voice-rag/
├── run.py                     # Main launcher
├── requirements.txt
├── LICENSE
├── README.md
└── voice_rag/
    ├── agent.py               # Core RAG agent
    ├── agent_helpers.py       # Audio recording & utilities
    ├── cli.py                 # CLI + hotkeys + interactive REPL
    ├── config.py              # Config + directory setup
    ├── embeddings.py          # ChromaDB + embeddings
    ├── history.py             # Chat history storage/export
    ├── hotkeys.py             # Hotkey manager
    ├── pdf_loader.py          # PDF/TXT parsing & chunking
    ├── reranker.py            # Optional reranking
    ├── stt.py                 # Faster Whisper STT wrapper
    ├── tts.py                 # Coqui TTS wrapper
    ├── ui.py                  # Gradio UI
    └── utils.py               # Logging & helpers
```

---

## Notes & Tips

- **Large PDFs** may take longer to embed on CPU
- **Incremental indexing** prevents re-processing already-indexed documents
- **Audio device selection** configurable in `config.py`
- **Scanned PDFs** require OCR preprocessing

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| Gradio UI not loading | Check port 7861 availability |
| STT errors | Verify microphone and Faster Whisper installation |
| TTS errors | Check audio output device & Coqui TTS installation |
| ChromaDB issues | Delete `voice_rag/chroma_db` and reindex |

---

## License

MIT License

---

## Contributing

- PRs and issues are welcome
- Suggest new features or optimizations
- Maintain consistent style & doc