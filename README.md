# Local Voice-RAG (Demo)

Local Voice-RAG is a **CPU-optimized local Retrieval-Augmented Generation (RAG) assistant** for text and voice queries. It supports **PDF/TXT indexing**, **semantic search**, **speech-to-text (STT)**, and **text-to-speech (TTS)** — all running **fully offline**.

Powered by **Ollama LLM**, **Faster Whisper**, **Coqui TTS**, and **ChromaDB**.

---

## Features

- **Text & Voice Queries**
- **PDF/TXT Document Indexing**
- **Retrieval-Augmented Generation (RAG)**
- **Local TTS & STT**
- **CPU‑Optimized**
- **Gradio UI**

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/voice-rag.git
cd voice-rag
```

### 2. Create a Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Optional:
```bash
pip install faster-whisper sentence-transformers chromadb TTS pypdf pymupdf gradio
```

---

## Configuration

```python
CONFIG = {
    "whisper_model": "small",
    "coqui_model": "tts_models/en/ljspeech/fast_pitch",
    "embedding_model": "voice_rag/bge-small-en",
    "chroma_dir": "voice_rag/chroma_db",
    "docs_dir": "voice_rag/docs",
    "ollama_http": "http://localhost:11434",
    "ollama_model": "qwen2.5:3b",
    "record_seconds_default": 10,
    "sample_rate": 16000,
    "mic_device_id": None,
    "mic_channels": 1,
}
```

---

## Usage

### Index all documents
```bash
python run.py --reindex
```

### Index specific PDFs
```bash
python run.py --pdfs doc1.pdf doc2.pdf --reindex
```

### Launch UI
```bash
python run.py --ui
```

Visit: <http://127.0.0.1:7861>

---

## Folder Structure

```
voice-rag/
├── voice_rag/
│   ├── agent.py          # Core RAG assistant logic
│   ├── embeddings.py     # ChromaDB + embeddings
│   ├── ollama.py         # Ollama HTTP client
│   ├── stt.py            # Faster Whisper STT
│   ├── tts.py            # Coqui TTS wrapper
│   ├── pdf_loader.py     # PDF/TXT parsing & chunking
│   ├── ui.py             # Gradio UI
│   ├── cli.py            # Indexing CLI
│   ├── config.py         # Config system
│   └── utils.py          # Helper utilities
├── run.py
├── requirements.txt
├── LICENSE
└── README.md

```

---
## Notes & Tips

- Ensure Ollama is running and reachable via ollama_http.
- PDF parsing works best with text-based PDFs (OCR required for scans).
- Chroma DB stores embeddings locally in voice_rag/chroma_db.
- TTS requires sounddevice and scipy installed.
- STT/TTS models can be swapped in config.py.

---
## Troubleshooting

- Gradio won’t launch → Check port 7861  
- STT errors → Verify Faster Whisper + microphone  
- TTS fails → Check audio output  
- ChromaDB errors → Delete `chroma_db` and reindex  

---

## License
MIT License.

## Contributing
Open issues and PRs welcome!
