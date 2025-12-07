## \# Local Voice-RAG

![My Picture](voice_rag/img/voice-rag.PNG)

Local Voice-RAG is a **fully offline Retrieval-Augmented Generation
(RAG) assistant** optimized for CPU, with **optional online web
retrieval** capabilities. It supports:

-   **Text & voice queries**
-   **PDF/TXT indexing**
-   **Semantic search**
-   **Speech-to-Text (STT)**
-   **Text-to-Speech (TTS)**
-   **Web retrievers (Wikipedia, arXiv, Tavily)**
-   **Interactive CLI & Gradio UI**

Powered by:

-   **Ollama** (LLM)
-   **Faster-Whisper** (STT)
-   **Coqui TTS**
-   **ChromaDB** (vector store)

------------------------------------------------------------------------

## Features

### Core

-   Offline **PDF/TXT ingestion** with chunking
-   Local **embedding generation** for semantic search
-   **RAG-augmented answers** using local LLM
-   **TTS & STT support**
-   **CPU-optimized** Gradio UI
-   **Chat history** storage and export

### Web Retrieval (New)

-   Unified **Web Retriever Layer** with:
    -   **Wikipedia search**
    -   **arXiv scientific paper search**
    -   **Tavily** (if API key provided)
-   Automatic fallback between **local RAG** ↔ **web search**
-   Configurable priority and toggles in `config.py`
-   Included in both **CLI** and **UI** workflows

### New

-   Interactive **CLI REPL**
    -   Multiline input (`:ml` or `Ctrl+J`)
    -   Audio recording (`record` or `Ctrl+R`)
    -   TTS toggle (`speakon` / `speakoff`)
    -   File-based audio queries (`file <wav|mp3>`)
-   **Incremental indexing**: only index new documents
-   **Hybrid UI + CLI mode**

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/yourusername/voice-rag.git
cd voice-rag
```

### 2. Create Python environment

**Recommended (Conda)**

``` bash
conda create -n voicerag python=3.10 -y
conda activate voicerag
```

**Or Python venv**

``` bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Optional (if not included in `requirements.txt`):

``` bash
pip install faster-whisper chromadb sentence-transformers TTS pypdf pymupdf gradio
```

Ensure **Ollama** is installed and running:

``` bash
ollama serve
ollama pull qwen2.5:3b
```

------------------------------------------------------------------------

## Configuration

All configuration lives in `voice_rag/config.py`. Example:

``` python
CONFIG = {
    "whisper_model": "small",
    "embedding_model": "voice_rag/bge-small-en",
    "chroma_dir": "voice_rag/chroma_db",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ollama_http": "http://localhost:11434",
    "ollama_model": "qwen2.5:3b",
    "coqui_model": "tts_models/en/ljspeech/fast_pitch",
    "record_seconds_default": 15,
    "sample_rate": 16000,
    "mic_channels": 1,
    "output_dir": "output",
}
```

------------------------------------------------------------------------

## Usage

Run the main entrypoint:

``` bash
python run.py [options]
```

### 1. Build or rebuild index

``` bash
python run.py --reindex
```

### 2. Index specific PDFs

``` bash
python run.py --pdfs doc1.pdf doc2.pdf --reindex
```

### 3. Launch Gradio UI

``` bash
python run.py --ui
```

Visit: http://127.0.0.1:7861

### 4. Interactive CLI Chat

``` bash
python run.py --chat
```

#### Commands

Command                  Description
  ------------------------ --------------------------------
`record`                 Record audio → STT → RAG → LLM
`file <path>`            Use an audio file as query
`speakon` / `speakoff`   Enable/disable TTS
`:ml`                    Multiline input
`exit` / `quit`          Exit REPL

#### Hotkeys

Hotkey   Action
  -------- -----------------
Ctrl+J   Multiline query
Ctrl+R   Record audio

### 5. Hybrid Mode (UI + CLI)

``` bash
python run.py --ui --chat
```

------------------------------------------------------------------------

## System Flow

The following diagram illustrates the high-level architecture and data flow of **Local Voice-RAG**:

```mermaid
flowchart TD
    A[User Input] -->|Text| B[Text Handler]
    A -->|Voice| C[Voice Handler STT]
    B --> D[RAG Agent]
    C --> D

    D --> M{Web Retrieval Enabled?}
    M -->|Yes| W[Web Retriever]
    M -->|No| E[Local Retrieval]

    W --> E[Context Retrieval]

    E --> F[LLM Ollama]
    F --> G[Answer Generation]

    G --> H{TTS Enabled?}
    H -->|Yes| I[Coqui TTS]
    H -->|No| J[Text Response]

    I --> J
    J --> K[Display in UI or CLI]
    K --> L[Chat History Storage]

```

### Flow Description

1. **User Input**: Text or audio is captured via CLI or Gradio UI.
2. **Voice Handler**: Audio is transcribed to text using **Faster-Whisper STT**.
3. **Text Handler**: Handles user queries, forwards to **LocalRAGAgent**.
4. **Retriever Layer**: Optional context retrieval from:
    - Wikipedia
    - arXiv
    - Tavily (if API key present)
5. **LocalRAGAgent + LLM**: Combines retrieved context with the query and generates an answer.
6. **TTS (Optional)**: Generates spoken output via **Coqui TTS** if enabled.
7. **Chat History**: Stores the conversation, context, and optional audio output in `output/chat_history.json`.

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
    ├── utils.py               # Logging & helpers
    └── web_retreivers.py      # Web Retreivers (wiki, ArXiv, Tavily)
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
- Suggestions