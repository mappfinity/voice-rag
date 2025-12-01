"""
Voice-RAG CLI entrypoint.

Concise CLI for indexing PDF documents, launching a Gradio UI,
and running an interactive chat loop. Follows standard 2025 ML
tooling conventions: minimal side effects, clear docstrings,
and explicit threading for background UI.
"""
# ===============================================
# Environment & Telemetry Configuration
# ===============================================
from __future__ import annotations

import os

# Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Gradio
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

# ChromaDB
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Generic PostHog client disable
os.environ["POSTHOG_DISABLED"] = "1"

# Completely disable PostHog client if installed
try:
    import posthog

    class NoOpPostHog:
        def __getattr__(self, name):
            def noop(*args, **kwargs):
                return None
            return noop

    posthog.Posthog = NoOpPostHog()
    posthog.capture = lambda *a, **k: None
    posthog.identify = lambda *a, **k: None
    posthog.flush = lambda *a, **k: None

    print("PostHog telemetry: BLOCKED")
except ImportError:
    print("posthog package not installed — nothing to block.")

# ===============================================
# Warnings
# ===============================================
import warnings
warnings.filterwarnings('ignore')
print("All warnings suppressed.")

# ===============================================
# Standard Library Imports
# ===============================================
import sys
import logging
import argparse
import threading

# ===============================================
# Logging Configuration
# ===============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("optimized_rag")

# ===============================================
# Optional / Third-Party Imports
# ===============================================
# NLTK
try:
    import nltk
    # nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
    HAS_NLTK_SENT = True
except Exception:
    HAS_NLTK_SENT = False

# Numerical and HTTP libraries
try:
    import numpy as np
except Exception:
    np = None

try:
    import requests
except Exception:
    requests = None

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None

# NLP / Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# PDF / Document processing
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    fitz = None
    HAS_FITZ = False

# Whisper / TTS / Audio
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from TTS.api import TTS
except Exception:
    TTS = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

# Gradio
try:
    import gradio as gr
except Exception:
    gr = None

# ===============================================
# Application-Specific Imports
# ===============================================
from voice_rag.cli import setup_index, interactive_chat
from voice_rag.ui import launch_gradio_app
from voice_rag.utils import info



def launch_ui_background(agent):
    """
    Launch the Gradio UI in a background daemon thread.

    Args:
        agent: The Voice-RAG agent instance.
    """
    thread = threading.Thread(
        target=launch_gradio_app,
        args=(agent,),
        daemon=True,
    )
    thread.start()
    info("UI running on http://localhost:7861")


def is_interactive():
    """
    Determine whether running inside an interactive environment
    (Jupyter, REPL, IDE console).

    Returns:
        bool: True if interactive, False otherwise.
    """
    return not sys.stdout.isatty() or hasattr(sys, "ps1")


def parse_args():
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Local Voice-RAG")
    parser.add_argument("--pdfs", nargs="*", default=None,
                        help="Paths to PDF files to index.")
    parser.add_argument("--reindex", action="store_true",
                        help="Force rebuild of the index.")
    parser.add_argument("--ui", action="store_true",
                        help="Launch Gradio UI.")
    parser.add_argument("--chat", action="store_true",
                        help="Run terminal chat interface.")
    return parser.parse_args()


def main():
    """
    Main CLI entrypoint.

    Workflow:
        1. Index PDFs (optional reindex).
        2. Launch UI and/or terminal chat per flags.
        3. Auto-launch UI in interactive environments when no mode is chosen.
    """
    args = parse_args()

    agent = setup_index(pdf_filepaths=args.pdfs, reindex=args.reindex)
    info("Index ready.")

    # Auto-launch UI by default in interactive sessions
    if is_interactive() and not (args.ui or args.chat):
        launch_gradio_app(agent)
        return

    # Combined UI + chat: UI in background, chat in foreground
    if args.ui and args.chat:
        launch_ui_background(agent)
        interactive_chat(agent)
        return

    if args.ui:
        launch_gradio_app(agent)
        return

    if args.chat:
        interactive_chat(agent)
        return

    info("Nothing to do. Use --ui and/or --chat.")


if __name__ == "__main__":
    main()
