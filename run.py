"""
CLI entrypoint for building and launching the local Voice-RAG index and UI.
"""

import argparse
from voice_rag.cli import setup_index
from voice_rag.ui import launch_gradio_app
from voice_rag.utils import info


def main():
    """
    Main CLI function for indexing documents and optionally launching the Gradio UI.

    Usage Examples:
        python main.py --pdfs doc1.pdf doc2.pdf --reindex --ui
    """
    parser = argparse.ArgumentParser(
        description="Local Voice-RAG: Index PDFs/TXTs and launch Gradio UI."
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=None,
        help="Optional PDF or TXT file paths to index. If omitted, defaults to docs_dir."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear and rebuild the Chroma DB before indexing."
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio UI after indexing is complete."
    )
    args = parser.parse_args()

    # Build or update the local RAG index
    agent = setup_index(pdf_filepaths=args.pdfs, reindex=args.reindex)
    info("Index ready.")

    # Launch Gradio UI if requested
    if args.ui:
        launch_gradio_app(agent)


if __name__ == "__main__":
    main()
