"""
CLI entrypoint for building/launching the local Voice-RAG index and UI.
"""

import argparse
from voice_rag.cli import setup_index
from voice_rag.ui import launch_gradio_app
from voice_rag.utils import info


def main():
    parser = argparse.ArgumentParser(description="Local Voice-RAG: Index PDFs and launch Gradio UI.")
    parser.add_argument(
        "--pdfs", nargs="*", default=None, help="Optional PDF/TXT file paths to index"
    )
    parser.add_argument(
        "--reindex", action="store_true", help="Clear and rebuild the Chroma DB"
    )
    parser.add_argument(
        "--ui", action="store_true", help="Launch Gradio UI after indexing"
    )
    args = parser.parse_args()

    # Build the index (from files or default docs_dir)
    agent = setup_index(pdf_filepaths=args.pdfs, reindex=args.reindex)
    info("Index ready.")

    # Launch UI if requested
    if args.ui:
        launch_gradio_app(agent)

if __name__ == "__main__":
    main()


#python main.py --pdfs doc1.pdf doc2.pdf --reindex --ui
