"""
Gradio UI construction for Local Voice-RAG.

Provides text and voice chat interfaces, TTS/STT handling, and
retrieval-context visualization.
"""

from typing import Any, List, Dict
import os
import time
import uuid
from pathlib import Path

from voice_rag.config import CONFIG
from voice_rag.utils import die, warn
from voice_rag.agent import LocalRAGAgent
from voice_rag.history import ChatHistory

history = ChatHistory()

try:
    import gradio as gr
except Exception:
    gr = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _format_retrieved_for_display(retrieved: list) -> str:
    """Produce a short, human-readable snippet list of retrieved docs."""
    if not retrieved:
        return "No context retrieved."

    lines = []
    for i, item in enumerate(retrieved):
        try:
            doc = str(item[0]) if len(item) > 0 else ""
            meta = item[1] if len(item) > 1 else {}
            dist = float(item[2]) if len(item) > 2 else 0.0
        except Exception:
            doc, meta, dist = str(item), {}, 0.0

        src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
        snippet = doc[:800].replace("\n", " ").strip()
        lines.append(f"[{i}] source={src} | dist={dist:.4f}\n{snippet}...")
    return "\n\n".join(lines)

def safe_relative_path(file_path: str) -> str:
    """
    Returns a path relative to the current working directory if possible,
    otherwise returns the absolute path.
    """
    if not file_path:
        return None
    p = Path(file_path).resolve()
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)

def _fix_latex_for_gradio(text: str) -> str:
    """
    Fix LaTeX rendering for Gradio's Markdown component.

    Gradio uses KaTeX for LaTeX rendering, which requires proper delimiters:
    - Inline math: $...$
    - Display math: $$...$$

    This function:
    1. Converts escaped backslashes from JSON (\\) to single backslashes (\)
    2. Converts LaTeX bracket notation to dollar notation
    3. Preserves existing dollar-delimited math
    """
    if not text:
        return text

    # First, unescape double backslashes from JSON encoding
    text = text.replace("\\\\", "\\")

    # Convert display math: \[ ... \] to $$ ... $$
    import re
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Convert inline math: \( ... \) to $ ... $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text

# ---------------------------------------------------------------------------
# Gradio App Builder
# ---------------------------------------------------------------------------

def build_gradio_app(agent: LocalRAGAgent, title: str = "Local Voice-RAG (Optimized CPU)") -> Any:
    """
    Construct the complete Gradio UI: text chat, voice chat, and
    retrieval debug panes.
    """
    if gr is None:
        die("Gradio not installed. Install with `pip install gradio`.")

    chat_history: List[Dict[str, str]] = []

    OUTPUT_DIR = Path(CONFIG.get("output_dir", "output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.json"

    # ---------------- TEXT HANDLER ----------------

    def handle_text_submit(user_text: str, tts_enabled: bool, top_k: int):
        nonlocal chat_history
        user_text = (user_text or "").strip()
        if not user_text:
            return chat_history, "Please provide a question.", None, None

        answer, rag_sources, voice_file = agent.answer_text(
            user_text, top_k=top_k, speak=False
        )

        # Fix LaTeX for Gradio rendering
        answer = _fix_latex_for_gradio(answer)

        if tts_enabled:
            try:
                safe_text = agent.clean_for_tts(answer)
                fname = f"tts_{int(time.time())}.wav"
                fpath = OUTPUT_DIR / fname
                agent.tts.speak_to_file(safe_text, str(fpath))
                voice_file = fpath
            except Exception as e:
                warn(f"TTS generation failed: {e}")
                voice_file = None

        contexts_str = _format_retrieved_for_display(rag_sources)

        chat_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer}
        ])
        chat_history = chat_history[-40:]

        try:
            import json
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            warn(f"Failed to save chat history: {e}")

        voice_path = safe_relative_path(str(voice_file) if voice_file else None)
        history.add_interaction(
            user_text, answer, rag_sources=rag_sources,
            voice_output_path=voice_path
        )

        return chat_history, contexts_str, str(Path(voice_file).resolve()) if voice_file else None, ""

    # ---------------- VOICE HANDLER ----------------

    def handle_voice_submit(audio_in, tts_enabled: bool, top_k: int):
        nonlocal chat_history
        if not audio_in:
            return chat_history, "No audio provided.", None, None

        tmp_path = ""
        try:
            # Normalize audio input
            if isinstance(audio_in, str) and os.path.exists(audio_in):
                tmp_path = audio_in
            elif isinstance(audio_in, (tuple, list)) and len(audio_in) == 2:
                sr, arr = audio_in
                import numpy as np
                arr = np.asarray(arr)
                arr16 = (
                    (np.clip(arr, -1.0, 1.0) * 32767).astype("int16")
                    if arr.dtype.kind == "f" else arr.astype("int16")
                )
                tmp_path = f"gr_audio_{int(time.time())}.wav"
                wavfile.write(tmp_path, int(sr), arr16)
            else:
                return chat_history, "Unsupported audio input.", None, None

            transcription = agent.stt.transcribe_file(tmp_path)
            if not transcription:
                return chat_history, "Could not transcribe audio.", None, None

            answer, rag_sources, voice_file = agent.answer_text(
                transcription, top_k=top_k, speak=False
            )

            # Fix LaTeX for Gradio rendering
            answer = _fix_latex_for_gradio(answer)

            if tts_enabled:
                try:
                    safe_text = agent.clean_for_tts(answer)
                    fname = f"tts_{uuid.uuid4().hex}.wav"
                    voice_file = OUTPUT_DIR / fname
                    agent.tts.speak_to_file(safe_text, str(voice_file))
                except Exception as e:
                    warn(f"TTS generation failed: {e}")
                    voice_file = None

            contexts_str = _format_retrieved_for_display(rag_sources)

            chat_history.extend([
                {"role": "user", "content": f"[Voice] {transcription}"},
                {"role": "assistant", "content": answer}
            ])
            chat_history = chat_history[-40:]

            voice_path = safe_relative_path(str(voice_file) if voice_file else None)
            history.add_interaction(
                transcription, answer, rag_sources=rag_sources,
                voice_output_path=voice_path
            )

            return chat_history, contexts_str, str(Path(voice_file).resolve()) if voice_file else None, ""

        finally:
            if tmp_path.startswith("gr_audio_") and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ---------------- UI LAYOUT ----------------

    with gr.Blocks(title=title) as demo:
        gr.Markdown("## Local Voice-RAG — Optimized for CPU")

        with gr.Tabs():

            # ---------- TEXT CHAT ----------
            with gr.Tab("Text Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            render_markdown=True,
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ]
                        )
                        txt_in = gr.Textbox(label="Your question", lines=2)
                        with gr.Row():
                            tts_flag = gr.Checkbox(label="Enable TTS", value=False)
                            topk = gr.Slider(
                                1, CONFIG.get("max_retrieval_topk", 8),
                                step=1, value=4, label="Retrieval top_k"
                            )
                            send = gr.Button("Send")
                    with gr.Column(scale=2):
                        contexts = gr.Textbox(
                            label="Retrieved Contexts",
                            interactive=False, lines=20
                        )
                        audio_out = gr.Audio(
                            label="TTS Audio (if generated)",
                            interactive=False
                        )
                        status = gr.Textbox(
                            label="Status / Log",
                            interactive=False, lines=4
                        )

                send.click(
                    lambda t, tt, tk: handle_text_submit(t, tt, int(tk)),
                    inputs=[txt_in, tts_flag, topk],
                    outputs=[chat, contexts, audio_out, status]
                )
                txt_in.submit(
                    lambda t, tt, tk: handle_text_submit(t, tt, int(tk)),
                    inputs=[txt_in, tts_flag, topk],
                    outputs=[chat, contexts, audio_out, status]
                )

            # ---------- VOICE CHAT ----------
            with gr.Tab("Voice Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        voice_chat = gr.Chatbot(
                            label="Conversation (Voice)",
                            height=400,
                            render_markdown=True,
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ]
                        )
                        gr.Markdown(
                            "Record in your browser or upload an audio file."
                        )
                        audio_in = gr.Audio(
                            label="Record or upload audio",
                            type="filepath", interactive=True
                        )
                        with gr.Row():
                            v_tts = gr.Checkbox(label="Enable TTS", value=True)
                            v_topk = gr.Slider(
                                1, CONFIG.get("max_retrieval_topk", 8),
                                step=1, value=4, label="Retrieval top_k"
                            )
                            v_send = gr.Button("Process Audio")
                    with gr.Column(scale=2):
                        v_contexts = gr.Textbox(
                            label="Retrieved Contexts",
                            interactive=False, lines=20
                        )
                        v_audio_out = gr.Audio(
                            label="TTS Audio (if generated)",
                            interactive=False
                        )
                        v_status = gr.Textbox(
                            label="Status / Log",
                            interactive=False, lines=4
                        )

                v_send.click(
                    lambda a, tt, tk: handle_voice_submit(a, tt, int(tk)),
                    inputs=[audio_in, v_tts, v_topk],
                    outputs=[voice_chat, v_contexts, v_audio_out, v_status]
                )
                audio_in.change(
                    lambda a, tt, tk: handle_voice_submit(a, tt, int(tk)),
                    inputs=[audio_in, v_tts, v_topk],
                    outputs=[voice_chat, v_contexts, v_audio_out, v_status]
                )

        gr.Markdown("Local RAG demo. Ensure local models are running.")

    return demo

# ---------------------------------------------------------------------------
# App Launcher
# ---------------------------------------------------------------------------

def launch_gradio_app(agent: Any, ui_title: str = "Local Voice-RAG (CPU)") -> None:
    """
    Launch the Gradio app, handling required event-loop settings
    on Windows.
    """
    import asyncio
    import sys

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if gr is None:
        die("Gradio not installed.")

    app = build_gradio_app(agent, title=ui_title)

    app.launch(share=False, server_name="127.0.0.1", server_port=7861)