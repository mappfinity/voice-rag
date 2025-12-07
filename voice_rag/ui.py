"""
Gradio UI for Local Voice-RAG
Fully In-Memory Audio Pipeline (numpy-based) with safe fallbacks.
"""

from typing import Any, List, Dict, Optional, Tuple
import os
import time
import uuid
from pathlib import Path
import io

from voice_rag.config import CONFIG
from voice_rag.utils import die, warn
from voice_rag.agent import LocalRAGAgent
from voice_rag.history import ChatHistory

history = ChatHistory()

# Track fallback temp files (safe cleanup only when new recording occurs)
LAST_TEMP_AUDIO_FILE: Optional[str] = None
LAST_TTS_FILE: Optional[str] = None

try:
    import gradio as gr
except Exception:
    gr = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_bool(val) -> bool:
    import numpy as np
    if isinstance(val, bool):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    return False


def _format_retrieved_for_display(retrieved: list) -> str:
    if not retrieved:
        return "No context retrieved."

    lines = []
    for i, item in enumerate(retrieved):
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                doc = str(item[0]) if item[0] else ""
                meta = item[1] or {}
                dist = float(item[2]) if len(item) > 2 and item[2] else 0.0
            elif isinstance(item, dict):
                doc = str(item.get("text", ""))
                meta = item.get("meta", {}) or {}
                dist = item.get("dist", 0.0)
            else:
                doc, meta, dist = str(item), {}, 0.0
        except Exception:
            doc, meta, dist = str(item), {}, 0.0

        src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
        snippet = doc[:800].replace("\n", " ").strip()
        lines.append(f"[{i}] **source:** {src} | **dist:** {dist:.4f}\n{snippet}...")
    return "\n\n".join(lines)


def safe_relative_path(file_path: str) -> Optional[str]:
    if not file_path:
        return None
    p = Path(file_path).resolve()
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def _fix_latex_for_gradio(text: str) -> str:
    if not text:
        return text

    text = text.replace("\\\\", "\\")
    import re
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


def _wav_bytes_from_numpy(sr: int, arr: "np.ndarray") -> bytes:
    """
    Encode a numpy array (float in -1..1 or int16) into WAV bytes in-memory.
    Returns bytes ready for in-memory STT consumption or writing to file if needed.
    """
    import numpy as np
    buf = io.BytesIO()

    # Convert float -> int16 if necessary
    if arr.dtype.kind == "f":
        # ensure range
        max_abs = max(1e-9, float(np.max(np.abs(arr))))
        arr16 = (arr / max_abs * 32767).astype("int16")
    elif arr.dtype == np.int16:
        arr16 = arr
    else:
        # coerce
        arr16 = arr.astype("int16")

    if wavfile is None:
        raise RuntimeError("scipy.io.wavfile is required for in-memory WAV encoding.")

    # scipy.io.wavfile.write accepts file-like objects in recent SciPy versions.
    wavfile.write(buf, int(sr), arr16)
    buf.seek(0)
    return buf.read()


def _numpy_from_wav_bytes(wav_bytes: bytes) -> Tuple[int, "np.ndarray"]:
    """
    Decode WAV bytes into (sr, float32 np.array in -1..1).
    Uses scipy.io.wavfile.read on an in-memory buffer.
    """
    import numpy as np
    if wavfile is None:
        raise RuntimeError("scipy.io.wavfile is required for WAV decoding.")
    buf = io.BytesIO(wav_bytes)
    sr, arr = wavfile.read(buf)
    arr = np.asarray(arr)
    # convert int16 -> float32 [-1,1]
    if arr.dtype == np.int16:
        arr_f = (arr.astype("float32") / 32767.0)
    else:
        # if already float, just cast
        arr_f = arr.astype("float32")
    return int(sr), arr_f


def _write_safe_temp_file(output_dir: Path, prefix: str, data: bytes) -> str:
    """
    Write bytes to a uniquely-named file in output_dir and return the path.
    We use OUTPUT_DIR so it's easy to manage/inspect cleanup.
    """
    fname = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    fpath = output_dir / fname
    try:
        with open(fpath, "wb") as fh:
            fh.write(data)
        return str(fpath)
    except Exception as e:
        warn(f"Failed to write fallback temp file {fpath}: {e}")
        raise


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def build_gradio_app(agent: LocalRAGAgent, title: str = "Local Voice-RAG (In-Memory Audio)") -> Any:
    if gr is None:
        die("Gradio not installed. Install with `pip install gradio`.")

    chat_history: List[Dict[str, str]] = []

    OUTPUT_DIR = Path(CONFIG.get("output_dir", "output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.json"

    # ---------------- TEXT HANDLER ----------------
    def handle_text_submit(user_text: str, tts_enabled, top_k,
                           use_wiki, use_arxiv, use_tavily):
        nonlocal chat_history
        user_text = (user_text or "").strip()
        if not user_text:
            return chat_history, "Please provide a question.", None, ""

        wiki_flag = _to_bool(use_wiki)
        arxiv_flag = _to_bool(use_arxiv)
        tavily_flag = _to_bool(use_tavily)
        tts_flag = _to_bool(tts_enabled)

        answer, rag_sources, _ = agent.answer_text(
            user_text,
            top_k=int(top_k),
            speak=False,  # TTS handled separately (in-memory)
            use_wiki=wiki_flag,
            use_arxiv=arxiv_flag,
            use_tavily=tavily_flag
        )

        answer = _fix_latex_for_gradio(answer)

        # Attempt in-memory TTS (preferable)
        tts_out = None
        try:
            if tts_flag and answer:
                # preferred: speak_to_array or synthesize_to_array returning (sr, np_array_float)
                if hasattr(agent.tts, "speak_to_array"):
                    sr, arr = agent.tts.speak_to_array(answer)
                    tts_out = (int(sr), arr.astype("float32"))
                elif hasattr(agent.tts, "speak_to_numpy") or hasattr(agent.tts, "synthesize_to_array"):
                    fn = getattr(agent.tts, "speak_to_numpy", None) or getattr(agent.tts, "synthesize_to_array", None)
                    sr, arr = fn(answer)
                    tts_out = (int(sr), arr.astype("float32"))
                else:
                    # fallback: speak_to_file -> read into memory
                    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
                    fpath = OUTPUT_DIR / fname
                    agent.tts.speak_to_file(answer, str(fpath))
                    # read back into numpy
                    sr_read, arr_read = None, None
                    if wavfile is not None:
                        sr_read, arr_read = wavfile.read(str(fpath))
                        import numpy as np
                        if arr_read.dtype == "int16":
                            arr_read = (arr_read.astype("float32") / 32767.0)
                        tts_out = (int(sr_read), arr_read.astype("float32"))
                        # record fallback file for cleanup next time
                        global  LAST_TTS_FILE
                        # Use global variable declared at top-level; but assign via module-level
                        # We'll store it in OUTPUT_DIR as a string
                        # We'll re-use LAST_TTS_FILE in voice handler too for safe cleanup
                        try:
                            # set global
                            globals()["LAST_TTS_FILE"] = str(fpath)
                        except Exception:
                            pass
                    else:
                        warn("scipy.io.wavfile not found; cannot read fallback TTS file.")
        except Exception as e:
            warn(f"TTS (in-memory) failed: {e}")
            tts_out = None

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

        voice_path = safe_relative_path(str(tts_out) if tts_out else None)
        history.add_interaction(
            user_text, answer, rag_sources=rag_sources,
            voice_output_path=voice_path
        )

        # tts_out is either (sr, np_array) or None
        return chat_history, contexts_str, tts_out, ""

    # ---------------- VOICE HANDLER ----------------
    def handle_voice_submit(audio_in, tts_enabled, top_k,
                            use_wiki, use_arxiv, use_tavily):
        nonlocal chat_history
        global LAST_TEMP_AUDIO_FILE, LAST_TTS_FILE

        if audio_in is None:
            return chat_history, "No audio provided.", None, ""

        # audio_in: (sr, np_array)
        try:
            sr, arr = audio_in
        except Exception:
            return chat_history, "Unsupported audio input format. Expecting (sr, np.ndarray).", None, ""

        # Normalize array to float32 in [-1,1]
        import numpy as np
        arr = np.asarray(arr)
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            # If integer, convert to float in [-1,1] (assume int16 range if ints)
            if np.issubdtype(arr.dtype, np.integer):
                max_val = float(np.iinfo(arr.dtype).max)
                arr = arr.astype("float32") / max(1.0, max_val)
            else:
                arr = arr.astype("float32")
        else:
            arr = arr.astype("float32")
        # avoid silence errors
        if arr.size == 0:
            return chat_history, "Empty audio provided.", None, ""

        # --- Safe cleanup of previous fallback files (ONLY NOW) ---
        if LAST_TEMP_AUDIO_FILE and os.path.exists(LAST_TEMP_AUDIO_FILE):
            try:
                os.remove(LAST_TEMP_AUDIO_FILE)
            except Exception:
                pass
        LAST_TEMP_AUDIO_FILE = None

        if LAST_TTS_FILE and os.path.exists(LAST_TTS_FILE):
            try:
                os.remove(LAST_TTS_FILE)
            except Exception:
                pass
        LAST_TTS_FILE = None

        # --- Prepare WAV bytes for in-memory STT ---
        try:
            wav_bytes = _wav_bytes_from_numpy(int(sr), arr)
        except Exception as e:
            warn(f"Failed to encode WAV bytes: {e}")
            return chat_history, f"Failed to prepare audio for transcription: {e}", None, ""

        # --- Try in-memory transcription ---
        transcription = None
        try:
            # Try common in-memory transcribe method names in preference order
            if hasattr(agent.stt, "transcribe_bytes"):
                transcription = agent.stt.transcribe_bytes(wav_bytes)
            elif hasattr(agent.stt, "transcribe_audio_bytes"):
                transcription = agent.stt.transcribe_audio_bytes(wav_bytes)
            elif hasattr(agent.stt, "transcribe_array"):
                transcription = agent.stt.transcribe_array(int(sr), arr)
            elif hasattr(agent.stt, "transcribe"):
                # some implementations accept (bytes) or (sr, np_array)
                fn = getattr(agent.stt, "transcribe")
                try:
                    transcription = fn(wav_bytes)
                except Exception:
                    try:
                        transcription = fn(int(sr), arr)
                    except Exception:
                        transcription = None
            else:
                transcription = None
        except Exception as e:
            warn(f"In-memory transcription attempt failed: {e}")
            transcription = None

        # --- Fallback: write a temp file and ask file-based transcribe ---
        if not transcription:
            try:
                fallback_path = _write_safe_temp_file(OUTPUT_DIR, "gr_inmem_fallback", wav_bytes)
                # record for cleanup on the next new recording
                LAST_TEMP_AUDIO_FILE = fallback_path
                # try file-based transcribe API
                if hasattr(agent.stt, "transcribe_file"):
                    transcription = agent.stt.transcribe_file(fallback_path)
                elif hasattr(agent.stt, "transcribe_from_file"):
                    transcription = agent.stt.transcribe_from_file(fallback_path)
                else:
                    # give up gracefully
                    raise RuntimeError("STT component has no recognized transcribe API.")
            except Exception as e:
                warn(f"Fallback file-based transcription failed: {e}")
                return chat_history, f"Could not transcribe audio: {e}", None, ""

        if not transcription:
            return chat_history, "Could not transcribe audio.", None, ""

        wiki_flag = _to_bool(use_wiki)
        arxiv_flag = _to_bool(use_arxiv)
        tavily_flag = _to_bool(use_tavily)
        tts_flag = _to_bool(tts_enabled)

        # Ask the agent for an answer (text)
        try:
            answer, rag_sources, _ = agent.answer_text(
                transcription,
                top_k=int(top_k),
                speak=False,
                use_wiki=wiki_flag,
                use_arxiv=arxiv_flag,
                use_tavily=tavily_flag
            )
        except Exception as e:
            warn(f"agent.answer_text failed: {e}")
            return chat_history, f"Failed generating answer: {e}", None, ""

        answer = _fix_latex_for_gradio(answer)

        # --- Generate TTS in-memory if requested ---
        tts_out = None
        if tts_flag and answer:
            try:
                # Preferred: agent.tts.speak_to_array or similar -> return (sr, np_array)
                if hasattr(agent.tts, "speak_to_array"):
                    sr_t, arr_t = agent.tts.speak_to_array(answer)
                    tts_out = (int(sr_t), arr_t.astype("float32"))
                elif hasattr(agent.tts, "speak_to_numpy") or hasattr(agent.tts, "synthesize_to_array"):
                    fn = getattr(agent.tts, "speak_to_numpy", None) or getattr(agent.tts, "synthesize_to_array", None)
                    sr_t, arr_t = fn(answer)
                    tts_out = (int(sr_t), arr_t.astype("float32"))
                else:
                    # fallback: file-based TTS -> read into memory and register fallback for cleanup
                    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
                    fpath = OUTPUT_DIR / fname
                    # assume speak_to_file(text, path) signature
                    if hasattr(agent.tts, "speak_to_file"):
                        agent.tts.speak_to_file(answer, str(fpath))
                    elif hasattr(agent.tts, "synthesize_to_file"):
                        agent.tts.synthesize_to_file(answer, str(fpath))
                    else:
                        raise RuntimeError("TTS component has no recognized speak_to_array or speak_to_file API.")

                    # read back into numpy if scipy available
                    if wavfile is not None:
                        sr_read, arr_read = wavfile.read(str(fpath))
                        if arr_read.dtype == "int16":
                            arr_read = (arr_read.astype("float32") / 32767.0)
                        tts_out = (int(sr_read), arr_read.astype("float32"))
                        # mark fallback file for cleanup next invocation
                        try:
                            globals()["LAST_TTS_FILE"] = str(fpath)
                        except Exception:
                            pass
                    else:
                        warn("scipy.io.wavfile not found; cannot read fallback TTS file into memory.")
                        tts_out = None
            except Exception as e:
                warn(f"TTS (in-memory) failed: {e}")
                tts_out = None

        contexts_str = _format_retrieved_for_display(rag_sources)

        chat_history.extend([
            {"role": "user", "content": f"[Voice] {transcription}"},
            {"role": "assistant", "content": answer}
        ])
        chat_history = chat_history[-40:]

        voice_path = safe_relative_path(str(tts_out) if tts_out else None)
        history.add_interaction(
            transcription, answer, rag_sources=rag_sources,
            voice_output_path=voice_path
        )

        # Return: chat history, contexts, in-memory TTS output (sr,nparray) or None, status message
        return chat_history, contexts_str, tts_out, ""

    # -------------------------------------------------------------------
    # UI Layout (use numpy audio type for both input and output)
    # -------------------------------------------------------------------
    with gr.Blocks(title=title) as demo:

        css_content = ""
        css_file = Path("assets/style.css")
        if css_file.exists():
            with open(css_file, "r", encoding="utf-8") as f:
                css_content = f.read()

        gr.HTML(f"<style>{css_content}</style>")

        gr.Markdown("## 🎧 **Local Voice-RAG — In-Memory Audio**")

        with gr.Tabs():

            # ======================================
            # TEXT CHAT TAB
            # ======================================
            with gr.Tab("💬 Text Chat"):

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(
                            label="💬 Conversation",
                            height=460,
                            render_markdown=True,
                        )
                        txt_in = gr.Textbox(
                            label="✏️ Your Question",
                            placeholder="Type your question here...",
                            lines=2
                        )

                    with gr.Column(scale=2):
                        contexts = gr.Textbox(
                            label="📑 Retrieved Sources",
                            interactive=False,
                            lines=20
                        )
                        # NOTE: numpy audio for TTS output
                        audio_out = gr.Audio(
                            label="🔊 TTS Audio Output",
                            interactive=False,
                            type="numpy"
                        )
                        status = gr.Textbox(
                            label="Status / Log",
                            interactive=False,
                            lines=4
                        )

                # ---- Control Panel ----
                with gr.Group():
                    gr.Markdown("### ⚙️ **Controls**")

                    with gr.Row():
                        tts_flag_text = gr.Checkbox(label="🔉 Enable TTS", value=False)
                        topk_text = gr.Slider(
                            1, CONFIG.get("max_retrieval_topk", 8),
                            step=1, value=4,
                            label="📚 Retrieval Top-K"
                        )

                    with gr.Row():
                        wiki_flag_text = gr.Checkbox(label="🌐 Use Wikipedia", value=False)
                        arxiv_flag_text = gr.Checkbox(label="📄 Use arXiv", value=False)
                        tavily_flag_text = gr.Checkbox(label="🔎 Use Tavily (API key required)", value=False)

                    send_text = gr.Button("🚀 Send", elem_classes="compact-button")

                # bind events
                send_text.click(
                    lambda t, tt, tk, w, a, tv: handle_text_submit(t, tt, int(tk), w, a, tv),
                    inputs=[txt_in, tts_flag_text, topk_text, wiki_flag_text, arxiv_flag_text, tavily_flag_text],
                    outputs=[chat, contexts, audio_out, status]
                )
                txt_in.submit(
                    lambda t, tt, tk, w, a, tv: handle_text_submit(t, tt, int(tk), w, a, tv),
                    inputs=[txt_in, tts_flag_text, topk_text, wiki_flag_text, arxiv_flag_text, tavily_flag_text],
                    outputs=[chat, contexts, audio_out, status]
                )

            # ======================================
            # VOICE CHAT TAB
            # ======================================
            with gr.Tab("🎤 Voice Chat"):

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        voice_chat = gr.Chatbot(
                            label="🎤 Conversation (Voice)",
                            height=460,
                            render_markdown=True
                        )
                        audio_in = gr.Audio(
                            label="🎙️ Record or Upload Audio",
                            type="numpy",
                            interactive=True
                        )

                    with gr.Column(scale=2):
                        v_contexts = gr.Textbox(
                            label="📑 Retrieved Sources",
                            interactive=False,
                            lines=20
                        )
                        # numpy TTS output here too
                        v_audio_out = gr.Audio(
                            label="🔊 TTS Audio Output",
                            interactive=False,
                            type="numpy"
                        )
                        v_status = gr.Textbox(
                            label="Status / Log",
                            interactive=False,
                            lines=4
                        )

                # ---- Control Panel ----
                with gr.Group():
                    gr.Markdown("### ⚙️ **Controls**")

                    with gr.Row():
                        tts_flag_voice = gr.Checkbox(label="🔉 Enable TTS", value=False)
                        topk_voice = gr.Slider(
                            1, CONFIG.get("max_retrieval_topk", 8),
                            step=1, value=4,
                            label="📚 Retrieval Top-K"
                        )

                    with gr.Row():
                        wiki_flag_voice = gr.Checkbox(label="🌐 Use Wikipedia", value=False)
                        arxiv_flag_voice = gr.Checkbox(label="📄 Use arXiv", value=False)
                        tavily_flag_voice = gr.Checkbox(label="🔎 Use Tavily (API key required)", value=False)

                    send_voice = gr.Button("🚀 Send", elem_classes="compact-button")

                send_voice.click(
                    lambda a, tt, tk, w, ar, tv: handle_voice_submit(a, tt, int(tk), w, ar, tv),
                    inputs=[audio_in, tts_flag_voice, topk_voice, wiki_flag_voice, arxiv_flag_voice, tavily_flag_voice],
                    outputs=[voice_chat, v_contexts, v_audio_out, v_status]
                )
                audio_in.change(
                    lambda a, tt, tk, w, ar, tv: handle_voice_submit(a, tt, int(tk), w, ar, tv),
                    inputs=[audio_in, tts_flag_voice, topk_voice, wiki_flag_voice, arxiv_flag_voice, tavily_flag_voice],
                    outputs=[voice_chat, v_contexts, v_audio_out, v_status]
                )

            # ======================================
            # ABOUT TAB
            # ======================================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(""" 
### ℹ️ **About Local Voice-RAG (In-Memory)**
- Input and output audio are handled entirely in memory using NumPy arrays (no disk I/O in normal flow).
- If your STT/TTS backend lacks in-memory APIs, a safe fallback to disk is used (files written to `output/` and cleaned up only when new audio is recorded).
- Fully supports: **STT**, **TTS**, **chat history**, **retrieval context visualization**  
""")

        gr.Markdown("---")
        gr.Markdown("*All processing happens locally except optional online retrieval.*")

    return demo


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def launch_gradio_app(agent: Any, ui_title: str = "Local Voice-RAG (In-Memory)") -> None:
    import asyncio
    import sys

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if gr is None:
        die("Gradio not installed.")

    app = build_gradio_app(agent, title=ui_title)
    app.launch(share=False, server_name="127.0.0.1", server_port=7861)


# """
# Gradio UI for Local Voice-RAG
# Enhanced for readability, consistent audio handling, and Windows-safe streaming.
# """
#
# from typing import Any, List, Dict
# import os
# import time
# import uuid
# from pathlib import Path
#
# from voice_rag.config import CONFIG
# from voice_rag.utils import die, warn
# from voice_rag.agent import LocalRAGAgent
# from voice_rag.history import ChatHistory
#
# history = ChatHistory()
#
# # NEW: global holder for safe cleanup
# LAST_TEMP_AUDIO = None
#
# try:
#     import gradio as gr
# except Exception:
#     gr = None
#
# try:
#     from scipy.io import wavfile
# except Exception:
#     wavfile = None
#
#
# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def _to_bool(val) -> bool:
#     import numpy as np
#     if isinstance(val, bool):
#         return val
#     if isinstance(val, np.bool_):
#         return bool(val)
#     return False
#
#
# def _format_retrieved_for_display(retrieved: list) -> str:
#     if not retrieved:
#         return "No context retrieved."
#
#     lines = []
#     for i, item in enumerate(retrieved):
#         try:
#             if isinstance(item, (list, tuple)) and len(item) >= 2:
#                 doc = str(item[0]) if item[0] else ""
#                 meta = item[1] or {}
#                 dist = float(item[2]) if len(item) > 2 and item[2] else 0.0
#             elif isinstance(item, dict):
#                 doc = str(item.get("text", ""))
#                 meta = item.get("meta", {}) or {}
#                 dist = item.get("dist", 0.0)
#             else:
#                 doc, meta, dist = str(item), {}, 0.0
#         except Exception:
#             doc, meta, dist = str(item), {}, 0.0
#
#         src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
#         snippet = doc[:800].replace("\n", " ").strip()
#         lines.append(f"[{i}] **source:** {src} | **dist:** {dist:.4f}\n{snippet}...")
#     return "\n\n".join(lines)
#
#
# def safe_relative_path(file_path: str) -> str:
#     if not file_path:
#         return None
#     p = Path(file_path).resolve()
#     try:
#         return str(p.relative_to(Path.cwd()))
#     except ValueError:
#         return str(p)
#
#
# def _fix_latex_for_gradio(text: str) -> str:
#     if not text:
#         return text
#
#     text = text.replace("\\\\", "\\")
#     import re
#     text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
#     text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
#     return text
#
#
# # ---------------------------------------------------------------------------
# # Main App
# # ---------------------------------------------------------------------------
# def build_gradio_app(agent: LocalRAGAgent, title: str = "Local Voice-RAG (Optimized CPU)") -> Any:
#     if gr is None:
#         die("Gradio not installed. Install with `pip install gradio`.")
#
#     chat_history: List[Dict[str, str]] = []
#
#     OUTPUT_DIR = Path(CONFIG.get("output_dir", "output"))
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.json"
#
#     # ---------------- TEXT HANDLER ----------------
#     def handle_text_submit(user_text: str, tts_enabled, top_k,
#                            use_wiki, use_arxiv, use_tavily):
#         nonlocal chat_history
#         user_text = (user_text or "").strip()
#         if not user_text:
#             return chat_history, "Please provide a question.", None, ""
#
#         wiki_flag = _to_bool(use_wiki)
#         arxiv_flag = _to_bool(use_arxiv)
#         tavily_flag = _to_bool(use_tavily)
#         tts_flag = _to_bool(tts_enabled)
#
#         answer, rag_sources, _ = agent.answer_text(
#             user_text,
#             top_k=int(top_k),
#             speak=False,
#             use_wiki=wiki_flag,
#             use_arxiv=arxiv_flag,
#             use_tavily=tavily_flag
#         )
#
#         answer = _fix_latex_for_gradio(answer)
#
#         voice_file = None
#         if tts_flag and answer:
#             try:
#                 safe_text = agent.clean_for_tts(answer)
#                 fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
#                 fpath = OUTPUT_DIR / fname
#                 agent.tts.speak_to_file(safe_text, str(fpath))
#                 voice_file = fpath
#             except Exception as e:
#                 warn(f"TTS generation failed: {e}")
#                 voice_file = None
#
#         contexts_str = _format_retrieved_for_display(rag_sources)
#
#         chat_history.extend([
#             {"role": "user", "content": user_text},
#             {"role": "assistant", "content": answer}
#         ])
#         chat_history = chat_history[-40:]
#
#         try:
#             import json
#             with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#                 json.dump(chat_history, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             warn(f"Failed to save chat history: {e}")
#
#         voice_path = safe_relative_path(str(voice_file) if voice_file else None)
#         history.add_interaction(
#             user_text, answer, rag_sources=rag_sources,
#             voice_output_path=voice_path
#         )
#
#         return chat_history, contexts_str, str(Path(voice_file).resolve()) if voice_file else None, ""
#
#     # ---------------- VOICE HANDLER ----------------
#     def handle_voice_submit(audio_in, tts_enabled, top_k,
#                             use_wiki, use_arxiv, use_tavily):
#         nonlocal chat_history
#         global LAST_TEMP_AUDIO
#
#         if not audio_in:
#             return chat_history, "No audio provided.", None, ""
#
#         tmp_path = None
#         try:
#             # SAFE PATCH: Clean up previous temp file ONLY NOW
#             if LAST_TEMP_AUDIO and os.path.exists(LAST_TEMP_AUDIO):
#                 try:
#                     os.remove(LAST_TEMP_AUDIO)
#                 except Exception:
#                     pass
#             LAST_TEMP_AUDIO = None
#
#             # Handle Gradio audio input
#             if isinstance(audio_in, str) and os.path.exists(audio_in):
#                 tmp_path = audio_in
#
#             elif isinstance(audio_in, (tuple, list)) and len(audio_in) == 2:
#                 sr, arr = audio_in
#                 import numpy as np
#                 arr = np.asarray(arr)
#                 arr16 = (
#                     (np.clip(arr, -1.0, 1.0) * 32767).astype("int16")
#                     if arr.dtype.kind == "f" else arr.astype("int16")
#                 )
#                 tmp_path = f"gr_audio_{int(time.time())}.wav"
#                 wavfile.write(tmp_path, int(sr), arr16)
#
#                 # remember for next cleanup
#                 LAST_TEMP_AUDIO = tmp_path
#
#             else:
#                 return chat_history, "Unsupported audio input.", None, ""
#
#             transcription = agent.stt.transcribe_file(tmp_path)
#             if not transcription:
#                 return chat_history, "Could not transcribe audio.", None, ""
#
#             wiki_flag = _to_bool(use_wiki)
#             arxiv_flag = _to_bool(use_arxiv)
#             tavily_flag = _to_bool(use_tavily)
#             tts_flag = _to_bool(tts_enabled)
#
#             answer, rag_sources, _ = agent.answer_text(
#                 transcription,
#                 top_k=int(top_k),
#                 speak=False,
#                 use_wiki=wiki_flag,
#                 use_arxiv=arxiv_flag,
#                 use_tavily=tavily_flag
#             )
#
#             answer = _fix_latex_for_gradio(answer)
#
#             voice_file = None
#             if tts_flag and answer:
#                 try:
#                     safe_text = agent.clean_for_tts(answer)
#                     fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
#                     voice_file = OUTPUT_DIR / fname
#                     agent.tts.speak_to_file(safe_text, str(voice_file))
#                 except Exception as e:
#                     warn(f"TTS generation failed: {e}")
#                     voice_file = None
#
#             contexts_str = _format_retrieved_for_display(rag_sources)
#
#             chat_history.extend([
#                 {"role": "user", "content": f"[Voice] {transcription}"},
#                 {"role": "assistant", "content": answer}
#             ])
#             chat_history = chat_history[-40:]
#
#             voice_path = safe_relative_path(str(voice_file) if voice_file else None)
#             history.add_interaction(
#                 transcription, answer, rag_sources=rag_sources,
#                 voice_output_path=voice_path
#             )
#
#             return chat_history, contexts_str, str(Path(voice_file).resolve()) if voice_file else None, ""
#
#         except Exception as e:
#             warn(f"Voice submit error: {e}")
#             return chat_history, f"Error processing audio: {e}", None, ""
#
#     # -------------------------------------------------------------------
#     # UI Layout
#     # -------------------------------------------------------------------
#     with gr.Blocks(title=title) as demo:
#
#         css_content = ""
#         css_file = Path("assets/style.css")
#         if css_file.exists():
#             with open(css_file, "r", encoding="utf-8") as f:
#                 css_content = f.read()
#
#         gr.HTML(f"<style>{css_content}</style>")
#
#         gr.Markdown("## 🎧 **Local Voice-RAG — CPU-Optimized**")
#
#         with gr.Tabs():
#
#             # ======================================
#             # TEXT CHAT TAB
#             # ======================================
#             with gr.Tab("💬 Text Chat"):
#
#                 with gr.Row(equal_height=True):
#                     with gr.Column(scale=3):
#                         chat = gr.Chatbot(
#                             label="💬 Conversation",
#                             height=460,
#                             render_markdown=True,
#                         )
#                         txt_in = gr.Textbox(
#                             label="✏️ Your Question",
#                             placeholder="Type your question here...",
#                             lines=2
#                         )
#
#                     with gr.Column(scale=2):
#                         contexts = gr.Textbox(
#                             label="📑 Retrieved Sources",
#                             interactive=False,
#                             lines=20
#                         )
#                         audio_out = gr.Audio(
#                             label="🔊 TTS Audio Output",
#                             interactive=False
#                         )
#                         status = gr.Textbox(
#                             label="Status / Log",
#                             interactive=False,
#                             lines=4
#                         )
#
#                 with gr.Group():
#                     gr.Markdown("### ⚙️ **Controls**")
#
#                     with gr.Row():
#                         tts_flag_text = gr.Checkbox(label="🔉 Enable TTS", value=False)
#                         topk_text = gr.Slider(
#                             1, CONFIG.get("max_retrieval_topk", 8),
#                             step=1, value=4,
#                             label="📚 Retrieval Top-K"
#                         )
#
#                     with gr.Row():
#                         wiki_flag_text = gr.Checkbox(label="🌐 Use Wikipedia", value=False)
#                         arxiv_flag_text = gr.Checkbox(label="📄 Use arXX", value=False)
#                         tavily_flag_text = gr.Checkbox(label="🔎 Use Tavily (API key required)", value=False)
#
#                     send_text = gr.Button("🚀 Send", elem_classes="compact-button")
#
#                 send_text.click(
#                     lambda t, tt, tk, w, a, tv: handle_text_submit(t, tt, int(tk), w, a, tv),
#                     inputs=[txt_in, tts_flag_text, topk_text, wiki_flag_text, arxiv_flag_text, tavily_flag_text],
#                     outputs=[chat, contexts, audio_out, status]
#                 )
#                 txt_in.submit(
#                     lambda t, tt, tk, w, a, tv: handle_text_submit(t, tt, int(tk), w, a, tv),
#                     inputs=[txt_in, tts_flag_text, topk_text, wiki_flag_text, arxiv_flag_text, tavily_flag_text],
#                     outputs=[chat, contexts, audio_out, status]
#                 )
#
#             # ======================================
#             # VOICE CHAT TAB
#             # ======================================
#             with gr.Tab("🎤 Voice Chat"):
#
#                 with gr.Row(equal_height=True):
#                     with gr.Column(scale=3):
#                         voice_chat = gr.Chatbot(
#                             label="🎤 Conversation (Voice)",
#                             height=460,
#                             render_markdown=True
#                         )
#                         audio_in = gr.Audio(
#                             label="🎙️ Record or Upload Audio",
#                             type="filepath",
#                             interactive=True
#                         )
#
#                     with gr.Column(scale=2):
#                         v_contexts = gr.Textbox(
#                             label="📑 Retrieved Sources",
#                             interactive=False,
#                             lines=20
#                         )
#                         v_audio_out = gr.Audio(
#                             label="🔊 TTS Audio Output",
#                             interactive=False
#                         )
#                         v_status = gr.Textbox(
#                             label="Status / Log",
#                             interactive=False,
#                             lines=4
#                         )
#
#                 with gr.Group():
#                     gr.Markdown("### ⚙️ **Controls**")
#
#                     with gr.Row():
#                         tts_flag_voice = gr.Checkbox(label="🔉 Enable TTS", value=False)
#                         topk_voice = gr.Slider(
#                             1, CONFIG.get("max_retrieval_topk", 8),
#                             step=1, value=4,
#                             label="📚 Retrieval Top-K"
#                         )
#
#                     with gr.Row():
#                         wiki_flag_voice = gr.Checkbox(label="🌐 Use Wikipedia", value=False)
#                         arxiv_flag_voice = gr.Checkbox(label="📄 Use arXiv", value=False)
#                         tavily_flag_voice = gr.Checkbox(label="🔎 Use Tavily (API key required)", value=False)
#
#                     send_voice = gr.Button("🚀 Send", elem_classes="compact-button")
#
#                 send_voice.click(
#                     lambda a, tt, tk, w, ar, tv: handle_voice_submit(a, tt, int(tk), w, ar, tv),
#                     inputs=[audio_in, tts_flag_voice, topk_voice, wiki_flag_voice, arxiv_flag_voice, tavily_flag_voice],
#                     outputs=[voice_chat, v_contexts, v_audio_out, v_status]
#                 )
#                 audio_in.change(
#                     lambda a, tt, tk, w, ar, tv: handle_voice_submit(a, tt, int(tk), w, ar, tv),
#                     inputs=[audio_in, tts_flag_voice, topk_voice, wiki_flag_voice, arxiv_flag_voice, tavily_flag_voice],
#                     outputs=[voice_chat, v_contexts, v_audio_out, v_status]
#                 )
#
#             with gr.Tab("ℹ️ About"):
#                 gr.Markdown("""
# ### ℹ️ **About Local Voice-RAG**
# - Optional retrieval sources: **Wikipedia**, **arXiv**, **Tavily**
# - Tavily auto-disabled when no `TAVILY_API_KEY` is present
# - Fully supports: **STT**, **TTS**, **chat history**, **retrieval context visualization**
# """)
#
#         gr.Markdown("---")
#         gr.Markdown("*All processing happens locally except optional online retrieval.*")
#
#     return demo
#
#
# def launch_gradio_app(agent: Any, ui_title: str = "Local Voice-RAG (CPU)") -> None:
#     import asyncio
#     import sys
#
#     if sys.platform.startswith("win"):
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#
#     if gr is None:
#         die("Gradio not installed.")
#
#     app = build_gradio_app(agent, title=ui_title)
#     app.launch(share=False, server_name="127.0.0.1", server_port=7861)
