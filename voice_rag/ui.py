from typing import Any, List, Tuple, Dict
from .config import CONFIG
from .utils import die, warn
import os, time
from voice_rag.agent import LocalRAGAgent

try:
    import gradio as gr
except Exception:
    gr = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None


def _format_retrieved_for_display(retrieved: List[Tuple[str, dict, float]]) -> str:
    if not retrieved:
        return "No context retrieved."
    lines = []
    for i, (doc, meta, dist) in enumerate(retrieved):
        src = meta.get("source", "unknown")
        snippet = doc[:800].replace("\n", " ").strip()
        lines.append(f"[{i}] source={src} | dist={dist:.4f}\n{snippet}...")
    return "\n\n".join(lines)


def build_gradio_app(agent: LocalRAGAgent, title: str = "Local Voice-RAG (Optimized CPU)") -> Any:
    if gr is None:
        die("Gradio not installed. Install with `pip install gradio`.")

    chat_history: List[Dict[str, str]] = []

    # -----------------------------
    # Handle text input
    # -----------------------------
    def handle_text_submit(user_text: str, tts_enabled: bool, top_k: int):
        nonlocal chat_history
        user_text = (user_text or "").strip()
        if not user_text:
            return chat_history, "Please provide a question.", None, None

        retrieved = agent.index.query(user_text, top_k=top_k)
        contexts_display = _format_retrieved_for_display(retrieved)
        answer = agent.answer_text(user_text, top_k=top_k, speak=False)

        audio_file = None
        if tts_enabled:
            try:
                safe_text = agent.clean_for_tts(answer)
                audio_file = agent.tts.speak_to_file(safe_text)
            except Exception as e:
                warn(f"TTS generation failed: {e}")

        chat_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer}
        ])
        chat_history = chat_history[-40:]  # keep history manageable
        return chat_history, contexts_display, audio_file, ""


    # -----------------------------
    # Handle voice input
    # -----------------------------
    def handle_voice_submit(audio_in, tts_enabled: bool, top_k: int):
        nonlocal chat_history
        if not audio_in:
            return chat_history, "No audio provided.", None, None

        tmp_path: str = ""
        try:
            # Gradio audio input can be filepath or (sr, array)
            if isinstance(audio_in, str) and os.path.exists(audio_in):
                tmp_path = audio_in
            elif isinstance(audio_in, (tuple, list)) and len(audio_in) == 2:
                sr, arr = audio_in
                import numpy as np
                arr_np = np.asarray(arr)
                arr_int16 = (np.clip(arr_np, -1.0, 1.0) * 32767).astype(np.int16) if arr_np.dtype.kind == 'f' else arr_np.astype(np.int16)
                tmp_path = f"gr_audio_{int(time.time())}.wav"
                wavfile.write(tmp_path, int(sr), arr_int16)
            else:
                return chat_history, "Unsupported audio input.", None, None

            transcription = agent.stt.transcribe_file(tmp_path)
            if not transcription:
                return chat_history, "Could not transcribe audio.", None, None

            retrieved = agent.index.query(transcription, top_k=top_k)
            contexts_display = _format_retrieved_for_display(retrieved)
            answer = agent.answer_text(transcription, top_k=top_k, speak=False)

            audio_file = None
            if tts_enabled:
                try:
                    safe_text = agent.clean_for_tts(answer)
                    audio_file = agent.tts.speak_to_file(safe_text)
                except Exception as e:
                    warn(f"TTS generation failed: {e}")

            chat_history.extend([
                {"role": "user", "content": f"[Voice] {transcription}"},
                {"role": "assistant", "content": answer}
            ])
            chat_history = chat_history[-40:]
            return chat_history, contexts_display, audio_file, ""
        finally:
            try:
                if tmp_path and tmp_path.startswith("gr_audio_") and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


    # -----------------------------
    # Build Gradio UI
    # -----------------------------
    with gr.Blocks(title=title) as demo:
        gr.Markdown("## Local Voice-RAG — Optimized for CPU (Python 3.11)")

        with gr.Tabs():
            # Text Chat Tab
            with gr.Tab("Text Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(label="Conversation", elem_id="chatbot", height=400)
                        txt_input = gr.Textbox(label="Your question", placeholder="Ask something...", lines=2)
                        with gr.Row():
                            tts_toggle = gr.Checkbox(label="Enable TTS", value=False)
                            topk_slider = gr.Slider(minimum=1, maximum=CONFIG.get("max_retrieval_topk",8), step=1, value=4, label="Retrieval top_k")
                            send_btn = gr.Button("Send")
                    with gr.Column(scale=2):
                        contexts_box = gr.Textbox(label="Retrieved Contexts", interactive=False, lines=20)
                        audio_out = gr.Audio(label="TTS Audio (if generated)", interactive=False)
                        status = gr.Textbox(label="Status / Log", interactive=False, lines=4)

                send_btn.click(lambda t, tt, tk: handle_text_submit(t, tt, int(tk)),
                               inputs=[txt_input, tts_toggle, topk_slider],
                               outputs=[chat, contexts_box, audio_out, status])
                txt_input.submit(lambda t, tt, tk: handle_text_submit(t, tt, int(tk)),
                                 inputs=[txt_input, tts_toggle, topk_slider],
                                 outputs=[chat, contexts_box, audio_out, status])

            # Voice Chat Tab
            with gr.Tab("Voice Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        voice_chat = gr.Chatbot(label="Conversation (Voice)", elem_id="voice_chatbot", height=400)
                        gr.Markdown("Record in the browser or upload an audio file (wav/mp3).")
                        audio_input = gr.Audio(label="Record or upload audio", type="filepath", interactive=True)
                        with gr.Row():
                            v_tts_toggle = gr.Checkbox(label="Enable TTS", value=True)
                            v_topk = gr.Slider(minimum=1, maximum=CONFIG.get("max_retrieval_topk",8), step=1, value=4, label="Retrieval top_k")
                            v_send = gr.Button("Process Audio")
                    with gr.Column(scale=2):
                        v_contexts = gr.Textbox(label="Retrieved Contexts", interactive=False, lines=20)
                        v_audio_out = gr.Audio(label="TTS Audio (if generated)", interactive=False)
                        v_status = gr.Textbox(label="Status / Log", interactive=False, lines=4)

                v_send.click(lambda a, tt, tk: handle_voice_submit(a, tt, int(tk)),
                             inputs=[audio_input, v_tts_toggle, v_topk],
                             outputs=[voice_chat, v_contexts, v_audio_out, v_status])
                audio_input.change(lambda a, tt, tk: handle_voice_submit(a, tt, int(tk)),
                                   inputs=[audio_input, v_tts_toggle, v_topk],
                                   outputs=[voice_chat, v_contexts, v_audio_out, v_status])

        gr.Markdown("Made with ❤️ — Local RAG demo. Ensure Ollama & models are running.")

    return demo


def launch_gradio_app(agent: Any, ui_title: str = "Local Voice-RAG (CPU)") -> None:
    import asyncio, sys
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if gr is None:
        die("Gradio not installed.")
    app = build_gradio_app(agent, title=ui_title)
    app.launch(share=False, server_name="127.0.0.1", server_port=7861)
