"""
Persistent chat history management.

Stores interactions as JSON and supports export to TXT. Designed for
lightweight RAG/assistant logging with minimal assumptions and clear,
readable records.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class ChatHistory:
    """
    Manage persistent chat interactions.

    Attributes:
        path (Path): Path to the JSON log file.
        items (List[Dict]): In-memory history entries.
    """

    def __init__(self, json_filename: str = "chat_history.json") -> None:
        """
        Initialize history manager and load existing entries.

        Args:
            json_filename (str): JSON file name inside OUTPUT_DIR.
        """
        self.path = OUTPUT_DIR / json_filename
        self.items: List[Dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing history from disk if present."""
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = []

    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current history to disk."""
        try:
            self.path.write_text(
                json.dumps(self.items, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: Failed to save chat history: {e}")

    # ------------------------------------------------------------------

    def add_interaction(
            self,
            user_prompt: str,
            bot_response: str,
            rag_sources: Optional[list] = None,
            voice_output_path: Optional[str] = None,
            snippet_len: int = 50,
    ) -> None:
        """
        Add a new interaction entry.

        Args:
            user_prompt (str): User message.
            bot_response (str): Assistant message.
            rag_sources (list): Retrieved docs (raw tuples).
            voice_output_path (str): Optional generated TTS audio path.
            snippet_len (int): Number of characters used for text previews.
        """
        # Normalize voice path to internal relative reference
        voice_rel = None
        if voice_output_path:
            try:
                voice_rel = str(Path(voice_output_path).relative_to(OUTPUT_DIR))
            except ValueError:
                voice_rel = Path(voice_output_path).name

        # Condense retrievals for logs
        condensed: List[Dict[str, Any]] = []
        if rag_sources:
            for item in rag_sources:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    text, meta, dist = item
                    condensed.append(
                        {
                            "id": meta.get("id", "unknown") if isinstance(meta, dict) else "unknown",
                            "snippet": text[:snippet_len] + "...",
                            "dist": dist,
                        }
                    )
                else:
                    condensed.append(
                        {"id": "unknown", "snippet": "", "dist": None}
                    )

        # Construct entry
        entry = {
            "role": "interaction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_prompt,
            "bot": bot_response,
            #"rag_sources": condensed,
            "voice_output_path": voice_rel,
        }

        self.items.append(entry)
        self.save()

    # ------------------------------------------------------------------

    def export_txt(self, txt_filename: str = "chat_history.txt") -> None:
        """
        Export history to a flat-text transcript.

        Args:
            txt_filename (str): Name of exported TXT file.
        """
        txt_path = OUTPUT_DIR / txt_filename
        out: List[str] = []

        for h in self.items:
            if h.get("role") == "interaction":
                rag_text = (
                    "\nRAG Sources:\n" + "\n".join(map(str, h.get("rag_sources", [])))
                    if h.get("rag_sources")
                    else ""
                )
                voice_text = (
                    f"\nVoice Output: {h.get('voice_output_path')}"
                    if h.get("voice_output_path")
                    else ""
                )
                out.append(
                    f"[USER] {h['timestamp']}\n{h['user']}"
                    f"\n[BOT] {h['timestamp']}\n{h['bot']}{rag_text}{voice_text}\n"
                )
            else:
                out.append(
                    f"[{h.get('role', 'log')}] {h.get('timestamp','')}\n"
                    f"{h.get('message','')}\n"
                )

        txt_path.write_text("\n".join(out), encoding="utf-8")
