from dataclasses import dataclass
from typing import Any, Optional
from .config import CONFIG
from .utils import info, warn
import time

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


# -----------------------------
# Local TTS wrapper (Coqui)
# -----------------------------
@dataclass
class LocalTTS:
    tts_model_name: str = CONFIG["coqui_model"]
    tts: Any = None

    def init(self):
        """Initialize Coqui TTS model if not loaded."""
        if TTS is None:
            warn("Coqui TTS is not installed; TTS unavailable.")
            return
        if self.tts is None:
            info(f"Loading TTS model: {self.tts_model_name}")
            try:
                self.tts = TTS(self.tts_model_name)
            except Exception as e:
                warn(f"TTS model loading failed: {e}")
                self.tts = None

    # -----------------------------
    # Generate TTS and save to WAV file
    # -----------------------------
    def speak_to_file(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        if TTS is None:
            warn("TTS requested but Coqui TTS not available.")
            return None
        if self.tts is None:
            self.init()
        out_path = output_path or f"tts_{int(time.time())}.wav"
        try:
            self.tts.tts_to_file(text=text, file_path=out_path)
            info(f"TTS generated and saved to {out_path}")
            return out_path
        except Exception as e:
            warn(f"TTS generation failed: {e}")
            return None

    # -----------------------------
    # Generate and play TTS immediately
    # -----------------------------
    def speak(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        out_file = self.speak_to_file(text, output_path)
        if out_file and sd is not None and wavfile is not None:
            try:
                rate, data = wavfile.read(out_file)
                sd.play(data, samplerate=rate)
                sd.wait()
            except Exception as e:
                warn(f"TTS playback failed: {e}")
        return out_file
