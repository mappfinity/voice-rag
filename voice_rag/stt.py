from dataclasses import dataclass
from typing import Any
from .utils import die, info, warn
from .config import CONFIG
import os

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None


# -----------------------------
# Local STT wrapper (Faster-Whisper)
# -----------------------------
@dataclass
class LocalSTT:
    model_size: str = CONFIG["whisper_model"]
    model: Any = None
    device: str = "cpu"

    def init_model(self):
        """Initialize Whisper model if not already loaded."""
        if WhisperModel is None:
            die("faster-whisper is required. Install with `pip install faster-whisper`.")
        if self.model is None:
            info(f"Loading Whisper model: {self.model_size} on {self.device}")
            compute = CONFIG.get("whisper_compute_type", None)
            try:
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=compute)
            except Exception:
                info("Requested compute_type not supported; falling back to default.")
                self.model = WhisperModel(self.model_size, device=self.device)

    # -----------------------------
    # File-based transcription
    # -----------------------------
    def transcribe_file(self, path: str) -> str:
        if self.model is None:
            self.init_model()
        info(f"Transcribing file: {path}")
        try:
            segments, _ = self.model.transcribe(
                path,
                beam_size=CONFIG.get("whisper_beam_size", 1),
                language=CONFIG.get("whisper_language", None),
                temperature=CONFIG.get("whisper_temperature", None)
            )
            text = " ".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            warn(f"Transcription failed for {path}: {e}")
            return ""

    # -----------------------------
    # Array-based transcription
    # -----------------------------
    def transcribe_audio_array(self, audio: Any, sample_rate: int = CONFIG["sample_rate"]) -> str:
        """Write numpy audio array to temporary WAV file and transcribe."""
        from tempfile import NamedTemporaryFile

        if wavfile is None:
            die("scipy is required to write temporary WAV files.")

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name

        try:
            wavfile.write(tmp_name, sample_rate, audio)
            return self.transcribe_file(tmp_name)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass
