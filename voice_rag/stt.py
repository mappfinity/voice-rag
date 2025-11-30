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


# ---------------------------------------------------------------------------
# Local Speech-to-Text (Faster-Whisper)
# ---------------------------------------------------------------------------
@dataclass
class LocalSTT:
    """
    Local speech-to-text wrapper using Faster-Whisper.

    Parameters
    ----------
    model_size : str
        Whisper checkpoint size (e.g., "tiny", "base", "medium").
    model : Any
        Loaded WhisperModel instance.
    device : str
        Target device ("cpu", "cuda").
    """
    model_size: str = CONFIG["whisper_model"]
    model: Any = None
    device: str = "cpu"

    # ----------------------------------------------------------------------
    # Model Initialization
    # ----------------------------------------------------------------------
    def init_model(self) -> None:
        """Load the Whisper model on first use."""
        if WhisperModel is None:
            die("faster-whisper is required. Install with `pip install faster-whisper`.")

        if self.model is None:
            info(f"Loading Whisper model: {self.model_size} (device={self.device})")
            compute = CONFIG.get("whisper_compute_type")

            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute,
                )
            except Exception:
                info("Requested compute_type unsupported; falling back to default.")
                self.model = WhisperModel(self.model_size, device=self.device)

    # ----------------------------------------------------------------------
    # File Transcription
    # ----------------------------------------------------------------------
    def transcribe_file(self, path: str) -> str:
        """
        Transcribe an audio file.

        Parameters
        ----------
        path : str
            Path to an audio file readable by Faster-Whisper.

        Returns
        -------
        str
            Transcribed text, or empty string on failure.
        """
        if self.model is None:
            self.init_model()

        info(f"Transcribing file: {path}")
        try:
            segments, _ = self.model.transcribe(
                path,
                beam_size=CONFIG.get("whisper_beam_size", 1),
                language=CONFIG.get("whisper_language"),
                temperature=CONFIG.get("whisper_temperature"),
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            warn(f"Transcription failed for {path}: {e}")
            return ""

    # ----------------------------------------------------------------------
    # Array Transcription
    # ----------------------------------------------------------------------
    def transcribe_audio_array(
            self,
            audio: Any,
            sample_rate: int = CONFIG["sample_rate"],
    ) -> str:
        """
        Transcribe an in-memory audio array by writing a temporary WAV.

        Parameters
        ----------
        audio : Any
            Numpy array representing audio samples.
        sample_rate : int
            Sample rate of the array.

        Returns
        -------
        str
            Transcribed text.
        """
        from tempfile import NamedTemporaryFile

        if wavfile is None:
            die("scipy is required for writing temporary WAV files.")

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
