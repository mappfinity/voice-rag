from dataclasses import dataclass
from typing import Any, Optional
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
        Whisper checkpoint size (e.g., "tiny", "base", "small", "medium", "large").
        Defaults to config value.
    model : Any
        Loaded WhisperModel instance.
    device : str
        Target device ("cpu", "cuda").
    compute_type : str
        Compute type for inference (e.g., "int8", "float16", "float32").
    """
    model_size: str = None
    model: Any = None
    device: str = "cpu"
    compute_type: str = None

    def __post_init__(self):
        """Initialize model parameters from config if not provided."""
        if self.model_size is None:
            self.model_size = CONFIG.get("whisper_model", "medium")
        if self.compute_type is None:
            self.compute_type = CONFIG.get("whisper_compute_type", "int8")

        info(f"LocalSTT initialized: model={self.model_size}, "
             f"device={self.device}, compute_type={self.compute_type}")

    # ----------------------------------------------------------------------
    # Model Initialization
    # ----------------------------------------------------------------------
    def init_model(self) -> None:
        """Load the Whisper model on first use with enhanced configuration."""
        if WhisperModel is None:
            die("faster-whisper is required. Install with `pip install faster-whisper`.")

        if self.model is None:
            info(f"Loading Whisper model: {self.model_size} (device={self.device}, compute={self.compute_type})")

            try:
                # Initialize with enhanced configuration
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None,  # Use default cache
                    local_files_only=False,
                )
                info(f"Whisper model loaded successfully")
            except Exception as e:
                warn(f"Failed to load with compute_type={self.compute_type}: {e}")
                info("Falling back to default compute type")
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device
                    )
                    info("Whisper model loaded with default compute type")
                except Exception as e2:
                    die(f"Failed to load Whisper model: {e2}")

    # ----------------------------------------------------------------------
    # File Transcription
    # ----------------------------------------------------------------------
    def transcribe_file(
            self,
            path: str,
            language: Optional[str] = None,
            beam_size: Optional[int] = None,
            temperature: Optional[tuple] = None,
            vad_filter: Optional[bool] = None,
            initial_prompt: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio file with enhanced Whisper parameters.

        Parameters
        ----------
        path : str
            Path to an audio file readable by Faster-Whisper.
        language : str, optional
            Language code (e.g., "en"). Defaults to config value.
        beam_size : int, optional
            Beam size for decoding. Higher = more accurate but slower.
            Defaults to config value (3).
        temperature : tuple, optional
            Temperature fallback sequence for sampling.
            Defaults to config value (0.0, 0.2, 0.4, 0.6, 0.8).
        vad_filter : bool, optional
            Enable Voice Activity Detection to reduce hallucinations.
            Defaults to config value (True).
        initial_prompt : str, optional
            Prompt to bias transcription toward specific vocabulary.
            Defaults to config technical prompt.

        Returns
        -------
        str
            Transcribed text, or empty string on failure.
        """
        if self.model is None:
            self.init_model()

        # Use config defaults if not specified
        if language is None:
            language = CONFIG.get("whisper_language", "en")
        if beam_size is None:
            beam_size = CONFIG.get("whisper_beam_size", 3)
        if temperature is None:
            temperature = CONFIG.get("whisper_temperature", (0.0, 0.2, 0.4, 0.6, 0.8))
        if vad_filter is None:
            vad_filter = CONFIG.get("whisper_vad_filter", True)
        if initial_prompt is None:
            initial_prompt = CONFIG.get(
                "whisper_initial_prompt",
                "Technical discussion about machine learning, data science, AI engineering, and programming."
            )

        info(f"Transcribing file: {path}")

        if CONFIG.get("verbose_retrieval", False):
            info(f"Whisper parameters: language={language}, beam_size={beam_size}, "
                 f"vad_filter={vad_filter}, temperature={temperature}")

        try:
            # Prepare transcription options
            transcribe_options = {
                "beam_size": beam_size,
                "language": language,
                "temperature": temperature,
            }

            # Add VAD filter if enabled
            if vad_filter:
                transcribe_options["vad_filter"] = True
                # Optional: configure VAD parameters
                vad_parameters = CONFIG.get("whisper_vad_parameters")
                if vad_parameters:
                    transcribe_options["vad_parameters"] = vad_parameters

            # Add initial prompt if provided
            if initial_prompt:
                transcribe_options["initial_prompt"] = initial_prompt

            # Transcribe with all options
            segments, info_dict = self.model.transcribe(
                path,
                **transcribe_options
            )

            # Collect segments
            text_segments = []
            for seg in segments:
                text_segments.append(seg.text)

            result = " ".join(text_segments).strip()

            # Log transcription info
            if CONFIG.get("verbose_retrieval", False) and hasattr(info_dict, 'language'):
                info(f"Detected language: {info_dict.language}, "
                     f"probability: {info_dict.language_probability:.2f}")

            if result:
                info(f"Transcription complete: {len(result)} characters")
            else:
                warn(f"Transcription produced empty result for {path}")

            return result

        except Exception as e:
            warn(f"Transcription failed for {path}: {e}")
            return ""

    # ----------------------------------------------------------------------
    # Array Transcription
    # ----------------------------------------------------------------------
    def transcribe_audio_array(
            self,
            audio: Any,
            sample_rate: int = None,
            **kwargs
    ) -> str:
        """
        Transcribe an in-memory audio array by writing a temporary WAV.

        Parameters
        ----------
        audio : Any
            Numpy array representing audio samples.
        sample_rate : int, optional
            Sample rate of the array. Defaults to config value.
        **kwargs
            Additional arguments passed to transcribe_file (language, beam_size, etc.)

        Returns
        -------
        str
            Transcribed text.
        """
        from tempfile import NamedTemporaryFile

        if wavfile is None:
            die("scipy is required for writing temporary WAV files.")

        if sample_rate is None:
            sample_rate = CONFIG.get("sample_rate", 16000)

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name

        try:
            wavfile.write(tmp_name, sample_rate, audio)
            return self.transcribe_file(tmp_name, **kwargs)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns
        -------
        dict
            Model information including size, device, and compute type.
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": self.model is not None,
        }

    def reload_model(self, model_size: Optional[str] = None) -> None:
        """Reload the model with a different size.

        Parameters
        ----------
        model_size : str, optional
            New model size. If None, reloads with current size.
        """
        if model_size:
            self.model_size = model_size

        # Clear existing model
        self.model = None

        # Reinitialize
        self.init_model()
        info(f"Model reloaded: {self.model_size}")