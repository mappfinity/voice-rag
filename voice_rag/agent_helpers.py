from typing import Any, Tuple, Optional, List
from .utils import die, info
from .config import CONFIG

# Optional imports with graceful fallback
try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None


# ============================================================================
# Microphone Recording Helpers
# ============================================================================

def record_to_numpy(
        duration_seconds: Optional[int] = None
) -> Tuple[Any, int]:
    """
    Record audio from the microphone into a NumPy int16 array.

    Parameters
    ----------
    duration_seconds : int, optional
        Length of recording in seconds. If None, uses CONFIG["record_seconds_default"].

    Returns
    -------
    Tuple[data, sample_rate]
        data : np.ndarray (int16)
            The recorded audio data (mono or multi-channel).
        sample_rate : int
            The sampling rate used for recording.

    Raises
    ------
    SystemExit
        If sounddevice or numpy is unavailable, or recording fails.
    """
    if sd is None or np is None:
        die("sounddevice and numpy are required for recording audio.")

    # Determine duration
    if duration_seconds is None:
        duration_seconds = CONFIG.get("record_seconds_default", 5)

    # Device settings
    target_device = CONFIG.get("mic_device_id", None)
    target_channels = max(1, int(CONFIG.get("mic_channels", 1)))

    # Determine sample rate with a robust fallback
    try:
        if target_device is not None:
            dev_info = sd.query_devices(target_device, "input")
        else:
            dev_info = sd.query_devices(kind="input")

        fs = int(CONFIG.get("sample_rate", 16000))
    except Exception as e:
        info(
            f"Could not query microphone device; using fallback sample rate. ({e})")
        fs = 44100

    info(f"Recording {duration_seconds}s at {fs}Hz "
         f"(device={target_device}, channels={target_channels})")

    # Record audio safely
    try:
        frames = int(duration_seconds * fs)
        if frames <= 0:
            die("Invalid duration; number of audio frames would be <= 0.")

        audio = sd.rec(
            frames,
            samplerate=fs,
            channels=target_channels,
            dtype="int16",
            device=target_device
        )
        sd.wait()

    except Exception as e:
        die(f"Recording failed: {e}")

    # Ensure output is a clean ndarray
    try:
        data = np.asarray(audio, dtype=np.int16)
        data = np.squeeze(data)
    except Exception as e:
        die(f"Unexpected error converting audio to numpy array: {e}")

    return data, fs


def save_numpy_to_wav(audio: Any, path: str, fs: int):
    """
    Save a NumPy array containing int16 audio data to a WAV file.

    Parameters
    ----------
    audio : Any
        Should be a NumPy ndarray containing int16 audio samples.
    path : str
        Output file path.
    fs : int
        Sample rate.

    Raises
    ------
    SystemExit
        If scipy is not installed or writing fails.
    """
    if wavfile is None:
        die("scipy is required to write WAV files.")

    if audio is None:
        die("save_numpy_to_wav() received empty audio data.")

    if not isinstance(fs, int) or fs <= 0:
        die(f"Invalid sample rate: {fs}")

    try:
        wavfile.write(path, fs, audio)
        info(f"WAV saved: {path} ({fs}Hz)")
    except Exception as e:
        die(f"Could not write WAV file '{path}': {e}")
