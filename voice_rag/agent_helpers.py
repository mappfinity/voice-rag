from typing import Any, Tuple, Optional, List
from .utils import die, info
from .config import CONFIG

# Optional imports with graceful fallback for environments without audio libs
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
    Record microphone audio into a NumPy int16 array.

    - Uses sounddevice for capture and falls back to config defaults.
    - Ensures deterministic sample rate selection with a simple fallback.
    - Returns a squeezed ndarray for consistent downstream processing.
    """
    if sd is None or np is None:
        die("sounddevice and numpy are required for recording audio.")

    # Resolve recording duration
    if duration_seconds is None:
        duration_seconds = CONFIG.get("record_seconds_default", 5)

    # Device + channel configuration from settings
    target_device = CONFIG.get("mic_device_id", None)
    target_channels = max(1, int(CONFIG.get("mic_channels", 1)))

    # Determine sample rate; fall back to a safe default if querying fails
    try:
        dev_info = sd.query_devices(
            target_device, "input"
        ) if target_device is not None else sd.query_devices(kind="input")

        fs = int(CONFIG.get("sample_rate", 16000))
    except Exception as e:
        info(f"Could not query microphone device; using fallback rate. ({e})")
        fs = 44100

    info(
        f"Recording {duration_seconds}s at {fs}Hz "
        f"(device={target_device}, channels={target_channels})"
    )

    # Capture audio frames with input validation
    try:
        frames = int(duration_seconds * fs)
        if frames <= 0:
            die("Invalid duration; resulting frame count is <= 0.")

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

    # Standardize return type to a clean int16 ndarray
    try:
        data = np.asarray(audio, dtype=np.int16)
        data = np.squeeze(data)
    except Exception as e:
        die(f"Unexpected error converting audio to numpy array: {e}")

    return data, fs


def save_numpy_to_wav(audio: Any, path: str, fs: int):
    """
    Write a NumPy int16 audio array to a WAV file.

    - Requires scipy's wavfile backend.
    - Performs basic validation on input data and sample rate.
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
