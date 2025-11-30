import sys
import logging
import subprocess
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("voice_rag.utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------
def die(msg: str):
    """
    Log a critical error and exit the program.

    Parameters
    ----------
    msg : str
        Error message to log.
    """
    logger.critical(msg)
    raise SystemExit(msg)


def info(msg: str):
    """
    Log an informational message.

    Parameters
    ----------
    msg : str
        Message to log.
    """
    logger.info(msg)


def warn(msg: str):
    """
    Log a warning message.

    Parameters
    ----------
    msg : str
        Message to log.
    """
    logger.warning(msg)


# ---------------------------------------------------------------------------
# Command execution utility
# ---------------------------------------------------------------------------
def run_cmd(
        cmd: List[str],
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 180
) -> Tuple[int, str, str]:
    """
    Execute a shell command safely and return its results.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments to run.
    capture_output : bool
        If True, capture stdout and stderr.
    text : bool
        If True, return output as text instead of bytes.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    Tuple[int, str, str]
        returncode, stdout, stderr
    """
    try:
        cp = subprocess.run(cmd, capture_output=capture_output, text=text, timeout=timeout)
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)
