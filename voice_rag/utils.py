import sys
import logging
import subprocess
from typing import List, Tuple

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger("voice_rag.utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -----------------------------
# Utilities
# -----------------------------
def die(msg: str):
    """Log critical error and exit."""
    logger.critical(msg)
    raise SystemExit(msg)

def info(msg: str):
    """Log informational message."""
    logger.info(msg)

def warn(msg: str):
    """Log warning message."""
    logger.warning(msg)

def run_cmd(
        cmd: List[str],
        capture_output: bool = True,
        text: bool = True,
        timeout: int = 180
) -> Tuple[int, str, str]:
    """
    Run a shell command safely.

    Returns:
        returncode, stdout, stderr
    """
    try:
        cp = subprocess.run(cmd, capture_output=capture_output, text=text, timeout=timeout)
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)
