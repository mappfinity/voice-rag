from dataclasses import dataclass
from .utils import run_cmd, info, warn, die
from .config import CONFIG
import shutil
import json

try:
    import requests
except ImportError:
    requests = None


# -----------------------------
# Ollama CLI / HTTP Utilities
# -----------------------------
def has_ollama_cli() -> bool:
    """Return True if the `ollama` CLI exists in PATH."""
    return shutil.which("ollama") is not None


def check_ollama_http() -> bool:
    """Return True if the configured Ollama HTTP endpoint responds."""
    if requests is None:
        return False
    try:
        r = requests.get(CONFIG["ollama_http"], timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------
# Ollama local client wrapper
# -----------------------------
@dataclass
class OllamaLocal:
    """
    Lightweight wrapper for interacting with Ollama via CLI or HTTP.

    Attributes:
        model: Model name to run (from config).
        http_base: Base URL for the Ollama HTTP API.
        prefer_http: Whether to try HTTP before falling back to CLI.
    """
    model: str = CONFIG["ollama_model"]
    http_base: str = CONFIG["ollama_http"]
    prefer_http: bool = True

    # -----------------------------
    # CLI Generation
    # -----------------------------
    def generate_cli(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        """
        Generate text using the Ollama CLI.

        Raises:
            RuntimeError: If the CLI is unavailable or execution fails.
        """
        if not has_ollama_cli():
            raise RuntimeError("Ollama CLI is not available.")

        cmd = ["ollama", "run", self.model, prompt]
        info("Running Ollama via CLI...")
        rc, out, err = run_cmd(cmd, timeout=320)

        if rc != 0:
            raise RuntimeError(f"Ollama CLI failed: {err.strip()[:300]}")
        return out.strip()

    # -----------------------------
    # HTTP Generation
    # -----------------------------
    def generate_http(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        """
        Generate text using the Ollama HTTP API.

        Returns:
            The generated string response, or empty string on failure.
        """
        if requests is None:
            die("`requests` is required for HTTP mode. Install via `pip install requests`.")

        url = f"{self.http_base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
        }

        info(f"Calling Ollama HTTP API: {url}")
        try:
            r = requests.post(url, json=payload, timeout=320)
            r.raise_for_status()
            data = r.json()

            # Support multiple response field names across versions
            if isinstance(data, dict):
                for key in ("response", "text"):
                    if key in data:
                        return data[key]

            return json.dumps(data)

        except Exception as e:
            warn(f"Ollama HTTP request failed: {e}")
            return ""

    # -----------------------------
    # Unified Generation
    # -----------------------------
    def generate(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        """
        Generate text using Ollama.

        Tries HTTP first (if enabled and reachable), then falls back to CLI.
        Dies with an error message if neither method is available.
        """
        # Prefer HTTP if configured and responding
        if self.prefer_http and check_ollama_http():
            out = self.generate_http(prompt, max_tokens)
            if out:
                return out

        # Fallback to CLI if installed
        if has_ollama_cli():
            return self.generate_cli(prompt, max_tokens)

        die("No available Ollama backend (HTTP or CLI).")
