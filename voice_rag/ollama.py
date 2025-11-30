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
    """Check if Ollama CLI is available in PATH."""
    return shutil.which("ollama") is not None


def check_ollama_http() -> bool:
    """Check if Ollama HTTP endpoint is reachable."""
    if requests is None:
        return False
    try:
        r = requests.get(CONFIG["ollama_http"], timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------
# Ollama client wrapper
# -----------------------------
@dataclass
class OllamaLocal:
    model: str = CONFIG["ollama_model"]
    http_base: str = CONFIG["ollama_http"]
    prefer_http: bool = True

    # -----------------------------
    # CLI-based generation
    # -----------------------------
    def generate_cli(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        if not has_ollama_cli():
            raise RuntimeError("Ollama CLI not available.")
        cmd = ["ollama", "run", self.model, prompt]
        info("Running Ollama CLI...")
        rc, out, err = run_cmd(cmd, timeout=180)
        if rc != 0:
            raise RuntimeError(f"Ollama CLI failed: {err.strip()[:300]}")
        return out.strip()

    # -----------------------------
    # HTTP-based generation
    # -----------------------------
    def generate_http(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        if requests is None:
            die("requests library required for HTTP usage. Install with `pip install requests`.")
        url = f"{self.http_base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40}  # model-specific options
        }
        info(f"Calling Ollama HTTP API at {url}...")
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            # Try multiple response keys for compatibility
            if isinstance(data, dict):
                for key in ["response", "text"]:
                    if key in data:
                        return data[key]
            return json.dumps(data)
        except Exception as e:
            warn(f"Ollama HTTP call failed: {e}")
            return ""

    # -----------------------------
    # Unified generation
    # -----------------------------
    def generate(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        """
        Generate text using Ollama. Prefers HTTP if available, falls back to CLI.
        """
        if self.prefer_http and check_ollama_http():
            out = self.generate_http(prompt, max_tokens)
            if out:
                return out
        if has_ollama_cli():
            return self.generate_cli(prompt, max_tokens)
        die("No Ollama endpoint available (HTTP or CLI).")
