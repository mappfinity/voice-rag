from dataclasses import dataclass
from typing import Iterator, Optional
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

        # Use timeout from config
        timeout = CONFIG.get("llm_timeout", 120)
        rc, out, err = run_cmd(cmd, timeout=timeout)

        if rc != 0:
            raise RuntimeError(f"Ollama CLI failed: {err.strip()[:300]}")
        return out.strip()

    # -----------------------------
    # HTTP Generation (Non-Streaming)
    # -----------------------------
    def generate_http(self, prompt: str, max_tokens: int = CONFIG["llm_max_tokens"]) -> str:
        """
        Generate text using the Ollama HTTP API (non-streaming).

        Returns:
            The generated string response, or empty string on failure.
        """
        if requests is None:
            die("`requests` is required for HTTP mode. Install via `pip install requests`.")

        url = f"{self.http_base}/api/generate"

        # Build options with all parameters
        options = {
            "temperature": CONFIG.get("temperature", 0.1),
            "top_p": CONFIG.get("top_p", 0.95),
            "top_k": CONFIG.get("top_k", 50),
        }

        # Add repeat_penalty if configured
        if "repeat_penalty" in CONFIG:
            options["repeat_penalty"] = CONFIG["repeat_penalty"]

        # Add num_predict if configured and not default
        num_predict = CONFIG.get("num_predict", -1)
        if num_predict != -1:
            options["num_predict"] = num_predict

        # Add context window limit if configured
        if "llm_context_window" in CONFIG:
            options["num_ctx"] = CONFIG["llm_context_window"]

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        # Add stop sequences if configured
        stop_sequences = CONFIG.get("stop_sequences")
        if stop_sequences:
            payload["stop"] = stop_sequences

        # Use timeout from config
        timeout = CONFIG.get("llm_timeout", 120)

        if CONFIG.get("verbose_llm", False):
            info(f"Calling Ollama HTTP API: {url}")
            info(f"Payload: {json.dumps(payload, indent=2)}")
        else:
            info(f"Calling Ollama HTTP API: {url}")

        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            # Support multiple response field names across versions
            if isinstance(data, dict):
                # Log token usage if verbose
                if CONFIG.get("verbose_llm", False):
                    if "eval_count" in data:
                        info(f"Tokens generated: {data['eval_count']}")
                    if "prompt_eval_count" in data:
                        info(f"Prompt tokens: {data['prompt_eval_count']}")
                    if "total_duration" in data:
                        duration_sec = data["total_duration"] / 1e9
                        info(f"Generation time: {duration_sec:.2f}s")

                for key in ("response", "text"):
                    if key in data:
                        return data[key]

            return json.dumps(data)

        except requests.exceptions.Timeout:
            warn(f"Ollama HTTP request timed out after {timeout}s")
            return ""
        except requests.exceptions.RequestException as e:
            warn(f"Ollama HTTP request failed: {e}")
            return ""
        except Exception as e:
            warn(f"Unexpected error in Ollama HTTP: {e}")
            return ""

    # -----------------------------
    # HTTP Streaming Generation
    # -----------------------------
    def generate_http_stream(
            self,
            prompt: str,
            max_tokens: int = CONFIG["llm_max_tokens"]
    ) -> Iterator[str]:
        """
        Generate text using the Ollama HTTP API with streaming.

        Yields:
            String chunks as they are generated by the model.

        Raises:
            RuntimeError: If requests library is unavailable or request fails.
        """
        if requests is None:
            raise RuntimeError(
                "`requests` is required for HTTP mode. Install via `pip install requests`."
            )

        url = f"{self.http_base}/api/generate"

        # Build options with all parameters
        options = {
            "temperature": CONFIG.get("temperature", 0.1),
            "top_p": CONFIG.get("top_p", 0.95),
            "top_k": CONFIG.get("top_k", 50),
        }

        # Add repeat_penalty if configured
        if "repeat_penalty" in CONFIG:
            options["repeat_penalty"] = CONFIG["repeat_penalty"]

        # Add num_predict if configured and not default
        num_predict = CONFIG.get("num_predict", -1)
        if num_predict != -1:
            options["num_predict"] = num_predict

        # Add context window limit if configured
        if "llm_context_window" in CONFIG:
            options["num_ctx"] = CONFIG["llm_context_window"]

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
            "options": options,
        }

        # Add stop sequences if configured
        stop_sequences = CONFIG.get("stop_sequences")
        if stop_sequences:
            payload["stop"] = stop_sequences

        # Use timeout from config
        timeout = CONFIG.get("llm_timeout", 120)

        if CONFIG.get("verbose_llm", False):
            info(f"Calling Ollama HTTP API (streaming): {url}")
            info(f"Payload: {json.dumps(payload, indent=2)}")
        else:
            info(f"Calling Ollama HTTP API (streaming): {url}")

        try:
            with requests.post(
                    url,
                    json=payload,
                    timeout=timeout,
                    stream=True
            ) as response:
                response.raise_for_status()

                # Process streaming response line by line
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)

                        # Extract response text from chunk
                        if isinstance(chunk, dict):
                            # Look for response field
                            if "response" in chunk:
                                text = chunk["response"]
                                if text:
                                    yield text
                            elif "text" in chunk:
                                text = chunk["text"]
                                if text:
                                    yield text

                            # Check if generation is done
                            if chunk.get("done", False):
                                # Log final stats if verbose
                                if CONFIG.get("verbose_llm", False):
                                    if "eval_count" in chunk:
                                        info(f"Tokens generated: {chunk['eval_count']}")
                                    if "prompt_eval_count" in chunk:
                                        info(f"Prompt tokens: {chunk['prompt_eval_count']}")
                                    if "total_duration" in chunk:
                                        duration_sec = chunk["total_duration"] / 1e9
                                        info(f"Generation time: {duration_sec:.2f}s")
                                break

                    except json.JSONDecodeError as e:
                        warn(f"Failed to decode streaming chunk: {e}")
                        continue

        except requests.exceptions.Timeout:
            warn(f"Ollama HTTP streaming request timed out after {timeout}s")
            raise RuntimeError(f"Streaming request timed out after {timeout}s")
        except requests.exceptions.RequestException as e:
            warn(f"Ollama HTTP streaming request failed: {e}")
            raise RuntimeError(f"Streaming request failed: {e}")
        except Exception as e:
            warn(f"Unexpected error in Ollama HTTP streaming: {e}")
            raise RuntimeError(f"Unexpected streaming error: {e}")

    # -----------------------------
    # Streaming with Full Response Collection
    # -----------------------------
    def generate_stream_full(
            self,
            prompt: str,
            max_tokens: int = CONFIG["llm_max_tokens"],
            callback: Optional[callable] = None
    ) -> str:
        """
        Generate text using streaming and return the full response.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            callback: Optional callback function called for each chunk: callback(chunk_text)

        Returns:
            The complete generated text as a single string.

        Example:
            >>> def print_chunk(chunk):
            ...     print(chunk, end='', flush=True)
            >>> response = ollama.generate_stream_full(prompt, callback=print_chunk)
        """
        chunks = []

        try:
            for chunk in self.generate_http_stream(prompt, max_tokens):
                chunks.append(chunk)
                if callback:
                    try:
                        callback(chunk)
                    except Exception as e:
                        warn(f"Callback error: {e}")
        except Exception as e:
            warn(f"Streaming failed, falling back to non-streaming: {e}")
            # Fallback to non-streaming generation
            return self.generate_http(prompt, max_tokens)

        return "".join(chunks)

    # -----------------------------
    # Unified Generation
    # -----------------------------
    def generate(
            self,
            prompt: str,
            max_tokens: int = CONFIG["llm_max_tokens"],
            stream: bool = False,
            callback: Optional[callable] = None
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            stream: If True, use streaming (only works with HTTP backend)
            callback: Optional callback for streaming chunks (only used if stream=True)

        Returns:
            The complete generated text.

        Tries HTTP first (if enabled and reachable), then falls back to CLI.
        Dies with an error message if neither method is available.
        """
        # Handle streaming request
        if stream:
            if self.prefer_http and check_ollama_http():
                return self.generate_stream_full(prompt, max_tokens, callback)
            else:
                warn("Streaming requested but HTTP backend not available, using non-streaming")

        # Prefer HTTP if configured and responding
        if self.prefer_http and check_ollama_http():
            out = self.generate_http(prompt, max_tokens)
            if out:
                return out

        # Fallback to CLI if installed
        if has_ollama_cli():
            return self.generate_cli(prompt, max_tokens)

        die("No available Ollama backend (HTTP or CLI).")


# -----------------------------
# Convenience Functions
# -----------------------------
def stream_generate(
        prompt: str,
        model: Optional[str] = None,
        callback: Optional[callable] = None
) -> str:
    """
    Convenience function for streaming generation.

    Args:
        prompt: The input prompt
        model: Optional model name (defaults to CONFIG["ollama_model"])
        callback: Optional callback function called for each chunk

    Returns:
        The complete generated text.

    Example:
        >>> response = stream_generate(
        ...     "Explain quantum computing",
        ...     callback=lambda chunk: print(chunk, end='', flush=True)
        ... )
    """
    ollama = OllamaLocal(model=model or CONFIG["ollama_model"])
    return ollama.generate(prompt, stream=True, callback=callback)