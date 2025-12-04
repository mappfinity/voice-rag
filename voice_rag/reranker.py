from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import CONFIG


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two sets of vectors.

    Args:
        a: Array of shape (N, D) or (D,)
        b: Array of shape (M, D) or (D,)

    Returns:
        Similarity matrix of shape (N, M).
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)  # Ensure 2D for batch computation
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12

    return np.dot(a / a_norm, (b / b_norm).T)

# ---------------------------------------------------------
# Optimized Cross-Encoder (CPU/GPU auto)
# ---------------------------------------------------------
class MiniLMCrossEncoder(nn.Module):
    """
    Optimized MiniLM cross-encoder.

    Features:
        - Uses pre-trained cross-encoder scoring head
        - TorchScript tracing for CPU speed
        - CUDA acceleration if available
        - Reduced max_length for faster inference
        - Batched inference
    """

    def __init__(self, model_name: str = CONFIG["cross_encoder"]):
        super().__init__()

        # -----------------------
        # Device selection (CPU or GPU)
        # -----------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 192  # Shorter max length for faster throughput

        # -----------------------
        # Tokenizer
        # -----------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # -----------------------
        # Pre-trained cross-encoder model
        # -----------------------
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(self.device)

        # -----------------------
        # TorchScript tracing for CPU acceleration (optional)
        # -----------------------
        if self.device.type == "cpu":
            try:
                example_input = {
                    "input_ids": torch.ones(1, 8, dtype=torch.long),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)
                }
                self.model = torch.jit.trace(self.model, example_input)
                self.use_traced = True
            except Exception:
                self.use_traced = False
        else:
            # GPU: tracing not needed
            self.use_traced = False

        # Limit CPU threads for consistent performance
        if self.device.type == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 6))

    @torch.inference_mode()
    def forward(self, queries, docs, batch_size: int = 16):
        """
        Compute relevance scores for query-document pairs (CPU/GPU optimized).

        Args:
            queries: List of query strings
            docs: List of document strings
            batch_size: Number of pairs to process at once

        Returns:
            Tensor of relevance scores.
        """
        assert len(queries) == len(docs)
        all_scores = []

        for i in range(0, len(docs), batch_size):
            qs = queries[i:i + batch_size]
            ds = docs[i:i + batch_size]

            # -----------------------
            # Tokenize batch
            # -----------------------
            encoded = self.tokenizer(
                qs,
                ds,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)

            # -----------------------
            # Forward pass
            # -----------------------
            if self.device.type == "cuda":
                # GPU: optionally use autocast for FP16 speedup
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = self.model(**encoded)
            else:
                # CPU
                out = self.model(**encoded)

            scores = out.logits.squeeze(-1)  # shape [batch]
            all_scores.append(scores)

        # -----------------------
        # Concatenate all batch scores
        # -----------------------
        return torch.cat(all_scores, dim=0)

# ---------------------------------------------------------
# ReRanker (embedding + optional CE)
# ---------------------------------------------------------
class ReRanker:
    """
    Lightweight embedding-based reranker with optional Maximal Marginal Relevance (MMR).

    Args:
        embed_fn: Callable producing embeddings from texts
        ollama: Optional CE backend placeholder
        normalize_embeddings: Whether to normalize embeddings before scoring
        cross_encoder_batch_size: Batch size for cross-encoder inference
        mmr_lambda: Relevance-diversity trade-off for MMR
        use_cross_encoder: Flag for enabling cross-encoder rescoring
        cross_encoder: Optional pre-initialized cross-encoder
    """
    def __init__(
            self,
            embed_fn: Callable[[Sequence[str]], Union[np.ndarray, List[List[float]]]],
            ollama: Optional[Any] = None,
            normalize_embeddings: bool = True,
            cross_encoder_batch_size: int = CONFIG.get("reranker_cross_batch_size", 4),
            mmr_lambda: float = CONFIG.get("mmr_lambda", 0.6),
            use_cross_encoder: bool = CONFIG.get("reranker_use_cross_encoder", False),
            cross_encoder: Optional[nn.Module] = None,
    ):
        self.embed_fn = embed_fn
        self.ollama = ollama
        self.normalize_embeddings = normalize_embeddings
        self.cross_encoder_batch_size = cross_encoder_batch_size
        self.mmr_lambda = mmr_lambda
        self.use_cross_encoder = use_cross_encoder

        self.cross_encoder = cross_encoder or MiniLMCrossEncoder()
        self.cross_encoder.eval()

    # -----------------------------
    # Embedding helpers
    # -----------------------------
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Compute normalized embeddings for a list of texts."""
        if not texts:
            return np.array([])

        arr = self.embed_fn(list(texts))
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self.normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            arr = arr / norms

        return arr

    def embed_query(self, q: str) -> np.ndarray:
        """Compute normalized embedding for a single query string."""
        return self.embed_texts([q])[0]

    # -----------------------------
    # Cross-encoder scoring
    # -----------------------------
    def cross_encoder_rescore(self, query: str, docs: Sequence[str]) -> np.ndarray:
        """Compute cross-encoder scores for a query and documents in batches."""
        scores = []
        bs = self.cross_encoder_batch_size

        with torch.no_grad():
            for i in range(0, len(docs), bs):
                chunk = docs[i:i+bs]
                s = self.cross_encoder([query] * len(chunk), chunk)
                scores.append(s.cpu().numpy())

        return np.concatenate(scores, axis=0) if scores else np.array([])

    # -----------------------------
    # Normalization
    # -----------------------------
    def _safe_normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores safely, returning zeros if uniform."""
        if scores.size == 0:
            return scores
        lo, hi = scores.min(), scores.max()
        if hi == lo:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    # -----------------------------
    # Maximal Marginal Relevance (MMR)
    # -----------------------------
    def mmr(
            self,
            q_emb: np.ndarray,
            d_emb: np.ndarray,
            top_k: int,
            lambda_param: Optional[float] = None,
    ) -> List[int]:
        """
        Select top_k documents using MMR balancing relevance and diversity.

        Args:
            q_emb: Query embedding
            d_emb: Document embeddings
            top_k: Number of results to select
            lambda_param: Relevance-diversity trade-off
        """
        lambda_param = lambda_param if lambda_param is not None else self.mmr_lambda
        n = d_emb.shape[0]

        if top_k >= n:
            return list(range(n))

        selected = []
        remaining = set(range(n))

        q_emb_r = q_emb.reshape(1, -1)
        sim_q = _cosine_sim(q_emb_r, d_emb)[0]
        sim_docs = _cosine_sim(d_emb, d_emb)

        first = int(np.argmax(sim_q))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -1e10

            for idx in remaining:
                redundancy = max(sim_docs[idx, s] for s in selected)
                score = lambda_param * sim_q[idx] - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # -----------------------------
    # Main reranker
    # -----------------------------
    def rerank(
            self,
            query: str,
            candidates: Sequence[Dict[str, Any]],
            top_k: int = 5,
            use_mmr: bool = True,
            return_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents given a query using embeddings, optional MMR, or cross-encoder.

        Args:
            query: Query string
            candidates: List of candidate documents with 'text' and optional 'embedding'
            top_k: Number of results to return
            use_mmr: Whether to apply MMR for diversity
            return_scores: Whether to include normalized scores in output

        Returns:
            List of reranked candidates with optional '_score'.
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]

        # 1. Cross-encoder rescoring
        if self.use_cross_encoder:
            raw = self.cross_encoder_rescore(query, texts)
            norm_scores = self._safe_normalize_scores(raw)
            chosen_idx = np.argsort(-norm_scores)[:top_k].tolist()

        # 2. Vector-based reranking
        else:
            embeddings = []
            missing = []

            for i, c in enumerate(candidates):
                emb = c.get("embedding")
                if emb is None:
                    missing.append(i)
                    embeddings.append(None)
                else:
                    embeddings.append(np.asarray(emb))

            # Compute embeddings for missing candidates
            if missing:
                e = self.embed_texts([texts[i] for i in missing])
                for j, idx in enumerate(missing):
                    embeddings[idx] = e[j]

            d_emb = np.vstack(embeddings)
            q_emb = self.embed_query(query)

            raw = _cosine_sim(q_emb.reshape(1, -1), d_emb)[0]
            norm_scores = self._safe_normalize_scores(raw)

            if use_mmr and top_k < len(candidates):
                chosen_idx = self.mmr(q_emb, d_emb, top_k)
            else:
                chosen_idx = np.argsort(-norm_scores)[:top_k].tolist()

        # Build final results
        results = []
        for idx in chosen_idx:
            item = dict(candidates[idx])
            item["_score"] = float(norm_scores[idx])
            results.append(item)

        results.sort(key=lambda x: x["_score"], reverse=True)

        if not return_scores:
            for r in results:
                r.pop("_score", None)

        return results
