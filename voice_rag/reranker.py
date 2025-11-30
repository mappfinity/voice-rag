from dataclasses import dataclass
from typing import Sequence, Callable, Any, Optional, Dict, List
import numpy as np
import math
from .config import CONFIG


# -----------------------------
# Cosine Similarity Utility
# -----------------------------
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
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12

    return np.dot(a / a_norm, (b / b_norm).T)


# -----------------------------
# Reranker
# -----------------------------
@dataclass
class ReRanker:
    """
    Lightweight embedding reranker with optional Maximal Marginal Relevance (MMR).

    Args:
        embed_fn: Function(texts) -> np.ndarray, producing document embeddings.
        ollama: Optional cross-encoder backend placeholder (not required).
        normalize_embeddings: Normalize embeddings before scoring.
        cross_encoder_batch_size: Reserved for cross-encoder use.
        mmr_lambda: Trade-off factor between relevance and diversity.
        use_cross_encoder: Placeholder flag for future CE integration.
    """
    embed_fn: Callable[[Sequence[str]], np.ndarray]
    ollama: Optional[Any] = None
    normalize_embeddings: bool = True
    cross_encoder_batch_size: int = CONFIG.get("reranker_cross_batch_size", 4)
    mmr_lambda: float = CONFIG.get("mmr_lambda", 0.6)
    use_cross_encoder: bool = CONFIG.get("reranker_use_cross_encoder", False)

    # -----------------------------
    # Embedding Helpers
    # -----------------------------
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a collection of texts, optionally normalizing the vectors.
        """
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
        """
        Embed a single query string.
        """
        return self.embed_texts([q])[0]

    # -----------------------------
    # Scoring
    # -----------------------------
    def embedding_rescore(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query embedding and document embeddings.
        """
        return _cosine_sim(q_emb.reshape(1, -1), d_emb)[0]

    # -----------------------------
    # Maximal Marginal Relevance
    # -----------------------------
    def mmr(
            self,
            q_emb: np.ndarray,
            d_emb: np.ndarray,
            top_k: int,
            lambda_param: Optional[float] = None
    ) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) selection.

        Args:
            q_emb: Query embedding.
            d_emb: Document embeddings.
            top_k: Number of documents to select.
            lambda_param: Balance between relevance and diversity.

        Returns:
            List of selected document indices.
        """
        lambda_param = lambda_param or self.mmr_lambda
        n = d_emb.shape[0]

        if top_k >= n:
            return list(range(n))

        selected = []
        remaining = set(range(n))

        query_sim = _cosine_sim(q_emb.reshape(1, -1), d_emb)[0]
        first = int(np.argmax(query_sim))
        selected.append(first)
        remaining.remove(first)

        pairwise = _cosine_sim(d_emb, d_emb)

        while len(selected) < top_k and remaining:
            best_idx = None
            best_score = -math.inf

            for idx in remaining:
                redundancy = max(pairwise[idx, s] for s in selected) if selected else 0.0
                score = lambda_param * query_sim[idx] - (1 - lambda_param) * redundancy

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # -----------------------------
    # Reranking
    # -----------------------------
    def rerank(
            self,
            query: str,
            candidates: Sequence[Dict[str, Any]],
            top_k: int = 5,
            use_mmr: bool = True,
            return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents using embedding similarity (and optional MMR).

        Args:
            query: Query text.
            candidates: List of dicts: { "text": str, "embedding": np.ndarray or None, ... }
            top_k: Number of results to return.
            use_mmr: Whether to use MMR selection.
            return_scores: Include normalized similarity scores in output.

        Returns:
            Reranked list of candidates.
        """
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        embeddings: List[Optional[np.ndarray]] = []
        missing: List[int] = []

        # Collect embeddings, embed missing ones
        for i, c in enumerate(candidates):
            emb = c.get("embedding")
            if emb is None:
                missing.append(i)
                embeddings.append(None)
            else:
                embeddings.append(np.asarray(emb))

        if missing:
            new_emb = self.embed_texts([texts[i] for i in missing])
            for j, idx in enumerate(missing):
                embeddings[idx] = new_emb[j]

        d_emb = np.vstack(embeddings)
        q_emb = self.embed_query(query)

        # Similarity scoring
        raw_scores = self.embedding_rescore(q_emb, d_emb)
        low, high = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - low) / (high - low + 1e-12)

        # Selection
        if use_mmr and top_k < len(candidates):
            chosen_idx = self.mmr(q_emb, d_emb, top_k)
        else:
            chosen_idx = np.argsort(-norm_scores)[:top_k].tolist()

        # Collect results
        results = []
        for idx in chosen_idx:
            item = dict(candidates[idx])
            item["_score"] = float(norm_scores[idx])
            results.append(item)

        # Sort by score descending
        results.sort(key=lambda x: x["_score"], reverse=True)

        if not return_scores:
            for r in results:
                r.pop("_score", None)

        return results
