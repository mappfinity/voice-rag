from dataclasses import dataclass
from typing import Sequence, Callable, Any, Optional, Dict, List
import numpy as np
import math
from .config import CONFIG

# -----------------------------
# Utility: cosine similarity
# -----------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors."""
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
# ReRanker class
# -----------------------------
@dataclass
class ReRanker:
    embed_fn: Callable[[Sequence[str]], np.ndarray]
    ollama: Optional[Any] = None
    normalize_embeddings: bool = True
    cross_encoder_batch_size: int = CONFIG.get("reranker_cross_batch_size", 4)
    mmr_lambda: float = CONFIG.get("mmr_lambda", 0.6)
    use_cross_encoder: bool = CONFIG.get("reranker_use_cross_encoder", False)

    # -----------------------------
    # Embed texts
    # -----------------------------
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
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
        return self.embed_texts([q])[0]

    # -----------------------------
    # Embedding-based scoring
    # -----------------------------
    def embedding_rescore(self, q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
        sims = _cosine_sim(q_emb.reshape(1, -1), d_emb)[0]
        return sims

    # -----------------------------
    # Maximal Marginal Relevance (MMR)
    # -----------------------------
    def mmr(self, q_emb: np.ndarray, d_emb: np.ndarray, top_k: int, lambda_param: Optional[float] = None) -> List[int]:
        if lambda_param is None:
            lambda_param = self.mmr_lambda

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
    # Rerank candidates
    # -----------------------------
    def rerank(
            self,
            query: str,
            candidates: Sequence[Dict[str, Any]],
            top_k: int = 5,
            use_mmr: bool = True,
            return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        embeddings: List[np.ndarray] = []
        missing_idx: List[int] = []

        # Collect embeddings, handle missing
        for i, c in enumerate(candidates):
            emb = c.get("embedding")
            if emb is None:
                missing_idx.append(i)
                embeddings.append(None)
            else:
                embeddings.append(np.asarray(emb))

        if missing_idx:
            new_emb = self.embed_texts([texts[i] for i in missing_idx])
            for j, idx in enumerate(missing_idx):
                embeddings[idx] = new_emb[j]

        d_emb = np.vstack(embeddings)
        q_emb = self.embed_query(query)

        # Compute normalized similarity scores
        embed_scores = self.embedding_rescore(q_emb, d_emb)
        low, high = embed_scores.min(), embed_scores.max()
        embed_norm = (embed_scores - low) / (high - low + 1e-12)

        # Select top_k using MMR or simple top-k
        if use_mmr and top_k < len(candidates):
            chosen_idx = self.mmr(q_emb, d_emb, top_k)
        else:
            order = np.argsort(-embed_norm)
            chosen_idx = order[:top_k].tolist()

        results = []
        for idx in chosen_idx:
            c = dict(candidates[idx])
            c["_score"] = float(embed_norm[idx])
            results.append(c)

        results.sort(key=lambda x: x["_score"], reverse=True)

        if not return_scores:
            for r in results:
                r.pop("_score", None)

        return results
