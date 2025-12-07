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
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return np.dot(a / a_norm, (b / b_norm).T)

# ---------------------------------------------------------
# Optimized Cross-Encoder (CPU/GPU auto)
# ---------------------------------------------------------
class MiniLMCrossEncoder(nn.Module):
    def __init__(self, model_name: str = CONFIG["cross_encoder"]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 192
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(self.device)

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
            self.use_traced = False

        if self.device.type == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 6))

    @torch.inference_mode()
    def forward(self, queries, docs, batch_size: int = 16):
        assert len(queries) == len(docs)
        all_scores = []
        for i in range(0, len(docs), batch_size):
            qs = queries[i:i + batch_size]
            ds = docs[i:i + batch_size]
            encoded = self.tokenizer(
                qs,
                ds,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = self.model(**encoded)
            else:
                out = self.model(**encoded)
            scores = out.logits.squeeze(-1)
            all_scores.append(scores)
        return torch.cat(all_scores, dim=0)

# ---------------------------------------------------------
# ReRanker (embedding + optional CE)
# ---------------------------------------------------------
class ReRanker:
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
        return self.embed_texts([q])[0]

    # -----------------------------
    # Cross-encoder scoring
    # -----------------------------
    def cross_encoder_rescore(self, query: str, docs: Sequence[str]) -> np.ndarray:
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
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]

        # 1. Cross-encoder rescoring
        if self.use_cross_encoder:
            raw = self.cross_encoder_rescore(query, texts)
            norm_scores = self._safe_normalize_scores(raw)
            chosen_idx = np.argsort(-norm_scores)[:top_k].tolist()
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
            score = float(norm_scores[idx])
            item["_score"] = score
            item["score"] = score  # <-- FIX: propagate score to agent
            results.append(item)

        results.sort(key=lambda x: x["_score"], reverse=True)

        if not return_scores:
            for r in results:
                r.pop("_score", None)

        return results
