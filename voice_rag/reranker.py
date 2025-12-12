from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import CONFIG

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b.

    Ensures inputs are 2D and guards against zero norms.
    """
    if a is None or b is None:
        raise ValueError("Inputs to _cosine_sim must not be None")
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # safe norms
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12

    a_unit = a / a_norm
    b_unit = b / b_norm
    return np.dot(a_unit, b_unit.T)


# ---------------------------------------------------------
# Optimized Cross-Encoder (CPU/GPU auto)
# ---------------------------------------------------------
class MiniLMCrossEncoder(nn.Module):
    """Lightweight wrapper around a transformers sequence-classification model

    - Automatically places the model on CUDA if available
    - Attempts to torch.jit.trace on CPU to improve inference time when possible
    - Exposes a callable forward(queries, docs, batch_size)
    """

    def __init__(self, model_name: str = None):
        super().__init__()
        # Use upgraded cross-encoder model from config
        if model_name is None:
            model_name = CONFIG.get("cross_encoder", "cross-encoder/ms-marco-MiniLM-L-12-v2")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Upgraded max tokens from 48 to 96 for better context
        self.max_len = int(CONFIG.get("reranker_cross_max_tokens", 96))
        self.model_name = model_name

        # Lazy init tokenizer/model with defensive try/except
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval().to(self.device)
            logger.info(f"Loaded cross-encoder: {model_name} on {self.device}")
        except Exception as e:
            logger.exception("Failed to initialize cross-encoder model %s", model_name)
            raise

        # Try tracing on CPU to speed up small CPU deployments
        self.use_traced = False
        if self.device.type == "cpu":
            try:
                example_input = self.tokenizer("example", "doc", return_tensors="pt", max_length=8, truncation=True)
                # Move example inputs to cpu tensors of correct dtype
                example_input = {k: v for k, v in example_input.items()}
                self.model = torch.jit.trace(self.model, example_input)
                self.use_traced = True
                logger.info("Using traced cross-encoder on CPU")
            except Exception:
                logger.info("Tracing cross-encoder not available; using eager model")
                self.use_traced = False

        # Limit CPU threads for inference to avoid noisy oversubscription
        if self.device.type == "cpu":
            try:
                torch.set_num_threads(min(torch.get_num_threads(), 6))
            except Exception:
                pass

    @torch.inference_mode()
    def forward(self, queries: Sequence[str], docs: Sequence[str], batch_size: int = None) -> torch.Tensor:
        """Score (query, doc) pairs using the cross-encoder.

        Returns:
            Tensor of shape (len(docs),) with float scores.
        """
        assert len(queries) == len(docs), "queries and docs must be same length"

        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = CONFIG.get("reranker_cross_batch_size", 8)

        all_scores = []

        for i in range(0, len(docs), batch_size):
            qs = queries[i:i + batch_size]
            ds = docs[i:i + batch_size]

            # tokenizer will handle batching
            try:
                encoded = self.tokenizer(
                    qs,
                    ds,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
            except Exception:
                # fallback to pairwise encoding if tokenizer fails on batch
                enc_list = [self.tokenizer(q, d, truncation=True, max_length=self.max_len, return_tensors="pt") for q, d in zip(qs, ds)]
                # collate manually
                encoded = {k: torch.cat([e[k] for e in enc_list], dim=0) for k in enc_list[0].keys()}

            # move to device
            try:
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            except Exception:
                # if encoded contains non-tensor objects, ignore
                pass

            if self.device.type == "cuda":
                # use mixed precision on GPU if available
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = self.model(**encoded)
            else:
                out = self.model(**encoded)

            logits = out.logits
            # ensure shape (batch, ) or (batch, 1)
            if logits.ndim == 2 and logits.shape[1] == 1:
                scores = logits.squeeze(-1)
            elif logits.ndim == 2 and logits.shape[1] > 1:
                # assume binary classification with logits for each class; take positive class
                scores = logits[:, 1]
            else:
                scores = logits

            all_scores.append(scores.detach().cpu())

        if all_scores:
            return torch.cat(all_scores, dim=0)
        return torch.tensor([])


# ---------------------------------------------------------
# ReRanker (embedding + optional CE)
# ---------------------------------------------------------
class ReRanker:
    """ReRanker that supports:
    - embedding-based cosine similarity
    - optional MMR selection
    - optional cross-encoder rescoring

    Defensive and robust: normalizes inputs/outputs, catches and logs unexpected shapes,
    and returns a canonical list[dict] with 'text','meta','score' fields.
    """

    def __init__(
            self,
            embed_fn: Callable[[Sequence[str]], Union[np.ndarray, List[List[float]]]],
            ollama: Optional[Any] = None,
            normalize_embeddings: bool = True,
            cross_encoder_batch_size: int = None,
            mmr_lambda: float = None,
            use_cross_encoder: bool = None,
            cross_encoder: Optional[nn.Module] = None,
    ):
        self.embed_fn = embed_fn
        self.ollama = ollama
        self.normalize_embeddings = normalize_embeddings

        # Use config defaults if not explicitly provided
        if cross_encoder_batch_size is None:
            cross_encoder_batch_size = CONFIG.get("reranker_cross_batch_size", 8)
        if mmr_lambda is None:
            mmr_lambda = CONFIG.get("mmr_lambda", 0.7)
        if use_cross_encoder is None:
            use_cross_encoder = CONFIG.get("reranker_use_cross_encoder", True)

        self.cross_encoder_batch_size = int(cross_encoder_batch_size)
        self.mmr_lambda = float(mmr_lambda)
        self.use_cross_encoder = bool(use_cross_encoder)

        # Log configuration for debugging
        if CONFIG.get("verbose_retrieval", False):
            logger.info(f"ReRanker initialized: cross_encoder={self.use_cross_encoder}, "
                        f"mmr_lambda={self.mmr_lambda}, batch_size={self.cross_encoder_batch_size}")

        # Initialize cross_encoder lazily and defensively
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        elif self.use_cross_encoder:
            try:
                self.cross_encoder = MiniLMCrossEncoder()
            except Exception as e:
                logger.exception("Failed to initialize configured cross-encoder; disabling cross-encoder")
                self.cross_encoder = None
                self.use_cross_encoder = False
        else:
            self.cross_encoder = None

        if self.cross_encoder is not None:
            try:
                self.cross_encoder.eval()
            except Exception:
                pass

    # -----------------------------
    # Embedding helpers
    # -----------------------------
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts using embed_fn and return a 2D numpy array (n, d).

        Defensive: accepts lists, tuples or numpy arrays returned by embed_fn.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        try:
            arr = self.embed_fn(list(texts))
        except Exception as e:
            logger.exception("embed_fn raised an exception")
            raise

        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if self.normalize_embeddings and arr.size != 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            arr = arr / norms

        return arr

    def embed_query(self, q: str) -> np.ndarray:
        e = self.embed_texts([q])
        if e.size == 0:
            return np.zeros((1, 0), dtype=np.float32)
        return e[0]

    # -----------------------------
    # Cross-encoder scoring
    # -----------------------------
    def cross_encoder_rescore(self, query: str, docs: Sequence[str]) -> np.ndarray:
        """Rescore docs using the cross-encoder. Returns numpy array of scores (len(docs),)

        If cross-encoder is unavailable, raises RuntimeError.
        """
        if self.cross_encoder is None:
            raise RuntimeError("Cross-encoder is not initialized")

        scores = []
        bs = max(1, int(self.cross_encoder_batch_size))

        if CONFIG.get("verbose_retrieval", False):
            logger.info(f"Cross-encoder rescoring {len(docs)} documents with batch_size={bs}")

        try:
            for i in range(0, len(docs), bs):
                chunk = docs[i:i+bs]
                # cross_encoder expects (queries, docs) of equal length
                s = self.cross_encoder([query] * len(chunk), chunk, batch_size=bs)
                scores.append(s.cpu().numpy())
        except Exception:
            logger.exception("Cross-encoder rescoring failed")
            raise

        if scores:
            result = np.concatenate(scores, axis=0)
            if CONFIG.get("verbose_retrieval", False):
                logger.info(f"Cross-encoder scores: mean={result.mean():.3f}, std={result.std():.3f}")
            return result
        return np.array([])

    # -----------------------------
    # Normalization
    # -----------------------------
    def _safe_normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        if scores.size == 0:
            return scores
        lo, hi = float(scores.min()), float(scores.max())
        if hi == lo:
            # all equal -> return zeros
            return np.zeros_like(scores, dtype=float)
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
        """Select top_k documents using Maximal Marginal Relevance.

        Args:
            q_emb: Query embedding
            d_emb: Document embeddings matrix
            top_k: Number of documents to select
            lambda_param: Trade-off between relevance and diversity (higher = more relevance)
        """
        lambda_param = float(lambda_param) if lambda_param is not None else float(self.mmr_lambda)

        if CONFIG.get("verbose_retrieval", False):
            logger.info(f"MMR selection with lambda={lambda_param} (relevance vs diversity trade-off)")

        if d_emb is None or d_emb.size == 0:
            return []
        n = int(d_emb.shape[0])
        if top_k >= n:
            return list(range(n))

        selected: List[int] = []
        remaining = set(range(n))
        q_emb_r = q_emb.reshape(1, -1)
        sim_q = _cosine_sim(q_emb_r, d_emb)[0]
        sim_docs = _cosine_sim(d_emb, d_emb)

        # pick the highest-sim document to query first
        first = int(np.argmax(sim_q))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -1e12
            for idx in remaining:
                redundancy = max(sim_docs[idx, s] for s in selected) if selected else 0.0
                score = lambda_param * sim_q[idx] - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        if CONFIG.get("verbose_retrieval", False):
            logger.info(f"MMR selected {len(selected)} documents")

        return selected

    # -----------------------------
    # Main reranker
    # -----------------------------
    def rerank(
            self,
            query: str,
            candidates: Sequence[Dict[str, Any]],
            top_k: int = None,
            use_mmr: bool = None,
            return_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate dicts and return a list of dicts with 'text','meta','score'.

        Args:
            query: user query string
            candidates: sequence of {'text':..., 'meta':..., 'embedding':..., ...}
            top_k: number of final items to return (defaults to config max_retrieval_topk)
            use_mmr: whether to apply MMR selection (defaults to config mmr_use)
            return_scores: if True, keep the internal _score key for debugging
        """
        if not candidates:
            return []

        # Use config defaults if not specified
        if top_k is None:
            top_k = CONFIG.get("max_retrieval_topk", 5)
        if use_mmr is None:
            use_mmr = CONFIG.get("mmr_use", True)

        if CONFIG.get("verbose_retrieval", False):
            logger.info(f"Reranking {len(candidates)} candidates, target top_k={top_k}, use_mmr={use_mmr}")

        # Normalize candidate entries defensively
        normalized_cands = []
        texts: List[str] = []
        for i, c in enumerate(candidates):
            try:
                if isinstance(c, dict):
                    text = str(c.get("text", ""))
                    meta = dict(c.get("meta", {}) or {})
                    emb = c.get("embedding")
                else:
                    # fallback for unexpected shapes
                    text = str(c)
                    meta = {}
                    emb = None
            except Exception:
                text = str(c)
                meta = {}
                emb = None

            normalized_cands.append({"text": text, "meta": meta, "embedding": emb})
            texts.append(text)

        # 1) If using cross-encoder, prefer that path
        if self.use_cross_encoder and self.cross_encoder is not None:
            try:
                if CONFIG.get("verbose_retrieval", False):
                    logger.info("Using cross-encoder reranking path")

                raw_scores = self.cross_encoder_rescore(query, texts)
                norm_scores = self._safe_normalize_scores(raw_scores)
                # choose top_k by normalized score
                chosen_idx = np.argsort(-norm_scores)[:int(top_k)].tolist()
                results = []
                for idx in chosen_idx:
                    item = dict(normalized_cands[idx])
                    score = float(norm_scores[idx]) if idx < len(norm_scores) else 0.0
                    item["score"] = score
                    item["_score"] = score
                    results.append(item)

                # sort descending
                results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

                if CONFIG.get("verbose_retrieval", False):
                    logger.info(f"Cross-encoder reranking complete, returned {len(results)} results")

                if not return_scores:
                    for r in results:
                        r.pop("_score", None)
                return results
            except Exception:
                logger.exception("Cross-encoder path failed; falling back to embedding-based rerank")
                # continue to embedding path

        # 2) Embedding-based reranking (cosine similarity + optional MMR)
        if CONFIG.get("verbose_retrieval", False):
            logger.info("Using embedding-based reranking path")

        # Build embeddings matrix, computing missing ones via embed_fn
        embeddings: List[np.ndarray] = []
        missing_indices: List[int] = []
        for i, c in enumerate(normalized_cands):
            emb = c.get("embedding")
            if emb is None:
                missing_indices.append(i)
                embeddings.append(None)
            else:
                try:
                    e = np.asarray(emb)
                    if e.ndim == 1:
                        embeddings.append(e)
                    else:
                        embeddings.append(e.reshape(-1))
                except Exception:
                    embeddings.append(None)
                    missing_indices.append(i)

        if missing_indices:
            if CONFIG.get("verbose_retrieval", False):
                logger.info(f"Computing embeddings for {len(missing_indices)} missing candidates")
            try:
                computed = self.embed_texts([texts[i] for i in missing_indices])
                # assign them back
                for j, idx in enumerate(missing_indices):
                    embeddings[idx] = computed[j]
            except Exception:
                logger.exception("Failed to compute embeddings for missing candidates")
                # fallback: set zero vectors so they rank low
                for idx in missing_indices:
                    embeddings[idx] = np.zeros((computed.shape[1],), dtype=float) if 'computed' in locals() and computed.size else np.zeros((1,), dtype=float)

        # stack embeddings; ensure shape (n, d)
        try:
            d_emb = np.vstack(embeddings)
        except Exception:
            # last ditch: convert to array of objects then to float array
            d_emb = np.asarray([np.asarray(e).reshape(-1) for e in embeddings])

        q_emb = self.embed_query(query)

        # handle degenerate cases
        if d_emb.size == 0 or q_emb.size == 0:
            # return first top_k candidates with 0.0 scores
            out = []
            for c in normalized_cands[:int(top_k)]:
                it = dict(c)
                it["score"] = 0.0
                if return_scores:
                    it["_score"] = 0.0
                out.append(it)
            return out

        raw = _cosine_sim(q_emb.reshape(1, -1), d_emb)[0]
        norm_scores = self._safe_normalize_scores(raw)

        n = len(normalized_cands)
        if use_mmr and int(top_k) < n:
            try:
                chosen_idx = self.mmr(q_emb, d_emb, int(top_k))
            except Exception:
                logger.exception("MMR selection failed; falling back to top-k by score")
                chosen_idx = np.argsort(-norm_scores)[:int(top_k)].tolist()
        else:
            if CONFIG.get("verbose_retrieval", False):
                logger.info("Using simple top-k selection (no MMR)")
            chosen_idx = np.argsort(-norm_scores)[:int(top_k)].tolist()

        results = []
        for idx in chosen_idx:
            item = dict(normalized_cands[idx])
            score = float(norm_scores[idx]) if idx < len(norm_scores) else 0.0
            item["score"] = score
            item["_score"] = score
            results.append(item)

        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

        if CONFIG.get("verbose_retrieval", False):
            avg_score = np.mean([r.get("_score", 0.0) for r in results]) if results else 0.0
            logger.info(f"Embedding reranking complete, {len(results)} results, avg_score={avg_score:.3f}")

        if not return_scores:
            for r in results:
                r.pop("_score", None)

        return results