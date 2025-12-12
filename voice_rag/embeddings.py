from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple
from pathlib import Path
import threading
import uuid

from .utils import die, info, warn
from .config import CONFIG

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None


# ============================================================================
# LocalEmbeddingIndex
# ============================================================================
# Lightweight wrapper around SentenceTransformers + ChromaDB for:
# - Embedding text
# - Persisting vectors locally
# - Efficient similarity search
# All components initialize lazily and degrade gracefully if unavailable.
# ============================================================================

@dataclass
class LocalEmbeddingIndex:
    embedding_model_name: str = CONFIG["embedding_model"]
    device: str = "cpu"
    chroma_dir: Path = Path(CONFIG["chroma_dir"])

    model: Any = None
    client: Any = None
    collection: Any = None

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ----------------------------------------------------------------------
    # Embedding Model Setup
    # ----------------------------------------------------------------------
    def init_model(self):
        if SentenceTransformer is None:
            die("sentence-transformers required. Install via `pip install sentence-transformers`.")
        if self.model is None:
            info(f"Loading embedding model '{self.embedding_model_name}' on {self.device}")

            # Configure model with normalization support
            try:
                self.model = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device
                )
            except Exception as e:
                die(f"Failed to load embedding model: {e}")

            # Ensure it uses CPU explicitly when needed
            try:
                self.model._first_device = self.device
            except Exception:
                pass

            info(f"Embedding model loaded successfully")

    # ----------------------------------------------------------------------
    # ChromaDB Setup
    # ----------------------------------------------------------------------
    def init_chroma(self):
        if chromadb is None:
            die("chromadb required. Install via `pip install chromadb`.")

        if self.client is None:
            try:
                settings = ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.chroma_dir)
                )
                # Prefer PersistentClient (new API), fall back to generic client
                try:
                    self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
                except Exception:
                    self.client = chromadb.Client(settings=settings)
            except Exception:
                self.client = chromadb.Client()

        coll_name = "rag_documents"
        try:
            self.collection = self.client.get_collection(coll_name)
            info(f"Opened Chroma collection '{coll_name}'")
        except Exception:
            try:
                self.collection = self.client.create_collection(
                    name=coll_name,
                    metadata={"source": "local_rag_documents"}
                )
                info(f"Created Chroma collection '{coll_name}'")
            except Exception as e:
                die(f"Could not create Chroma collection: {e}")

    # ----------------------------------------------------------------------
    # Encode Text to Embeddings
    # ----------------------------------------------------------------------
    def embed_texts(
            self,
            texts: Sequence[str],
            batch_size: Optional[int] = None,
            show_progress: Optional[bool] = None,
            normalize: Optional[bool] = None
    ) -> np.ndarray:
        """Embed a sequence of texts into vectors.

        Args:
            texts: Sequence of text strings to embed
            batch_size: Batch size for encoding (defaults to config)
            show_progress: Whether to show progress bar (defaults to config)
            normalize: Whether to normalize embeddings (defaults to config)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.model is None:
            self.init_model()
        if np is None:
            die("numpy required for embedding output.")

        # Use config defaults if not specified
        if batch_size is None:
            batch_size = CONFIG.get("embedding_batch_size", 64)
        if show_progress is None:
            show_progress = CONFIG.get("embedding_show_progress", True)
        if normalize is None:
            normalize = CONFIG.get("embedding_normalize", True)

        embeddings = []
        num_workers = CONFIG.get("embedding_num_workers", 2)

        # Log configuration if verbose
        if CONFIG.get("verbose_retrieval", False):
            info(f"Embedding {len(texts)} texts with batch_size={batch_size}, "
                 f"workers={num_workers}, normalize={normalize}")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Try modern API with all features
                try:
                    emb = self.model.encode(
                        batch,
                        show_progress_bar=show_progress and (i == 0),  # Only show on first batch
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                        batch_size=batch_size
                    )
                except TypeError:
                    # Fallback for older sentence-transformers versions
                    emb = np.asarray(
                        self.model.encode(
                            batch,
                            convert_to_numpy=True,
                            batch_size=batch_size
                        )
                    )

                    # Manual normalization if requested
                    if normalize:
                        norms = np.linalg.norm(emb, axis=1, keepdims=True)
                        norms[norms == 0] = 1e-12
                        emb = emb / norms

            except Exception as e:
                warn(f"Embedding error on batch {i//batch_size}: {e}")
                # Fallback: create zero embeddings
                try:
                    emb = np.zeros((len(batch), self.model.get_sentence_embedding_dimension()), dtype=np.float32)
                except Exception:
                    emb = np.zeros((len(batch), 384), dtype=np.float32)  # Common default dimension

            embeddings.append(emb)

        result = np.vstack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)

        if CONFIG.get("verbose_retrieval", False):
            info(f"Embedding complete: shape={result.shape}")

        return result

    # ----------------------------------------------------------------------
    # Add Text Chunks to ChromaDB
    # ----------------------------------------------------------------------
    def add_texts(
            self,
            texts: List[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            batch_size: Optional[int] = None
    ):
        """Add text chunks to ChromaDB with embeddings.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of unique IDs (generated if not provided)
            batch_size: Batch size for processing (defaults to config)
        """
        if self.collection is None:
            self.init_chroma()

        ids = ids or [str(uuid.uuid4()) for _ in texts]

        if batch_size is None:
            batch_size = CONFIG.get("embedding_batch_size", 64)

        # Skip IDs already present in the DB
        try:
            existing = set(self.collection.get(ids=ids, include=["ids"]).get("ids", []))
        except Exception:
            existing = set()

        new_texts, new_ids, new_metas = [], [], []
        for i, doc_id in enumerate(ids):
            if doc_id not in existing:
                new_texts.append(texts[i])
                new_ids.append(doc_id)
                new_metas.append(metadatas[i] if metadatas else {})

        if not new_texts:
            info("No new texts to index.")
            return

        info(f"Indexing {len(new_texts)} new documents...")

        # Process in batches with progress indication
        show_progress = CONFIG.get("embedding_show_progress", True)
        for i in range(0, len(new_texts), batch_size):
            batch_t = new_texts[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]
            batch_meta = new_metas[i:i + batch_size]

            if show_progress:
                progress = f"[{min(i+batch_size, len(new_texts))}/{len(new_texts)}]"
                info(f"Processing batch {progress}")

            batch_emb = self.embed_texts(
                batch_t,
                batch_size=batch_size,
                show_progress=False  # Don't show nested progress bars
            )

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_t,
                    metadatas=batch_meta,
                    embeddings=batch_emb.tolist()
                )
            except Exception as e:
                warn(f"Chroma add failed for batch {i//batch_size}: {e}")

        info(f"Indexing complete: {len(new_texts)} documents added.")

    # ----------------------------------------------------------------------
    # Query Similar Documents
    # ----------------------------------------------------------------------
    def query(
            self,
            text: str,
            top_k: int = None,
            score_threshold: Optional[float] = None
    ) -> List[Tuple[str, dict, float]]:
        """Query for similar documents.

        Args:
            text: Query text
            top_k: Number of results to return (defaults to config max_retrieval_topk)
            score_threshold: Minimum similarity score (defaults to config)

        Returns:
            List of (document, metadata, score) tuples
        """
        if self.collection is None:
            self.init_chroma()
        if self.model is None:
            self.init_model()

        # Use config defaults if not specified
        if top_k is None:
            top_k = CONFIG.get("max_retrieval_topk", 5)
        if score_threshold is None:
            score_threshold = CONFIG.get("retrieval_score_threshold", 0.0)

        # Retrieve more initially for reranking
        initial_k = CONFIG.get("initial_retrieval_k", 20)
        initial_k = min(initial_k, 100)  # Cap at reasonable limit

        if CONFIG.get("verbose_retrieval", False):
            info(f"Querying with initial_k={initial_k}, target_k={top_k}, threshold={score_threshold}")

        q_emb = self.embed_texts([text], show_progress=False)[0]

        try:
            results = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=initial_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            warn(f"Chroma query failed: {e}")
            return []

        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            meta = meta or {}
            meta.setdefault(
                "id",
                f"{meta.get('source','?')}_{meta.get('chunk_index', i)}"
            )

            # Convert distance to similarity score (assuming cosine distance)
            # ChromaDB returns lower distances for more similar items
            # Convert to similarity: similarity = 1 - distance (for cosine in [0,2])
            similarity = 1.0 - (float(dist) / 2.0)

            # Apply score threshold
            if similarity >= score_threshold:
                out.append((doc, meta, similarity))

        # Sort by similarity (higher is better)
        out_sorted = sorted(out, key=lambda x: -x[2])

        if CONFIG.get("verbose_retrieval", False):
            info(f"Query returned {len(out_sorted)} results after threshold filtering")
            if out_sorted:
                info(f"Score range: [{out_sorted[-1][2]:.3f}, {out_sorted[0][2]:.3f}]")

        return out_sorted[:min(top_k, len(out_sorted))]

    # ----------------------------------------------------------------------
    # Collection Statistics
    # ----------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            self.init_chroma()

        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model_name,
                "chroma_dir": str(self.chroma_dir)
            }
        except Exception as e:
            warn(f"Failed to get collection stats: {e}")
            return {}

    # ----------------------------------------------------------------------
    # Clear Collection
    # ----------------------------------------------------------------------
    def clear(self):
        """Clear all documents from the collection."""
        if self.collection is None:
            self.init_chroma()

        try:
            # Get all IDs
            all_ids = self.collection.get(include=["ids"]).get("ids", [])
            if all_ids:
                self.collection.delete(ids=all_ids)
                info(f"Cleared {len(all_ids)} documents from collection")
            else:
                info("Collection is already empty")
        except Exception as e:
            warn(f"Failed to clear collection: {e}")