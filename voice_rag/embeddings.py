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
            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)

            # Ensure it uses CPU explicitly when needed
            try:
                self.model._first_device = self.device
            except Exception:
                pass

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
    def embed_texts(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            self.init_model()
        if np is None:
            die("numpy required for embedding output.")

        batch_size = batch_size or CONFIG.get("embedding_batch_size", 32)
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Prefer normalized embeddings (new ST API)
                try:
                    emb = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                except TypeError:
                    # Legacy fallback: normalize manually
                    emb = np.asarray(self.model.encode(batch, convert_to_numpy=True))
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-12
                    emb = emb / norms

            except Exception as e:
                warn(f"Embedding error: {e}")
                emb = np.zeros((len(batch), 768), dtype=np.float32)

            embeddings.append(emb)

        return np.vstack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)

    # ----------------------------------------------------------------------
    # Add Text Chunks to ChromaDB
    # ----------------------------------------------------------------------
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        if self.collection is None:
            self.init_chroma()

        ids = ids or [str(uuid.uuid4()) for _ in texts]

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

        batch_size = CONFIG.get("embedding_batch_size", 32)
        for i in range(0, len(new_texts), batch_size):
            batch_t = new_texts[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]
            batch_meta = new_metas[i:i + batch_size]

            batch_emb = self.embed_texts(batch_t, batch_size=batch_size)

            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_t,
                    metadatas=batch_meta,
                    embeddings=batch_emb.tolist()
                )
            except Exception as e:
                warn(f"Chroma add failed for batch: {e}")

        info("Indexing complete.")

    # ----------------------------------------------------------------------
    # Query Similar Documents
    # ----------------------------------------------------------------------
    def query(self, text: str, top_k: int = 4) -> List[Tuple[str, dict, float]]:
        if self.collection is None:
            self.init_chroma()
        if self.model is None:
            self.init_model()

        initial_k = min(CONFIG.get("initial_retrieval_k", 16), 64)
        q_emb = self.embed_texts([text])[0]

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
            out.append((doc, meta, float(dist)))

        # Sort by similarity (higher distance → more similar in Chroma cosine)
        out_sorted = sorted(out, key=lambda x: -x[2])
        return out_sorted[:min(top_k, len(out_sorted))]
