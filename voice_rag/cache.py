# voice_rag/cache.py
"""
RAG response caching with TTL, size limits, and LRU eviction.
"""
import hashlib
import json
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class RAGCache:
    """
    LRU cache for RAG responses with TTL and memory size limits.
    """

    def __init__(
            self,
            max_size_mb: int = 500,
            ttl_seconds: int = 3600,
            persist_path: Optional[Path] = None
    ):
        """
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live in seconds (cache entry expiration)
            persist_path: Path to save/load cache (None = memory only)
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size_bytes = 0

        # Load from disk if available
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate the size of an object in bytes.
        Uses sys.getsizeof with recursive handling for nested structures.
        """
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(self._estimate_size(k) + self._estimate_size(v)
                        for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, str):
            size += len(obj.encode('utf-8'))

        return size

    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key from normalized query.
        """
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available and not expired.
        
        Returns:
            Dict with 'response' and 'sources' keys, or None if not found/expired
        """
        cache_key = self._get_cache_key(query)

        if cache_key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[cache_key]

        # Check TTL expiration
        if self.ttl_seconds > 0:
            age = time.time() - entry["timestamp"]
            if age > self.ttl_seconds:
                # Expired - remove and return None
                self._remove_entry(cache_key)
                self.misses += 1
                logger.debug(f"Cache entry expired (age: {age:.0f}s, TTL: {self.ttl_seconds}s)")
                return None

        # Move to end (LRU - most recently used)
        self.cache.move_to_end(cache_key)
        self.hits += 1

        hit_rate = self.get_hit_rate()
        logger.info(f"✓ Cache hit! (hit rate: {hit_rate:.1%}, size: {self._format_size(self.current_size_bytes)})")

        return {
            "response": entry["response"],
            "sources": entry["sources"]
        }

    def set(
            self,
            query: str,
            response: str,
            sources: List[Tuple[str, Dict, float]]
    ):
        """
        Store response in cache with automatic eviction if needed.
        
        Args:
            query: User query string
            response: Generated response text
            sources: List of (text, meta, score) tuples
        """
        cache_key = self._get_cache_key(query)

        # Create entry
        entry = {
            "query": query,
            "response": response,
            "sources": sources,
            "timestamp": time.time()
        }

        # Estimate entry size
        entry_size = self._estimate_size(entry)

        # Check if single entry exceeds max size
        if entry_size > self.max_size_bytes:
            logger.warning(
                f"Cannot cache entry: size {self._format_size(entry_size)} "
                f"exceeds max cache size {self._format_size(self.max_size_bytes)}"
            )
            return

        # If key already exists, remove its size from current total
        if cache_key in self.cache:
            old_entry = self.cache[cache_key]
            old_size = self._estimate_size(old_entry)
            self.current_size_bytes -= old_size

        # Evict entries until we have enough space
        while (self.current_size_bytes + entry_size > self.max_size_bytes and
               len(self.cache) > 0):
            # Remove oldest entry (LRU)
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
            self.evictions += 1

        # Add new entry
        self.cache[cache_key] = entry
        self.current_size_bytes += entry_size

        logger.debug(
            f"Cached response (size: {self._format_size(entry_size)}, "
            f"total: {self._format_size(self.current_size_bytes)}/{self._format_size(self.max_size_bytes)})"
        )

        # Persist if configured
        if self.persist_path:
            self._save_to_disk()

    def _remove_entry(self, cache_key: str):
        """Remove an entry and update size tracking."""
        if cache_key in self.cache:
            entry = self.cache.pop(cache_key)
            entry_size = self._estimate_size(entry)
            self.current_size_bytes -= entry_size
            logger.debug(f"Evicted cache entry (freed: {self._format_size(entry_size)})")

    def clear(self):
        """Clear all cache entries and reset stats."""
        self.cache.clear()
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        if self.persist_path:
            self._save_to_disk()

        logger.info("Cache cleared")

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "size": len(self.cache),
            "size_bytes": self.current_size_bytes,
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": (self.current_size_bytes / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.get_hit_rate(),
            "ttl_seconds": self.ttl_seconds
        }

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def _save_to_disk(self):
        """Persist cache to disk."""
        if not self.persist_path:
            return

        try:
            # Convert to serializable format
            cache_data = {
                "cache": dict(self.cache),
                "stats": {
                    "hits": self.hits,
                    "misses": self.misses,
                    "evictions": self.evictions,
                    "current_size_bytes": self.current_size_bytes
                },
                "metadata": {
                    "max_size_bytes": self.max_size_bytes,
                    "ttl_seconds": self.ttl_seconds,
                    "saved_at": time.time()
                }
            }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then atomic rename
            temp_path = self.persist_path.with_suffix('.tmp')
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            temp_path.replace(self.persist_path)
            logger.debug(f"Cache persisted to {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to persist cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Restore cache (maintaining order)
            loaded_cache = cache_data.get("cache", {})
            self.cache = OrderedDict(loaded_cache)

            # Restore stats
            stats = cache_data.get("stats", {})
            self.hits = stats.get("hits", 0)
            self.misses = stats.get("misses", 0)
            self.evictions = stats.get("evictions", 0)
            self.current_size_bytes = stats.get("current_size_bytes", 0)

            # Validate loaded entries against TTL
            metadata = cache_data.get("metadata", {})
            saved_at = metadata.get("saved_at", 0)
            current_time = time.time()

            # Remove expired entries
            expired_keys = []
            for key, entry in self.cache.items():
                age = current_time - entry.get("timestamp", saved_at)
                if self.ttl_seconds > 0 and age > self.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            logger.info(
                f"Loaded cache from disk: {len(self.cache)} entries "
                f"({self._format_size(self.current_size_bytes)}, "
                f"{len(expired_keys)} expired entries removed)"
            )
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            self.cache = OrderedDict()
            self.current_size_bytes = 0