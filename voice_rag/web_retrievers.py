
"""
web_retrievers.py

Provides optional web retrieval (Wikipedia, arXiv, Tavily).
Each retriever returns a list of dicts compatible with the LocalRAGAgent:
    {"text": "...", "meta": {...}}

Tavily is disabled unless TAVILY_API_KEY env var is defined.
Network errors are caught and returned as empty lists (safe failure).
"""

from typing import List, Dict, Any, Optional
import logging
import os
import time
import json
import html
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = int(os.getenv("WEBRETRIEVERS_TIMEOUT", "15"))
_MAX_RETRIES = int(os.getenv("WEBRETRIEVERS_RETRIES", "2"))

_DEFAULT_HEADERS = {
    "User-Agent": os.getenv("WEBRETRIEVERS_UA", "LocalVoiceRAG/1.0 (+https://github.com/example)")
}

def _make_session(retries: int = _MAX_RETRIES, backoff: float = 0.3) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
              timeout: int = _REQUEST_TIMEOUT, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
    try:
        final_headers = dict(_DEFAULT_HEADERS)
        if headers:
            final_headers.update(headers)
        sess = session or _make_session()
        r = sess.get(url, params=params, headers=final_headers, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning("web_retrievers: GET %s failed: %s", url, e)
        return None

# -------------------------------
# Wikipedia Retriever
# -------------------------------

def _extract_wikipedia_page(page: dict) -> Optional[Dict[str, Any]]:
    try:
        extract = page.get("extract", "")
        if not extract or not isinstance(extract, str):
            return None
        title = page.get("title", "")
        fullurl = page.get("fullurl") or page.get("canonicalurl") or ""
        pageid = page.get("pageid") or page.get("id") or None
        meta = {
            "source": "wikipedia",
            "title": title,
            "url": fullurl,
            "page_id": pageid,
            "retrieved_at": time.time()
        }
        return {"text": extract, "meta": meta}
    except Exception as e:
        logger.exception("_extract_wikipedia_page failed: %s", e)
        return None

def retrieve_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not query:
        return results

    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
    }

    resp = _safe_get(search_url, params=search_params)
    if resp is None:
        return results

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("retrieve_wikipedia: invalid JSON response: %s", e)
        return results

    hits = data.get("query", {}).get("search", [])
    titles = [h.get("title") for h in hits if h.get("title")]
    if not titles:
        return results

    extract_params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": True,
        "titles": "|".join(titles),
        "inprop": "url",
        "format": "json",
    }

    extract_resp = _safe_get(search_url, params=extract_params)
    if extract_resp is None:
        return results

    try:
        pages = extract_resp.json().get("query", {}).get("pages", {})
    except Exception as e:
        logger.warning("retrieve_wikipedia: failed parsing extracts JSON: %s", e)
        return results

    for page in pages.values():
        v = _extract_wikipedia_page(page)
        if v:
            results.append(v)
            if len(results) >= max_results:
                break

    return results

# -------------------------------
# arXiv Retriever
# -------------------------------

def retrieve_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not query:
        return results

    # Fixed unterminated string literal
    tokens = [t.strip('.,!?()[]"\'"\n\r') for t in query.split()]
    tokens = [t for t in tokens if len(t) >= 4]
    if not tokens:
        return results

    q = " OR ".join(f"all:{t}" for t in tokens)
    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": q,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    resp = _safe_get(base, params=params)
    if resp is None:
        return results

    try:
        root = ET.fromstring(resp.text)
    except Exception as e:
        logger.warning("retrieve_arxiv: failed to parse XML: %s", e)
        return results

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    seen_ids = set()
    for entry in entries:
        try:
            id_elem = entry.find("atom:id", ns)
            article_id = id_elem.text.strip() if id_elem is not None and id_elem.text else None
            if not article_id or article_id in seen_ids:
                continue
            seen_ids.add(article_id)

            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            summary_raw = summary_elem.text if summary_elem is not None and summary_elem.text else ""
            summary = " ".join(html.unescape(summary_raw).split())

            if len(summary) < 20:
                continue

            published_elem = entry.find("atom:published", ns)
            published = published_elem.text if published_elem is not None and published_elem.text else None

            meta = {
                "source": "arxiv",
                "title": title,
                "id": article_id,
                "published": published,
                "retrieved_at": time.time()
            }

            text_blob = f"{title}\n\n{summary}"
            results.append({"text": text_blob, "meta": meta})

            if len(results) >= max_results:
                break

        except Exception as e:
            logger.warning("web_retrievers: arXiv entry parse failed: %s", e)
            continue

    return results

# -------------------------------
# Tavily Retriever (unchanged)
# -------------------------------

def retrieve_tavily(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not query:
        return results

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return results

    tavily_endpoint = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }

    try:
        sess = _make_session()
        r = sess.post(tavily_endpoint, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("web_retrievers: Tavily request failed: %s", e)
        return results

    items = data.get("results") or []
    for it in items[:max_results]:
        if not isinstance(it, dict):
            continue
        text = it.get("content") or it.get("text") or it.get("snippet") or ""
        meta: Dict[str, Any] = {
            "source": "tavily",
            "retrieved_at": time.time(),
            "url": it.get("url"),
            "title": it.get("title"),
        }
        if "score" in it:
            try:
                meta["score"] = float(it.get("score"))
            except Exception:
                meta["score"] = None
        existing_meta = it.get("metadata") or {}
        if isinstance(existing_meta, dict):
            for k, v in existing_meta.items():
                if k not in meta:
                    meta[k] = v
        results.append({"text": text, "meta": meta})

    return results

# -------------------------------
# Aggregator
# -------------------------------

def retrieve_all(query: str, use_wiki: bool = False, use_arxiv: bool = False,
                 use_tavily: bool = False, max_per_source: int = 3) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []
    if not query:
        return aggregated

    if use_wiki:
        try:
            aggregated.extend(retrieve_wikipedia(query, max_per_source))
        except Exception:
            pass

    if use_arxiv:
        try:
            aggregated.extend(retrieve_arxiv(query, max_per_source))
        except Exception:
            pass

    if use_tavily:
        try:
            aggregated.extend(retrieve_tavily(query, max_per_source))
        except Exception:
            pass

    return aggregated