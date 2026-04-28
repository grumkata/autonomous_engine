"""
orchestrator/tools/web.py — Web search and URL fetching tools.

Providers (in priority order):
  web_search:
    1. Tavily API (TAVILY_API_KEY) — best structured results, ~$0.01/search
    2. Serper API (SERPER_API_KEY) — Google results, ~$0.001/search
    3. DuckDuckGo (no key, free)   — fallback, no rate limits but less structured

  fetch_url:
    Uses httpx + BeautifulSoup.  No API key needed.
    Strips navigation, ads, scripts.  Returns clean article text.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AutonomousEngine/1.0; "
        "+https://github.com/autonomous_engine)"
    )
}


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

async def web_search(
    query: str,
    num_results: int = 5,
    project_id: str = "",
) -> dict[str, Any]:
    """
    Search the web and return structured results.
    Tries providers in order: Tavily → Serper → DuckDuckGo.
    """
    num_results = max(1, min(num_results, 10))

    from config import get_settings
    settings = get_settings()

    # ── Tavily ────────────────────────────────────────────────────────────
    tavily_key = getattr(settings, "tavily_api_key", "")
    if tavily_key:
        try:
            result = await _tavily_search(query, num_results, tavily_key)
            if result["success"]:
                return result
        except Exception as exc:
            log.warning("web_search.tavily_failed", error=str(exc))

    # ── Serper ────────────────────────────────────────────────────────────
    serper_key = getattr(settings, "serper_api_key", "")
    if serper_key:
        try:
            result = await _serper_search(query, num_results, serper_key)
            if result["success"]:
                return result
        except Exception as exc:
            log.warning("web_search.serper_failed", error=str(exc))

    # ── DuckDuckGo (free fallback) ────────────────────────────────────────
    try:
        return await _ddg_search(query, num_results)
    except Exception as exc:
        return {
            "tool": "web_search",
            "success": False,
            "error": f"All search providers failed. Last error: {exc}",
            "files_created": [],
        }


async def _tavily_search(query: str, num_results: int, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
                "include_answer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "score": r.get("score", 0),
        }
        for r in data.get("results", [])
    ]
    return {
        "tool": "web_search",
        "success": True,
        "query": query,
        "answer": data.get("answer", ""),
        "results": results,
        "provider": "tavily",
        "files_created": [],
    }


async def _serper_search(query: str, num_results: int, api_key: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
        )
        resp.raise_for_status()
        data = resp.json()

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", "")[:500],
        }
        for r in data.get("organic", [])
    ]
    return {
        "tool": "web_search",
        "success": True,
        "query": query,
        "answer": data.get("answerBox", {}).get("answer", ""),
        "results": results,
        "provider": "serper",
        "files_created": [],
    }


async def _ddg_search(query: str, num_results: int) -> dict:
    """DuckDuckGo search via their HTML endpoint (no API key needed)."""
    async with httpx.AsyncClient(
        timeout=15,
        headers=_HEADERS,
        follow_redirects=True,
    ) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        resp.raise_for_status()

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:num_results]:
            title_el = result.select_one(".result__title")
            url_el   = result.select_one(".result__url")
            snip_el  = result.select_one(".result__snippet")
            if title_el and url_el:
                url = url_el.get_text(strip=True)
                if not url.startswith("http"):
                    url = "https://" + url
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": url,
                    "snippet": snip_el.get_text(strip=True)[:500] if snip_el else "",
                })
    except ImportError:
        # BeautifulSoup not installed — return raw hint
        results = [{"title": "Install beautifulsoup4 for better results", "url": "", "snippet": ""}]

    return {
        "tool": "web_search",
        "success": True,
        "query": query,
        "answer": "",
        "results": results,
        "provider": "duckduckgo",
        "files_created": [],
    }


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

async def fetch_url(
    url: str,
    max_chars: int = 8000,
    project_id: str = "",
) -> dict[str, Any]:
    """
    Fetch a webpage and return cleaned text content.
    Strips scripts, styles, navigation, ads.
    """
    if not url.startswith(("http://", "https://")):
        return {
            "tool": "fetch_url",
            "success": False,
            "error": f"URL must start with http:// or https://, got: {url!r}",
            "files_created": [],
        }

    try:
        async with httpx.AsyncClient(
            timeout=20,
            headers=_HEADERS,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

        # Non-HTML responses — return metadata only
        if "text/html" not in content_type and "text/plain" not in content_type:
            return {
                "tool": "fetch_url",
                "success": True,
                "url": url,
                "content_type": content_type,
                "text": f"[Binary content: {content_type}, {len(resp.content)} bytes]",
                "title": "",
                "files_created": [],
            }

        text, title = _extract_text(resp.text)
        text = text[:max_chars]

        return {
            "tool": "fetch_url",
            "success": True,
            "url": url,
            "title": title,
            "text": text,
            "char_count": len(text),
            "files_created": [],
        }

    except httpx.HTTPStatusError as exc:
        return {
            "tool": "fetch_url",
            "success": False,
            "error": f"HTTP {exc.response.status_code}: {url}",
            "files_created": [],
        }
    except Exception as exc:
        return {
            "tool": "fetch_url",
            "success": False,
            "error": f"Failed to fetch {url}: {exc}",
            "files_created": [],
        }


def _extract_text(html: str) -> tuple[str, str]:
    """Extract clean text and title from HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "iframe", "noscript",
                          "[class*='ad']", "[class*='menu']", "[class*='cookie']"]):
            tag.decompose()

        # Try to find main content area
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find(class_="article") or
            soup.body or
            soup
        )

        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text, title

    except ImportError:
        # No BeautifulSoup — strip tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, ""
