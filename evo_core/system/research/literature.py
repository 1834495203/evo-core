"""Concrete review triggers and an injectable provider boundary; no invented citations."""

from __future__ import annotations

from typing import Any

from ..ports import LiteratureProvider
from .notebook import ResearchNotebook


class LiteratureReviewPolicy:
    def __init__(self, experiments_between_reviews: int = 3, max_consecutive_failures: int = 2):
        if experiments_between_reviews < 1:
            raise ValueError("Review interval must be positive")
        if max_consecutive_failures < 1:
            raise ValueError("Failure review threshold must be positive")
        self.interval = experiments_between_reviews
        self.max_consecutive_failures = max_consecutive_failures

    def reasons(self, snapshot: dict) -> list[str]:
        study, notes = snapshot["study"], snapshot["records"]
        reviews = [n for n in notes if n["kind"] == "source"]
        if not reviews:
            return ["No read literature has been recorded for this study."]
        latest = reviews[-1]
        reasons = []
        if latest["details"].get("plan_version") != study["plan_version"]:
            reasons.append("The direction changed after the last literature review.")
        newer_jobs = [j for j in snapshot["jobs"] if j["created_at"] > latest["created_at"]]
        if len(newer_jobs) >= self.interval:
            reasons.append("The configured experiment interval calls for another literature review.")

        recent_finished = [
            j for j in newer_jobs
            if j["status"] in {"failed", "succeeded"}
        ][-self.max_consecutive_failures:]
        if (
            len(recent_finished) >= self.max_consecutive_failures
            and all(j["status"] == "failed" or j.get("validity") == "invalid" for j in recent_finished)
        ):
            reasons.append(
                "Recent experiments failed or were invalidated consecutively; review literature to avoid a dead-end or known pitfall."
            )

        if study["phase"] == "method":
            observations = [n for n in notes if n["kind"] == "observation"]
            if latest["details"].get("plan_version") != study["plan_version"] or (
                observations and latest["created_at"] < observations[-1]["created_at"]
            ):
                reasons.append(
                    "Review related literature before finalizing a new method to verify novelty and baseline settings."
                )

        if study["phase"] == "reporting":
            reasons.append("Check the nearest prior work and citations before delivery.")
        return reasons


class PluginLiteratureProvider:
    """Adapter implementing LiteratureProvider using OpenAlex/arXiv and full-text paper extraction."""

    def __init__(self, *, timeout: float = 25.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        import xml.etree.ElementTree as ET
        from ...plugins.literature import _http_get, _reconstruct_openalex_abstract

        results: list[dict[str, Any]] = []
        try:
            resp = _http_get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max(1, min(20, int(limit))),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)
                for entry in root.findall("atom:entry", ns):
                    results.append(
                        {
                            "source": "arxiv",
                            "title": " ".join((entry.findtext("atom:title", "", ns) or "").split()),
                            "url": (entry.findtext("atom:id", "", ns) or "").strip(),
                            "published": (entry.findtext("atom:published", "", ns) or "")[:10],
                            "abstract": " ".join((entry.findtext("atom:summary", "", ns) or "").split()),
                        }
                    )
        except Exception:
            pass

        if len(results) < limit:
            try:
                resp = _http_get(
                    "https://api.openalex.org/works",
                    params={"search": query, "per-page": max(1, limit - len(results))},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    for work in resp.json().get("results") or []:
                        oa = work.get("open_access") or {}
                        results.append(
                            {
                                "source": "openalex",
                                "title": work.get("title") or "(untitled)",
                                "url": oa.get("oa_url") or work.get("doi") or work.get("id") or "",
                                "published": str(work.get("publication_year") or ""),
                                "abstract": _reconstruct_openalex_abstract(work.get("abstract_inverted_index")),
                            }
                        )
            except Exception:
                pass
        return results[:limit]

    def read(self, url: str) -> dict[str, Any]:
        from ...plugins.literature import fetch_paper_content

        paper = fetch_paper_content(url, timeout=self.timeout)
        return {
            "title": paper["title"],
            "url": paper["url"],
            "excerpt": paper["text"][:4000],
            "locator": paper["locator"],
            "reading_depth": paper["reading_depth"],
            "content": paper["text"],
        }


class LiteratureService:
    def __init__(self, provider: LiteratureProvider, notebook: ResearchNotebook):
        self.provider, self.notebook = provider, notebook

    def search(self, query: str, limit: int = 5, *, study_id: str | None = None) -> list[dict]:
        results = self.provider.search(query, limit)
        if study_id:
            self.notebook.record(study_id, "search", query, details={"results": results})
        return results

    def read_and_record(self, study_id: str, url: str):
        source = self.provider.read(url)
        note = self.notebook.source(study_id, source["title"], url, source["excerpt"], source["locator"],
                                    reading_depth=source.get("reading_depth", "abstract_only"),
                                    content=source.get("content", source["excerpt"]))
        return note
