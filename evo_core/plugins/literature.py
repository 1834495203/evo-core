"""Resilient literature & codebase discovery plugin (arXiv, OpenAlex, GitHub, Semantic Scholar, full-text reader)."""

from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ..protocol import BasePlugin, HookContext, RunState, ToolDef

_USER_AGENT = "EvoCore-Research-Agent/0.1 (mailto:research@example.org)"
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)(?:\.pdf)?",
    re.IGNORECASE,
)


def _http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 25.0,
) -> httpx.Response:
    merged_headers = {"User-Agent": _USER_AGENT, **(headers or {})}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, headers=merged_headers) as client:
                resp = client.get(url, params=params)
                if resp.status_code in {429, 503} and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return resp
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _strip_html(raw: str) -> str:
    try:
        from markdownify import markdownify

        return markdownify(raw)
    except Exception:
        cleaned = _TAG_RE.sub(" ", raw)
        cleaned = _HTML_RE.sub(" ", cleaned)
        cleaned = html.unescape(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()


def _extract_pdf_text(data: bytes) -> str:
    """Extract page-indexed text from PDF bytes using pypdf or pymupdf if available."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(f"[Page {idx}]\n{txt}")
        return "\n\n".join(pages)
    except Exception:
        pass
    try:
        import fitz  # type: ignore[import-not-found]

        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for idx, page in enumerate(doc, start=1):
            txt = page.get_text("text").strip()
            if txt:
                pages.append(f"[Page {idx}]\n{txt}")
        return "\n\n".join(pages)
    except Exception:
        return ""


def _reconstruct_openalex_abstract(inv_index: dict[str, list[int]] | None) -> str:
    if not isinstance(inv_index, dict) or not inv_index:
        return ""
    max_pos = 0
    for positions in inv_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    words = [""] * (max_pos + 1)
    for word, positions in inv_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join(w for w in words if w).strip()


def _extract_arxiv_id(raw: str) -> str | None:
    text = raw.strip()
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    m = _ARXIV_ID_RE.search(text)
    if m:
        return m.group(1)
    return None


def _select_focused_excerpt(full_text: str, focus_query: str, cap: int) -> tuple[str, str]:
    """Return (excerpt, locator_summary) prioritizing paragraphs matching *focus_query*."""
    cleaned = full_text.strip()
    if not cleaned:
        return "", "empty"
    if not focus_query.strip():
        snippet = cleaned[:cap]
        if len(cleaned) > cap:
            snippet += f"\n\n... [truncated at {cap} chars; total {len(cleaned)} chars]"
        return snippet, "beginning"

    terms = [t for t in re.split(r"\W+", focus_query.lower()) if len(t) >= 2]
    chunks = [c.strip() for c in re.split(r"\n{2,}", cleaned) if c.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        low = chunk.lower()
        score = sum(low.count(t) for t in terms) if terms else 0
        if score > 0:
            scored.append((score, idx, chunk))
    scored.sort(key=lambda x: (-x[0], x[1]))

    lead = cleaned[: min(3000, cap // 3)]
    if not scored:
        snippet = cleaned[:cap]
        if len(cleaned) > cap:
            snippet += f"\n\n... [no exact keyword matches for '{focus_query}'; showing opening {cap} chars]"
        return snippet, "beginning (no query hit)"

    parts = [f"## Opening Context\n{lead}", "## Focused Excerpts"]
    used_len = len(parts[0]) + 40
    hit_locs: list[str] = []
    for _, idx, chunk in scored[:12]:
        page_match = re.search(r"\[Page \d+\]", chunk)
        loc = page_match.group(0) if page_match else f"chunk#{idx + 1}"
        block = f"### Match ({loc})\n{chunk[:2000]}"
        if used_len + len(block) > cap:
            break
        parts.append(block)
        used_len += len(block)
        hit_locs.append(loc)
    return "\n\n".join(parts), ", ".join(hit_locs[:5]) or "matched chunks"


def fetch_paper_content(
    identifier_or_url: str,
    *,
    workspace: Path | None = None,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """Retrieve full text or abstract from local file, arXiv ID, or HTTP/HTTPS PDF/HTML URL."""
    raw = identifier_or_url.strip()
    is_remote = raw.startswith(("http://", "https://"))
    arxiv_id = _extract_arxiv_id(raw)

    # 1. Local file lookup when workspace is provided and input is not a remote URL/bare arXiv ID
    if workspace is not None and not is_remote:
        ws = Path(workspace).resolve()
        candidates = [ws / raw.lstrip("/"), ws / "literature_db" / raw]
        for cand in candidates:
            try:
                resolved = cand.resolve()
                resolved.relative_to(ws)
            except Exception:
                continue
            if resolved.is_file():
                suffix = resolved.suffix.lower()
                if suffix == ".pdf":
                    text = _extract_pdf_text(resolved.read_bytes())
                elif suffix in {".html", ".htm"}:
                    text = _strip_html(resolved.read_text(encoding="utf-8", errors="replace"))
                else:
                    text = resolved.read_text(encoding="utf-8", errors="replace")
                depth = "fulltext_read" if len(text.strip()) >= 1500 else "abstract_only"
                return {
                    "id": resolved.stem,
                    "title": resolved.stem,
                    "url": f"local://{resolved.relative_to(ws).as_posix()}",
                    "text": text,
                    "reading_depth": depth,
                    "locator": f"local:{resolved.relative_to(ws).as_posix()}",
                }

    # 2. arXiv ID resolution (API metadata + HTML / ar5iv / PDF)
    if arxiv_id and (not is_remote or "arxiv.org" in raw.lower() or "ar5iv" in raw.lower()):
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        title = f"arXiv:{arxiv_id}"
        abstract = ""
        try:
            meta_resp = _http_get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id},
                timeout=timeout,
            )
            if meta_resp.status_code == 200:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(meta_resp.text)
                entry = root.find("atom:entry", ns)
                if entry is not None:
                    title = " ".join((entry.findtext("atom:title", "", ns) or title).split())
                    abstract = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
        except Exception:
            pass

        for html_url in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
            try:
                hresp = _http_get(html_url, timeout=timeout)
                if hresp.status_code == 200 and "<html" in hresp.text[:2000].lower():
                    body = _strip_html(hresp.text)
                    if len(body.strip()) >= 1500:
                        return {
                            "id": f"arxiv_{arxiv_id.replace('/', '_')}",
                            "title": title,
                            "url": abs_url,
                            "text": f"# {title}\nURL: {abs_url}\n\n{body}",
                            "reading_depth": "fulltext_read",
                            "locator": f"fulltext HTML ({html_url})",
                        }
            except Exception:
                continue

        try:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            presp = _http_get(pdf_url, timeout=max(timeout, 45.0))
            if presp.status_code == 200 and presp.content.startswith(b"%PDF"):
                pdf_text = _extract_pdf_text(presp.content)
                if len(pdf_text.strip()) >= 1500:
                    return {
                        "id": f"arxiv_{arxiv_id.replace('/', '_')}",
                        "title": title,
                        "url": abs_url,
                        "text": f"# {title}\nURL: {abs_url}\n\n{pdf_text}",
                        "reading_depth": "fulltext_read",
                        "locator": f"fulltext PDF ({pdf_url})",
                    }
        except Exception:
            pass

        if abstract:
            return {
                "id": f"arxiv_{arxiv_id.replace('/', '_')}",
                "title": title,
                "url": abs_url,
                "text": f"# {title}\nURL: {abs_url}\n\n[reading_depth=abstract_only: full HTML/PDF unavailable]\n\n{abstract}",
                "reading_depth": "abstract_only",
                "locator": "arXiv abstract",
            }
        raise RuntimeError(f"Could not retrieve arXiv paper '{arxiv_id}'.")

    # 3. Direct HTTP/HTTPS URL (PDF or HTML/text)
    if is_remote:
        resp = _http_get(raw, timeout=max(timeout, 35.0))
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or raw.lower().endswith(".pdf") or resp.content[:4] == b"%PDF":
            text = _extract_pdf_text(resp.content)
            if not text.strip():
                raise RuntimeError(f"Downloaded PDF from {raw} but could not extract readable text.")
        elif "text/plain" in ctype or "json" in ctype or "markdown" in ctype:
            text = resp.text
        else:
            text = _strip_html(resp.text)
        depth = "fulltext_read" if len(text.strip()) >= 1500 else "abstract_only"
        safe_id = "url_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return {
            "id": safe_id,
            "title": raw,
            "url": raw,
            "text": text,
            "reading_depth": depth,
            "locator": f"{depth} ({raw})",
        }

    raise FileNotFoundError(f"Neither a local file nor a valid arXiv ID / URL: {identifier_or_url}")


class LiteratureSearchPlugin(BasePlugin):
    """Gives the agent TLS-safe, rate-limit-tolerant arXiv/OpenAlex/GitHub/full-text tools."""

    name = "literature"

    def enabled(self, cfg: Any) -> bool:
        return bool(cfg.plugins.literature.enabled)

    def system_prompt_section(self, state: RunState, ctx: HookContext) -> str | None:
        providers = ", ".join(ctx.config.plugins.literature.providers)
        return (
            "# Literature & Code Discovery Tools\n"
            f"Enabled search tools ({providers}): use `search_arxiv`, `search_openalex`, `search_github`, "
            "`search_semantic_scholar`, `fetch_webpage`, and `read_paper_fulltext` directly instead of writing ad-hoc "
            "HTTP scrapers. Always prefer `read_paper_fulltext` before reproducing a paper so formulas/protocols are verified, "
            "and record verified sources in `literature_notes.md` (and `record_research_note` when harness is enabled)."
        )

    def tools(self, ctx: HookContext) -> list[ToolDef]:
        lcfg = ctx.config.plugins.literature
        providers = set(lcfg.providers)
        timeout = float(lcfg.timeout)
        tools: list[ToolDef] = []

        if "arxiv" in providers:
            def search_arxiv(query: str, max_results: int = 8) -> str:
                limit = max(1, min(25, int(max_results)))
                try:
                    resp = _http_get(
                        "https://export.arxiv.org/api/query",
                        params={
                            "search_query": f"all:{query}",
                            "start": 0,
                            "max_results": limit,
                            "sortBy": "relevance",
                            "sortOrder": "descending",
                        },
                        timeout=timeout,
                    )
                    resp.raise_for_status()
                except Exception as exc:
                    return f"arXiv search error: {type(exc).__name__}: {exc}"

                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ns)
                if not entries:
                    return f"No arXiv results found for '{query}'."

                blocks: list[str] = []
                for idx, entry in enumerate(entries, start=1):
                    eid = (entry.findtext("atom:id", "", ns) or "").strip()
                    title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
                    pub = (entry.findtext("atom:published", "", ns) or "")[:10]
                    summary = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
                    authors = [
                        (a.findtext("atom:name", "", ns) or "").strip()
                        for a in entry.findall("atom:author", ns)[:5]
                    ]
                    blocks.append(
                        f"[{idx}] {title}\n"
                        f"    URL: {eid} | Published: {pub} | Authors: {', '.join(authors)}\n"
                        f"    Abstract: {summary[:700]}"
                    )
                return f"arXiv results for '{query}' ({len(blocks)}):\n\n" + "\n\n".join(blocks)

            tools.append(
                ToolDef(
                    name="search_arxiv",
                    description="Search arXiv papers by keywords/topic and return verified titles, dates, URLs, and abstracts.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query keywords."},
                            "max_results": {"type": "integer", "description": "Number of papers (1-25, default 8)."},
                        },
                        "required": ["query"],
                    },
                    handler=search_arxiv,
                )
            )

        if "openalex" in providers:
            def search_openalex(query: str, max_results: int = 8) -> str:
                limit = max(1, min(25, int(max_results)))
                try:
                    resp = _http_get(
                        "https://api.openalex.org/works",
                        params={"search": query, "per-page": limit},
                        timeout=timeout,
                    )
                    if resp.status_code == 429:
                        return "OpenAlex API rate-limited (HTTP 429). Try search_arxiv."
                    resp.raise_for_status()
                    items = resp.json().get("results") or []
                except Exception as exc:
                    return f"OpenAlex search error: {type(exc).__name__}: {exc}"

                if not items:
                    return f"No OpenAlex results found for '{query}'."

                lines: list[str] = []
                for idx, work in enumerate(items, start=1):
                    title = work.get("title") or "(untitled)"
                    year = work.get("publication_year") or "N/A"
                    cites = work.get("cited_by_count", 0)
                    oa = work.get("open_access") or {}
                    oa_url = oa.get("oa_url") or work.get("doi") or work.get("id")
                    abstract = _reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
                    lines.append(
                        f"[{idx}] {title} ({year}, cites={cites})\n"
                        f"    URL: {oa_url} | ID: {work.get('id')}\n"
                        f"    Abstract: {(abstract or '(none)')[:600]}"
                    )
                return f"OpenAlex results for '{query}' ({len(lines)}):\n\n" + "\n\n".join(lines)

            tools.append(
                ToolDef(
                    name="search_openalex",
                    description="Search OpenAlex works for scholarly papers, citation counts, open-access PDFs, and abstracts.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Academic search keywords."},
                            "max_results": {"type": "integer", "description": "Max works to return (1-25, default 8)."},
                        },
                        "required": ["query"],
                    },
                    handler=search_openalex,
                )
            )

        if "github" in providers:
            def search_github(query: str, max_results: int = 8) -> str:
                limit = max(1, min(20, int(max_results)))
                try:
                    resp = _http_get(
                        "https://api.github.com/search/repositories",
                        params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
                        headers={"Accept": "application/vnd.github+json"},
                        timeout=timeout,
                    )
                    if resp.status_code in {403, 429}:
                        return f"GitHub API rate-limited (HTTP {resp.status_code}). Try search_arxiv or fetch_webpage."
                    resp.raise_for_status()
                    items = resp.json().get("items") or []
                except Exception as exc:
                    return f"GitHub search error: {type(exc).__name__}: {exc}"

                if not items:
                    return f"No GitHub repositories found for '{query}'."
                lines = []
                for idx, repo in enumerate(items, start=1):
                    lines.append(
                        f"[{idx}] {repo.get('full_name')} (stars={repo.get('stargazers_count')}, "
                        f"lang={repo.get('language')}, updated={str(repo.get('updated_at', ''))[:10]})\n"
                        f"    URL: {repo.get('html_url')}\n"
                        f"    Desc: {repo.get('description') or '(none)'}"
                    )
                return f"GitHub repositories for '{query}':\n\n" + "\n\n".join(lines)

            tools.append(
                ToolDef(
                    name="search_github",
                    description="Search GitHub repositories for open-source baselines, benchmarks, and reference implementations.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Repository search query."},
                            "max_results": {"type": "integer", "description": "Max repos to return (1-20, default 8)."},
                        },
                        "required": ["query"],
                    },
                    handler=search_github,
                )
            )

        if "semantic_scholar" in providers:
            def search_semantic_scholar(query: str, max_results: int = 8) -> str:
                limit = max(1, min(20, int(max_results)))
                try:
                    resp = _http_get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": query,
                            "limit": limit,
                            "fields": "title,year,citationCount,url,abstract,externalIds",
                        },
                        timeout=timeout,
                    )
                    if resp.status_code == 429:
                        return "Semantic Scholar API rate-limited (HTTP 429). Use search_arxiv or search_openalex instead."
                    resp.raise_for_status()
                    items = resp.json().get("data") or []
                except Exception as exc:
                    return f"Semantic Scholar error: {type(exc).__name__}: {exc}"

                if not items:
                    return f"No Semantic Scholar papers found for '{query}'."
                lines = []
                for idx, item in enumerate(items, start=1):
                    ext = item.get("externalIds") or {}
                    arxiv_id = ext.get("ArXiv", "")
                    lines.append(
                        f"[{idx}] {item.get('title')} ({item.get('year')}, cites={item.get('citationCount')})\n"
                        f"    URL: {item.get('url')} | arXiv: {arxiv_id or 'N/A'}\n"
                        f"    Abstract: {str(item.get('abstract') or '(none)')[:500]}"
                    )
                return f"Semantic Scholar results for '{query}':\n\n" + "\n\n".join(lines)

            tools.append(
                ToolDef(
                    name="search_semantic_scholar",
                    description="Search Semantic Scholar for papers with citation counts and abstracts.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Academic search query."},
                            "max_results": {"type": "integer", "description": "Max papers (1-20, default 8)."},
                        },
                        "required": ["query"],
                    },
                    handler=search_semantic_scholar,
                )
            )

        if "webpage" in providers:
            def fetch_webpage(url: str, max_chars: int = 12_000) -> str:
                cap = max(1000, min(30_000, int(max_chars)))
                try:
                    resp = _http_get(url, timeout=timeout)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    text = resp.text if "text/plain" in ctype or "json" in ctype else _strip_html(resp.text)
                    if len(text) > cap:
                        text = text[:cap] + f"\n... [truncated at {cap} chars]"
                    return f"URL: {url}\n\n{text}"
                except Exception as exc:
                    return f"Fetch error for {url}: {type(exc).__name__}: {exc}"

            tools.append(
                ToolDef(
                    name="fetch_webpage",
                    description="Fetch a webpage, arXiv abstract page, or raw GitHub README/file as readable text.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch."},
                            "max_chars": {"type": "integer", "description": "Max characters to return (default 12000)."},
                        },
                        "required": ["url"],
                    },
                    handler=fetch_webpage,
                )
            )

        def read_paper_fulltext(
            identifier_or_url: str,
            focus_query: str = "",
            max_chars: int = 16_000,
        ) -> str:
            cap = max(2000, min(32_000, int(max_chars)))
            try:
                paper = fetch_paper_content(
                    identifier_or_url,
                    workspace=ctx.workspace,
                    timeout=timeout,
                )
            except Exception as exc:
                return f"Full-text read error for '{identifier_or_url}': {type(exc).__name__}: {exc}"

            lit_dir = ctx.workspace / "literature_db"
            lit_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", paper["id"]).strip("_") or "paper"
            saved_md = lit_dir / f"{safe_name}.md"
            saved_md.write_text(paper["text"], encoding="utf-8")
            rel_saved = f"literature_db/{safe_name}.md"

            index_file = lit_dir / "index.json"
            index: dict[str, Any] = {}
            if index_file.is_file():
                try:
                    loaded = json.loads(index_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        index = loaded
                except Exception:
                    index = {}
            index[paper["id"]] = {
                "id": paper["id"],
                "title": paper["title"],
                "url": paper["url"],
                "reading_depth": paper["reading_depth"],
                "chars": len(paper["text"]),
                "saved_path": rel_saved,
                "locator": paper["locator"],
            }
            index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

            excerpt, focus_loc = _select_focused_excerpt(paper["text"], focus_query, cap)
            locator = paper["locator"] if not focus_query.strip() else f"{paper['locator']} | {focus_loc}"

            note_id = ""
            if ctx.research_system is not None and ctx.study_id is not None:
                try:
                    url = paper["url"]
                    if url.startswith(("https://", "http://")):
                        rec = ctx.research_system.notebook.source(
                            ctx.study_id,
                            paper["title"],
                            url,
                            excerpt[:600] or paper["title"],
                            locator,
                            reading_depth=paper["reading_depth"],
                            saved_path=rel_saved,
                            content=paper["text"],
                        )
                    else:
                        rec = ctx.research_system.notebook.record(
                            ctx.study_id,
                            "source",
                            paper["title"],
                            details={
                                "url": url,
                                "excerpt": excerpt[:600] or paper["title"],
                                "locator": locator,
                                "reading_depth": paper["reading_depth"],
                                "saved_path": rel_saved,
                                "content_hash": ctx.research_system.artifacts.put(paper["text"].encode("utf-8")),
                                "plan_version": ctx.research_system.store.get(ctx.study_id).plan_version,
                            },
                        )
                    note_id = rec.id
                except Exception:
                    pass

            header = (
                f"Title: {paper['title']}\n"
                f"URL: {paper['url']}\n"
                f"reading_depth: {paper['reading_depth']}\n"
                f"saved_path: {rel_saved} ({len(paper['text'])} chars)\n"
                + (f"recorded_source_id: {note_id}\n" if note_id else "")
            )
            return f"{header}\n{excerpt}"

        tools.append(
            ToolDef(
                name="read_paper_fulltext",
                description=(
                    "Fetch, extract, and locally cache the full text of a paper (arXiv ID, PDF/HTML URL, or workspace file) "
                    "into literature_db/ with explicit reading_depth ('fulltext_read' or 'abstract_only') and optional focus_query."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "identifier_or_url": {
                            "type": "string",
                            "description": "arXiv ID (e.g. 2401.12345), PDF/HTML URL, or local workspace/literature_db path.",
                        },
                        "focus_query": {
                            "type": "string",
                            "description": "Optional query to extract matching sections/formulas/methods.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Max characters returned (2000-32000, default 16000).",
                        },
                    },
                    "required": ["identifier_or_url"],
                },
                handler=read_paper_fulltext,
            )
        )

        for item in tools:
            if item.name.startswith("search_"):
                original = item.handler
                def tracked_search(_handler=original, _name=item.name, **arguments):
                    result = _handler(**arguments)
                    if ctx.research_system is not None and ctx.study_id:
                        ctx.research_system.notebook.record(ctx.study_id, "search", str(arguments),
                            details={"tool": _name, "result": result})
                    return result
                item.handler = tracked_search
        return tools
