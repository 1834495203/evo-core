"""Lightweight HTTP + HTML live observability, companion Q&A, and steering dashboard."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..observability.events import ResearchEvents
from ..interaction.commands import UserCommandInbox
from ..interaction.qa import CompanionQAAgent
from ..interaction.queries import ResearchQueries

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>EvoCore Research Dashboard</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --err: #ef4444;
  }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  h1, h2 { margin: 0 0 10px; font-weight: 600; }
  h1 { font-size: 20px; } h2 { font-size: 15px; color: var(--accent); }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
  .badge { padding: 3px 9px; border-radius: 999px; font-size: 12px; background: #0f172a; border: 1px solid var(--border); }
  .ok { color: var(--ok); border-color: var(--ok); }
  .warn { color: var(--warn); border-color: var(--warn); }
  .err { color: var(--err); border-color: var(--err); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }
  pre, .log { background: #0b1120; padding: 10px; border-radius: 8px; font-size: 12px; overflow: auto; max-height: 260px; white-space: pre-wrap; }
  input, select, textarea, button { font: inherit; background: #0f172a; color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px; }
  button { background: #0284c7; border-color: #0284c7; cursor: pointer; font-weight: 600; }
  button:hover { opacity: 0.9; }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .muted { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 id="title">EvoCore Autonomous Research Dashboard</h1>
    <div class="badges" id="badges"></div>
    <div id="study-meta" class="muted"></div>
    <p id="next-action"></p>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Companion Q&amp;A (Read-Only)</h2>
      <div class="muted">Ask progress, active hypothesis, experiment outcomes, or citations without blocking research.</div>
      <div class="row">
        <input id="qa-input" style="flex:1" placeholder="Ask about current progress, metrics, or failures..." />
        <button onclick="askQA()">Ask</button>
      </div>
      <pre id="qa-answer" style="margin-top:10px">Ready.</pre>
    </div>

    <div class="card">
      <h2>User Steering &amp; Control</h2>
      <div class="muted">Send non-blocking suggestions, direction pivots, pause, or resume commands.</div>
      <div class="row">
        <select id="cmd-kind">
          <option value="suggestion">suggestion</option>
          <option value="change_direction">change_direction</option>
          <option value="pause">pause</option>
          <option value="resume">resume</option>
        </select>
        <input id="cmd-text" style="flex:1" placeholder="Advice or new direction text..." />
        <button onclick="sendCommand()">Send</button>
      </div>
      <pre id="cmd-status" style="margin-top:10px">No command sent yet.</pre>
    </div>
  </div>

  <div class="card">
    <h2>Experiments &amp; Verification</h2>
    <table>
      <thead><tr><th>ID</th><th>Purpose</th><th>Status</th><th>Validity</th><th>Metrics / Error</th></tr></thead>
      <tbody id="exp-body"></tbody>
    </table>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Research Ledger Notes</h2>
      <p><a href="/api/records?limit=100" target="_blank">Browse archived research records</a></p>
      <div id="notes-list" class="log"></div>
    </div>
    <div class="card">
      <h2>Live Event Trace</h2>
      <div id="events-list" class="log"></div>
    </div>
  </div>
</div>
<script>
let lastCursor = 0;
async function refresh() {
  const res = await fetch('/api/progress');
  const data = await res.json();
  const s = data.study;
  const rt = data.runtime || {};
  document.getElementById('title').textContent = 'Study ' + s.id + ' — ' + s.direction;
  const badges = [
    ['status: ' + s.status, s.status === 'completed' ? 'ok' : (s.status === 'paused' ? 'warn' : '')],
    ['phase: ' + s.phase, ''],
    ['plan v' + s.plan_version, ''],
    ['acceptance: ' + s.acceptance, s.acceptance === 'accepted' ? 'ok' : (s.acceptance === 'rejected' ? 'err' : 'warn')],
    ['outcome: ' + s.scientific_outcome, ''],
  ];
  if (rt.model_calls !== undefined) badges.push(['calls: ' + rt.model_calls, '']);
  if (rt.input_tokens !== undefined) badges.push(['tokens in/out: ' + rt.input_tokens + '/' + rt.output_tokens, '']);
  document.getElementById('badges').innerHTML = badges.map(b => '<span class="badge ' + b[1] + '">' + b[0] + '</span>').join('');
  document.getElementById('study-meta').textContent = 'as_of=' + data.as_of + ' | cursor=' + data.event_cursor + ' | budget=' + data.experiments.length + '/' + s.max_experiments;
  document.getElementById('next-action').textContent = 'Next: ' + s.next_action;

  const tbody = document.getElementById('exp-body');
  tbody.innerHTML = data.experiments.map(e => {
    const valCls = e.validity === 'verified' ? 'ok' : (e.validity === 'invalid' ? 'err' : 'warn');
    const detail = (e.result && e.result.error) ? e.result.error : JSON.stringify((e.result && e.result.metrics) || {});
    return '<tr><td><code>' + e.id + '</code></td><td>' + e.purpose + '</td><td>' + e.status + '</td><td><span class="badge ' + valCls + '">' + e.validity + '</span></td><td><code>' + detail + '</code></td></tr>';
  }).join('');

  document.getElementById('notes-list').textContent = (data.recent_notes || []).map(n =>
    '[' + n.kind + '] ' + n.id + ' refs=' + JSON.stringify(n.refs) + '\\n  ' + n.text
  ).join('\\n\\n') || '(no notes yet)';

  const evRes = await fetch('/api/events?after=' + Math.max(0, lastCursor - 20) + '&limit=50');
  const evData = await evRes.json();
  if (evData.events && evData.events.length) {
    lastCursor = evData.next_cursor;
    document.getElementById('events-list').textContent = evData.events.map(ev =>
      '#' + ev.sequence + ' ' + ev.kind + ' (' + ev.actor + ') @ ' + ev.created_at
    ).join('\\n');
  }
}
async function askQA() {
  const q = document.getElementById('qa-input').value.trim();
  if (!q) return;
  document.getElementById('qa-answer').textContent = 'Thinking...';
  const res = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
  const data = await res.json();
  document.getElementById('qa-answer').textContent = (data.answer || data.error || JSON.stringify(data)) + '\\n\\n[mode=' + data.mode + ', as_of=' + data.as_of + ']';
}
async function sendCommand() {
  const kind = document.getElementById('cmd-kind').value;
  const text = document.getElementById('cmd-text').value;
  const res = await fetch('/api/command', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({kind, text})});
  const data = await res.json();
  document.getElementById('cmd-status').textContent = JSON.stringify(data, null, 2);
  refresh();
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class ResearchDashboardServer:
    """Thread-safe local HTTP + HTML server for real-time progress, companion Q&A, and user steering."""

    def __init__(
        self,
        *,
        study_id: str,
        queries: ResearchQueries,
        inbox: UserCommandInbox,
        events: ResearchEvents,
        qa_agent: CompanionQAAgent | None = None,
        run_dir: Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.study_id = study_id
        self.queries = queries
        self.inbox = inbox
        self.events = events
        self.qa_agent = qa_agent or CompanionQAAgent(queries, events)
        self.run_dir = Path(run_dir).resolve() if run_dir else None
        self.host = host
        self.port = int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _json_response(code: int, payload: Any) -> tuple[int, str, bytes]:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        return code, "application/json; charset=utf-8", body

    def dispatch_request(
        self,
        method: str,
        path: str,
        body: bytes | str | dict[str, Any] | None = None,
    ) -> tuple[int, str, bytes]:
        """Dispatch an HTTP request synchronously; shared by the socket handler and tests."""
        parsed = urlparse(path)
        verb = method.upper()

        if verb == "GET":
            if parsed.path == "/":
                return 200, "text/html; charset=utf-8", _DASHBOARD_HTML.encode("utf-8")
            if parsed.path == "/api/progress":
                try:
                    prog = self.queries.progress(self.study_id)
                    if self.run_dir is not None:
                        snap_file = self.run_dir / "live_snapshot.json"
                        if snap_file.is_file():
                            try:
                                live = json.loads(snap_file.read_text(encoding="utf-8"))
                                if isinstance(live, dict) and "runtime" in live:
                                    prog["runtime"] = live["runtime"]
                            except Exception:
                                pass
                    return self._json_response(200, prog)
                except Exception as exc:
                    return self._json_response(500, {"error": str(exc)})
            if parsed.path == "/api/records":
                try:
                    qs = parse_qs(parsed.query)
                    return self._json_response(200, self.queries.records(self.study_id,
                        kind=(qs.get("kind") or [""])[0], offset=int((qs.get("offset") or ["0"])[0]),
                        limit=int((qs.get("limit") or ["20"])[0])))
                except (ValueError, TypeError) as exc:
                    return self._json_response(400, {"error": str(exc)})
            if parsed.path == "/api/events":
                try:
                    qs = parse_qs(parsed.query)
                    after = int((qs.get("after") or ["0"])[0])
                    limit = int((qs.get("limit") or ["100"])[0])
                    return self._json_response(200, self.events.read(self.study_id, after=after, limit=limit))
                except Exception as exc:
                    return self._json_response(400, {"error": str(exc)})
            return self._json_response(404, {"error": "Not found"})

        if verb == "POST":
            try:
                if body is None:
                    data: Any = {}
                elif isinstance(body, dict):
                    data = body
                elif isinstance(body, bytes):
                    data = json.loads((body or b"{}").decode("utf-8"))
                else:
                    data = json.loads(body or "{}")
                if not isinstance(data, dict):
                    raise ValueError("JSON body must be an object")
            except Exception as exc:
                return self._json_response(400, {"error": f"Invalid JSON body: {exc}"})

            if parsed.path == "/api/ask":
                question = str(data.get("question") or "")
                use_llm = bool(data.get("use_llm", True))
                try:
                    ans = self.qa_agent.answer(self.study_id, question, use_llm=use_llm)
                    return self._json_response(200, ans)
                except Exception as exc:
                    return self._json_response(400, {"error": str(exc)})

            if parsed.path == "/api/command":
                kind = str(data.get("kind") or "")
                text = str(data.get("text") or "")
                expected_pv = data.get("expected_plan_version")
                try:
                    cmd_id = self.inbox.send(
                        self.study_id,
                        kind,
                        text,
                        expected_plan_version=int(expected_pv) if expected_pv is not None else None,
                    )
                    return self._json_response(200, {"command_id": cmd_id, "status": "pending", "kind": kind})
                except Exception as exc:
                    return self._json_response(400, {"error": str(exc)})

            return self._json_response(404, {"error": "Not found"})

        return self._json_response(405, {"error": f"Method not allowed: {verb}"})

    def _build_httpd(self) -> ThreadingHTTPServer:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _write_reply(self, code: int, content_type: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                code, ctype, body = parent.dispatch_request("GET", self.path)
                self._write_reply(code, ctype, body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b"{}"
                code, ctype, body = parent.dispatch_request("POST", self.path, raw)
                self._write_reply(code, ctype, body)

        return _ReusableThreadingHTTPServer((self.host, self.port), Handler)

    def start_background(self) -> tuple[str, int]:
        if self._httpd is not None:
            return str(self._httpd.server_address[0]), int(self._httpd.server_address[1])
        self._httpd = self._build_httpd()
        bound_host, bound_port = self._httpd.server_address[:2]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="evo-dashboard",
            daemon=True,
        )
        self._thread.start()
        return str(bound_host), int(bound_port)

    def serve_forever(self) -> None:
        if self._httpd is None:
            self._httpd = self._build_httpd()
        try:
            self._httpd.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
