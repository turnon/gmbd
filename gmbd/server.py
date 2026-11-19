"""Serve the result gallery over HTTP with the built-in http server."""

import json
import mimetypes
import queue
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from gmbd.gallery import build_gallery

PAGE_SIZE = 20
GALLERY_HISTORY = 4

# main.py and the uv project root sit one level above this package.
PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = Path("/tmp")


def run_server(
    embed, db, query: str | None = None, scope: str | None = None
) -> None:
    """Serve the result gallery over HTTP on 0.0.0.0:8500 via http.server."""
    state: dict[str, object] = {
        "dirs": {},
        "query": query or "",
        "scope": scope or "",
    }
    state_lock = threading.Lock()

    # MLX state (streams) is thread-local and the model was loaded on the main
    # thread, so all embedding work has to run there. HTTP handlers run in
    # worker threads and hand work back to the main thread through this queue.
    jobs: queue.Queue = queue.Queue()

    def run_on_main(func):
        box: dict[str, object] = {"done": threading.Event()}
        jobs.put((func, box))
        box["done"].wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    class GalleryHandler(BaseHTTPRequestHandler):
        # Drop idle or stalled connections after 10s so threads do not pile up
        # on abandoned keep-alive connections. Local requests complete well
        # within this.
        timeout = 10
        # Keep-alive lets the browser reuse one connection for the gallery and
        # its thumbnails. Each connection runs in its own thread (see
        # ThreadingHTTPServer below), so an idle connection cannot block others.
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_redirect("/search")
                return
            if path == "/search":
                self.serve_search(self.path)
                return
            if path.startswith("/jobs/"):
                self.serve_job_log(unquote(path[len("/jobs/") :]))
                return
            if path.startswith("/result/"):
                self.serve_result_file(path)
                return
            self.send_error(404)

        def serve_search(self, url: str) -> None:
            # Keep blank values so an emptied query/scope field clears the
            # previous state instead of falling back to it.
            params = {
                key: values[0]
                for key, values in parse_qs(
                    urlparse(url).query, keep_blank_values=True
                ).items()
            }
            requested = params.get("query", state["query"])
            requested_scope = params.get("scope", state["scope"])
            try:
                page = max(1, int(params.get("page", "1")))
            except ValueError:
                page = 1

            state["query"] = requested
            state["scope"] = requested_scope
            if requested:

                def search():
                    embedding = embed(
                        text=[f"task: search result | query: {requested}"]
                    )
                    return db.search(
                        embedding,
                        scope=requested_scope or None,
                        page=page,
                        per_page=PAGE_SIZE,
                    )

                ranked, total = run_on_main(search)
            else:
                ranked, total = [], 0

            if self.wants_json():
                payload = {
                    "list": [
                        {"path": str(path), "score": score} for path, score in ranked
                    ],
                    "total": total,
                }
                self.send_body(
                    json.dumps(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return

            tmp_dir, index = build_gallery(
                ranked,
                query=requested,
                scope=requested_scope,
                page=page,
                per_page=PAGE_SIZE,
                total=total,
                prefix_links=True,
            )
            # Keep a few recent galleries so thumbnails from a page the client
            # is still loading stay reachable, then drop the oldest on disk.
            with state_lock:
                dirs = state["dirs"]
                dirs[tmp_dir.name] = tmp_dir
                while len(dirs) > GALLERY_HISTORY:
                    old_id = next(iter(dirs))
                    shutil.rmtree(dirs.pop(old_id), ignore_errors=True)
            body = index.read_text(encoding="utf-8")

            self.send_body(body.encode("utf-8"), "text/html; charset=utf-8")

        def wants_json(self) -> bool:
            """True when the caller asked for the results as JSON, not HTML."""
            for header in ("Accept", "Content-Type"):
                if "application/json" in (self.headers.get(header) or ""):
                    return True
            return False

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/dir":
                self.send_error(404)
                return
            params = parse_qs(parsed.query, keep_blank_values=True)
            # Discard any request body so a keep-alive connection is not left
            # with unread bytes that would corrupt the next request.
            body_length = int(self.headers.get("Content-Length") or 0)
            if body_length:
                self.rfile.read(body_length)
            directory = params.get("path", [""])[0]
            if not directory or not Path(directory).is_dir():
                self.send_error(400, f"Not a directory: {directory}")
                return

            job_id = uuid.uuid7().hex
            log_path = LOG_DIR / f"gmbd_{job_id}.log"
            # Detach the indexer from this server: its own session, no stdin,
            # and progress (tqdm writes carriage-return updates) only to the
            # log we poll through GET /?job_id=...
            with open(log_path, "wb") as log_file:
                subprocess.Popen(
                    ["uv", "run", "main.py", "--db", db.kind, "--dir", directory],
                    cwd=PROJECT_DIR,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log_file,
                    start_new_session=True,
                    close_fds=True,
                )
            self.send_body(
                (job_id + "\n").encode("utf-8"), "text/plain; charset=utf-8"
            )

        def serve_job_log(self, job_id: str) -> None:
            # Only hex uuids can name a log file; anything else (including
            # traversal attempts) is treated as an unknown job.
            if not job_id.isalnum():
                self.send_error(404, "Unknown job id")
                return
            log_path = LOG_DIR / f"gmbd_{job_id}.log"
            try:
                data = log_path.read_bytes()
            except FileNotFoundError:
                self.send_error(404, "Unknown job id")
                return
            # tqdm redraws its line with a leading carriage return, so the last
            # segment is the most recent progress line.
            line = data.split(b"\r")[-1]
            self.send_body(line, "text/plain; charset=utf-8")

        def send_redirect(self, location: str) -> None:
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def serve_result_file(self, url_path: str) -> None:
            """Serve /result/<gallery id>/<pic id> from the stored gallery dir."""
            parts = unquote(url_path.lstrip("/")).split("/")
            if len(parts) != 3 or parts[0] != "result":
                self.send_error(404)
                return
            _, gallery_id, pic_id = parts
            with state_lock:
                directory = state["dirs"].get(gallery_id)
            if directory is None or not pic_id or pic_id.startswith("."):
                self.send_error(404)
                return

            path = directory / pic_id
            if not path.is_file():
                self.send_error(404)
                return

            content_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            self.send_body(path.read_bytes(), content_type)

        def send_body(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Results depend on the current query/page selection, so avoid
            # serving a stale gallery or thumbnail from the browser cache.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

        def log_error(self, format: str, *args) -> None:
            # An idle keep-alive connection that hits the socket timeout is
            # expected (the response was already sent); don't log it as an error.
            if args and isinstance(args[0], TimeoutError):
                return
            super().log_error(format, *args)

    # A threaded server is required: browsers hold keep-alive connections open
    # while fetching thumbnails, and a single-threaded serve_forever() would
    # block on the first idle connection and stall every other request. It runs
    # in a background thread so the main thread stays free to service the
    # embedding queue above.
    server = ThreadingHTTPServer(("0.0.0.0", 8500), GalleryHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("Serving gallery on http://0.0.0.0:8500", file=sys.stderr)
    try:
        while True:
            # Poll with a timeout so the main thread periodically returns to
            # the interpreter and stays responsive to Ctrl+C.
            try:
                func, box = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                box["result"] = func()
            except BaseException as error:  # noqa: BLE001 - forward to requester
                box["error"] = error
            box["done"].set()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
