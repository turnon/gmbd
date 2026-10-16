"""Serve the result gallery over HTTP with the built-in http server."""

import mimetypes
import queue
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from gmbd.db import ImageDB
from gmbd.gallery import build_gallery

PAGE_SIZE = 20
GALLERY_HISTORY = 4


def run_server(embed, query: str | None = None, scope: str | None = None) -> None:
    """Serve the result gallery over HTTP on 0.0.0.0:8500 via http.server."""
    state: dict[str, object] = {
        "dirs": {},
        "query": query or "",
        "scope": scope or "",
    }
    state_lock = threading.Lock()
    db = ImageDB()

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
            parsed = urlparse(self.path)
            if parsed.path not in ("/", "/index.html"):
                self.serve_result_file(parsed.path)
                return

            params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
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

        def serve_result_file(self, url_path: str) -> None:
            parts = unquote(url_path.lstrip("/")).split("/")
            if len(parts) != 2:
                self.send_error(404)
                return
            gallery_id, name = parts
            with state_lock:
                directory = state["dirs"].get(gallery_id)
            if directory is None or not name or name.startswith("."):
                self.send_error(404)
                return

            path = directory / name
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
