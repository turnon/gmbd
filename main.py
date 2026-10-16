import argparse
import hashlib
import html
import itertools
import mimetypes
import os
import queue
import shutil
import sys
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import lancedb
import mlx.core as mx
import pyarrow as pa
from lancedb.index import BTree
from mlx_vlm import load
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_BATCH_SIZE = 32
PAGE_SIZE = 20
GALLERY_HISTORY = 4
DB_DIR = Path.home() / ".gmbd" / "db"
TABLE_NAME = "images"


def image_paths(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_images(paths: list[Path]) -> Iterator[tuple[Path, Image.Image]]:
    for path in paths:
        with Image.open(path) as image:
            yield path, image.convert("RGB")


def file_id(path: Path) -> str:
    return hashlib.md5(path.resolve().read_bytes(), usedforsecurity=False).hexdigest()


def fetch_db_ids(ids: list[str]) -> list[dict]:
    """Return the stored records matching the given ids."""
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        return []

    table = db.open_table(TABLE_NAME)
    stored = []
    for chunk in itertools.batched(sorted(ids), 500):
        quoted = ", ".join(
            "'" + record_id.replace("'", "''") + "'" for record_id in chunk
        )
        rows = (
            table.search()
            .select(["id", "dir", "name", "vector"])
            .where(f"id IN ({quoted})")
            .to_arrow()
            .to_pylist()
        )
        stored.extend(rows)
    return stored


def insert_db(
    paths: list[Path], ids: list[str], vectors: list[list[float]]
) -> lancedb.table.LanceTable:
    records = []
    for path, record_id, vector in zip(paths, ids, vectors):
        resolved = path.resolve()
        records.append(
            {
                "id": record_id,
                "name": path.name,
                "dir": str(resolved.parent),
                "vector": vector,
            }
        )

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("dir", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), len(vectors[0]))),
            ]
        )
        table = db.create_table(TABLE_NAME, schema=schema)
        table.create_index("id", config=BTree())
    else:
        table = db.open_table(TABLE_NAME)

    table.add(records)
    return table


def page_count(total: int, per_page: int) -> int:
    return max(1, (total + per_page - 1) // per_page)


def query_db(
    embedding: mx.array,
    scope: str | None = None,
    page: int = 1,
    per_page: int | None = None,
) -> tuple[list[tuple[Path, float]], int]:
    """Rank images by similarity against embeddings persisted in LanceDB.

    When per_page is given, only the requested page of the ranking is fetched
    from the database (offset/limit), and page is clamped to the page count.
    Returns (ranked results, total matching rows).
    """
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        return [], 0

    table = db.open_table(TABLE_NAME)

    filter = None
    if scope is not None:
        quoted_scope = scope.replace("'", "''")
        filter = f"regexp_like(dir, '{quoted_scope}')"

    total = table.count_rows(filter=filter)
    if per_page:
        page = min(max(1, page), page_count(total, per_page))
        limit = per_page
    else:
        limit = total

    query = table.search(
        embedding.astype(mx.float32)[0].tolist(),
        vector_column_name="vector",
    ).metric("dot")
    if filter is not None:
        query = query.where(filter)
    if per_page:
        query = query.offset((page - 1) * per_page)
    query = query.limit(limit)

    results = query.to_arrow().to_pylist()

    # LanceDB reports _distance = 1 - dot for the "dot" metric.
    ranked = [
        (Path(row["dir"]) / row["name"], 1.0 - row["_distance"]) for row in results
    ]
    return ranked, total


def pagination_nav(page: int, pages: int) -> str:
    """Render prev/next and page-number submit buttons for the search form."""
    if pages <= 1:
        return ""

    def jump(target: int, label: str) -> str:
        return f'<button type="submit" name="page" value="{target}">{label}</button>'

    parts = []
    if page > 1:
        parts.append(jump(page - 1, "Prev"))
    else:
        parts.append('<button type="submit" disabled>Prev</button>')

    shown = [
        number
        for number in range(1, pages + 1)
        if number in (1, pages) or abs(number - page) <= 2
    ]
    previous = 0
    for number in shown:
        if previous and number - previous > 1:
            parts.append('<span class="gap">…</span>')
        if number == page:
            parts.append(f'<span class="current">{number}</span>')
        else:
            parts.append(jump(number, str(number)))
        previous = number

    if page < pages:
        parts.append(jump(page + 1, "Next"))
    else:
        parts.append('<button type="submit" disabled>Next</button>')

    return f'<nav class="pages">{"".join(parts)}</nav>'


def write_gallery_html(
    tmp_dir: Path,
    entries: list[tuple[str, int, float, Path]],
    query: str = "",
    scope: str = "",
    total: int | None = None,
    page: int = 1,
    pages: int = 1,
    per_page: int = 0,
) -> Path:
    """Write an index.html search form and thumbnail gallery for ranked results."""
    if total is None:
        total = len(entries)
    if pages > 1 and per_page:
        lo = (page - 1) * per_page + 1
        heading = f"{lo}\u2013{min(page * per_page, total)} of {total} results"
    elif pages > 1:
        heading = f"0 of {total} results"
    else:
        heading = f"{len(entries)} results"
    nav = pagination_nav(page, pages)

    cards = []
    for link_name, rank, score, path in entries:
        cards.append(
            f"""
      <figure class="card">
        <a href="{html.escape(link_name)}" target="_blank">
          <img loading="lazy" src="{html.escape(link_name)}" alt="rank {rank}">
        </a>
        <figcaption>
          <span class="rank">#{rank + 1}</span>
          <span class="score">{score:.4f}</span>
          <span class="path" title="{html.escape(str(path))}">{html.escape(str(path))}</span>
        </figcaption>
      </figure>"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gmbd results</title>
<style>
  body {{ margin: 0; padding: 1rem; background: #111; color: #eee;
         font: 14px/1.4 -apple-system, system-ui, sans-serif; }}
  .top {{ position: sticky; top: 0; z-index: 10; background: #111;
         margin: -1rem -1rem 1rem; padding: 1rem;
         border-bottom: 1px solid #2a2a2c; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0; white-space: nowrap; }}
  .search {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0; align-items: center; }}
  .status {{ display: flex; flex-wrap: nowrap; gap: 8px 12px; align-items: center;
            flex: 1 1 100%; justify-content: center; }}
  .search input {{ flex: 1 1 220px; padding: 8px 10px; border: 1px solid #3a3a3c;
                  border-radius: 8px; background: #1c1c1e; color: #eee; font: inherit; }}
  .search input:focus {{ outline: 2px solid #ffd60a; outline-offset: -1px; }}
  .search .go {{ padding: 8px 18px; border: 0; border-radius: 8px; background: #ffd60a;
                color: #111; font: inherit; font-weight: 600; cursor: pointer; }}
  .search .go:hover {{ background: #ffe14d; }}
  .grid {{ display: grid; gap: 12px;
          grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }}
  .card {{ margin: 0; background: #1c1c1e; border-radius: 10px; overflow: hidden;
          display: flex; flex-direction: column; }}
  .card img {{ width: 100%; height: 220px; object-fit: contain; display: block;
              background: #2a2a2c; }}
  figcaption {{ padding: 8px 10px; display: flex; flex-wrap: wrap; gap: 6px 10px;
               align-items: baseline; }}
  .rank {{ font-weight: 700; color: #ffd60a; }}
  .score {{ font-variant-numeric: tabular-nums; color: #7dd3fc; }}
  .path {{ flex: 1 1 100%; color: #98989d; font-size: 11px; overflow-wrap: anywhere;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .pages {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0; }}
  .pages button, .pages span {{ padding: 8px 10px; border: 0; border-radius: 8px;
                              color: #eee; background: #1c1c1e; font: inherit; }}
  .pages button {{ cursor: pointer; }}
  .pages button:hover {{ background: #2a2a2c; }}
  .pages button:disabled {{ opacity: 0.35; cursor: default; }}
  .pages button:disabled:hover {{ background: #1c1c1e; }}
  .pages .current {{ background: #ffd60a; color: #111; font-weight: 700; }}
  .pages .gap {{ background: transparent; color: #98989d; }}
</style>
</head>
<body>
  <header class="top">
    <form class="search" method="get" action="/">
      <input type="search" name="query" placeholder="query" value="{html.escape(query)}" autofocus>
      <input type="search" name="scope" placeholder="scope (directory regex)" value="{html.escape(scope)}">
      <button class="go" type="submit">Search</button>
      <div class="status">
        {nav}
        <h1>{heading}</h1>
      </div>
    </form>
  </header>
  <div class="grid">{"".join(cards)}
  </div>
</body>
</html>
"""
    index = tmp_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def tree_db() -> None:
    """Print the directories of all persisted records in tree style."""
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        print(
            f"No embeddings found in {DB_DIR}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    table = db.open_table(TABLE_NAME)
    counts = Counter()
    for batch in table.search().select(["dir"]).to_batches():
        counts.update(batch.column("dir").to_pylist())
    if not counts:
        print(f"No records found in {DB_DIR}", file=sys.stderr)
        return

    # Build a nested tree of path components below their common root,
    # keeping the number of records at each directory.
    root: dict = {}
    base = os.path.commonpath(sorted(counts))
    base_len = len(Path(base).parts)
    for dir_path, count in sorted(counts.items()):
        parts = Path(dir_path).parts[base_len:]
        children = root
        for part in parts[:-1]:
            children = children.setdefault(part, {"count": 0, "children": {}})[
                "children"
            ]
        if parts:
            leaf = children.setdefault(parts[-1], {"count": 0, "children": {}})
            leaf["count"] += count

    def total(node: dict) -> int:
        return node["count"] + sum(total(child) for child in node["children"].values())

    print(f"{base} ({sum(counts.values())})")

    def render(children: dict, prefix: str = "") -> None:
        entries = sorted(children.items())
        for index, (name, node) in enumerate(entries):
            last = index == len(entries) - 1
            print(f"{prefix}{'└── ' if last else '├── '}{name} ({total(node)})")
            render(node["children"], prefix + ("    " if last else "│   "))

    render(root)


def clean_db() -> None:
    """Remove persisted records whose image files no longer exist on disk."""
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        print(
            f"No embeddings found in {DB_DIR}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    table = db.open_table(TABLE_NAME)
    total = table.count_rows()
    missing_ids = []
    for batch in table.search().select(["id", "dir", "name"]).to_batches():
        for row in batch.to_pylist():
            if not (Path(row["dir"]) / row["name"]).is_file():
                missing_ids.append(row["id"])

    for chunk in itertools.batched(missing_ids, 500):
        ids = ", ".join(f"'{record_id}'" for record_id in chunk)
        table.delete(f"id IN ({ids})")

    table.optimize(cleanup_older_than=timedelta(milliseconds=1))
    print(f"Removed {len(missing_ids)} of {total} record(s) with missing files")


def build_gallery(
    ranked: list[tuple[Path, float]],
    query: str = "",
    scope: str = "",
    page: int = 1,
    per_page: int | None = None,
    total: int | None = None,
    prefix_links: bool = False,
) -> tuple[Path, Path]:
    """Hard-link ranked results into a fresh temp dir and write the gallery html.

    ranked is expected to already hold just one page of results when per_page is
    given (as returned by query_db); total then carries the full result count for
    the pagination controls. When prefix_links is set, thumbnail URLs are made
    relative to the gallery id (used by the server, which serves each gallery
    under its own path). Returns (tmp_dir, index.html path).
    """
    tmp_dir = Path("/tmp") / str(uuid.uuid7())
    tmp_dir.mkdir()
    link_prefix = f"{tmp_dir.name}/" if prefix_links else ""

    if total is None:
        total = len(ranked)
    if per_page:
        pages = page_count(total, per_page)
        page = min(max(1, page), pages)
        start = (page - 1) * per_page
    else:
        pages, page, per_page, start = 1, 1, 0, 0

    entries = []
    skipped = 0
    for rank, (path, score) in enumerate(ranked, start=start):
        if not path.is_file():
            skipped += 1
            continue
        filename = f"{rank}{path.suffix}"
        os.link(path.resolve(), tmp_dir / filename)
        entries.append((f"{link_prefix}{filename}", rank, score, path))

    if skipped:
        print(f"skipped {skipped} missing image(s)", file=sys.stderr)

    index = write_gallery_html(
        tmp_dir,
        entries,
        query=query,
        scope=scope,
        total=total,
        page=page,
        pages=pages,
        per_page=per_page,
    )
    return tmp_dir, index


def output_ranked(ranked: list[tuple[Path, float]]) -> None:
    _, index = build_gallery(ranked)
    print(index)


def run_server(embed, query: str | None = None, scope: str | None = None) -> None:
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
                    return query_db(
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


def index_directory(directory: Path, embed) -> None:
    """Embed and persist every image under directory that is not yet stored."""
    paths = image_paths(directory)
    if not paths:
        print(f"No images found in {directory}", file=sys.stderr)
        return

    count = len(paths)
    inserted = 0
    copied = 0
    skipped = 0
    table = None

    for batch in itertools.batched(paths, IMAGE_BATCH_SIZE):
        ids = [file_id(path) for path in batch]
        stored_rows = fetch_db_ids(ids)
        stored_paths = {(row["id"], row["dir"], row["name"]) for row in stored_rows}
        stored_vectors = {row["id"]: row["vector"] for row in stored_rows}

        pending = [
            (path, record_id)
            for path, record_id in zip(batch, ids)
            if record_id not in stored_vectors
        ]
        copies = [
            (path, record_id)
            for path, record_id in zip(batch, ids)
            if record_id in stored_vectors
            and (record_id, str(path.resolve().parent), path.name) not in stored_paths
        ]

        if pending:
            batch_paths = [path for path, _ in pending]
            batch_ids = [record_id for _, record_id in pending]
            images = [image for _, image in load_images(batch_paths)]
            batch_embeddings = embed(images=[[image] for image in images])
            mx.eval(batch_embeddings)
            vectors = batch_embeddings.astype(mx.float32).tolist()
            table = insert_db(batch_paths, batch_ids, vectors)
            inserted += len(pending)

        if copies:
            batch_paths = [path for path, _ in copies]
            batch_ids = [record_id for _, record_id in copies]
            vectors = [stored_vectors[record_id] for record_id in batch_ids]
            table = insert_db(batch_paths, batch_ids, vectors)
            copied += len(copies)

        skipped += len(batch) - len(pending) - len(copies)

        print(
            f"Inserted {inserted}, copied {copied}, skipped {skipped} from {count} images ({(inserted + copied + skipped) * 100 // count}%)",
            end="\r",
            file=sys.stderr,
            flush=True,
        )

    if table is not None:
        table.optimize(cleanup_older_than=timedelta(milliseconds=1))

    print(file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("--query", default=None, help="The question (search query).")
    parser.add_argument(
        "--dir",
        default=None,
        type=Path,
        help="Directory containing images. Images whose content is already in "
        "the database are skipped. If omitted, query persisted embeddings "
        f"in {DB_DIR} instead.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Restrict query results to images whose stored directory (the "
        "images.dir column) matches this regular expression.",
        metavar="REGEX",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print the directories of all persisted records in tree style, "
        "with record counts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove persisted records whose image files no longer exist on disk.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Serve the result gallery over HTTP on 0.0.0.0:8500 with the built-in "
        "http server, so queries can be run interactively through the search form.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tree:
        tree_db()
        return

    if args.clean:
        clean_db()
        return

    model, processor = load("mlx-community/embeddinggemma-2-bf16")

    def embed(text=None, images=None) -> mx.array:
        inputs = processor(text=text, images=images, return_tensors="np")
        inputs = {key: mx.array(value) for key, value in inputs.items()}
        return model(**inputs).text_embeds

    if args.server:
        run_server(embed, query=args.query, scope=args.scope)
        return

    if args.dir is not None:
        if not args.dir.is_dir():
            print(f"Not a directory: {args.dir}", file=sys.stderr)
        else:
            index_directory(args.dir, embed)

    if args.query is not None:
        query_embedding = embed(text=[f"task: search result | query: {args.query}"])
        ranked, _ = query_db(query_embedding, scope=args.scope)
        if ranked:
            output_ranked(ranked)


if __name__ == "__main__":
    main()
