"""An Ollama-compatible embedding API: POST /api/embed.

The model is loaded on the main thread and MLX streams are thread-local, so HTTP
worker threads hand every embedding request back to the main thread through a
queue (the same pattern gmbd.server uses for its gallery searches).
"""

import base64
import io
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import mlx.core as mx
from PIL import Image, ImageOps

PORT = 11434
MODEL = "mlx-community/embeddinggemma-2-bf16"


def _decode_image(value: str) -> Image.Image:
    """Decode a base64 string (optionally a data URI) into an RGB image."""
    data = value.strip()
    if data.startswith("data:"):
        _, separator, data = data.partition(",")
        if not separator:
            raise ValueError("Invalid image data URI")
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as error:
        raise ValueError(f"Invalid base64 image: {error}") from error
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as error:
        raise ValueError(f"Invalid image data: {error}") from error


def parse_input(value) -> list[tuple[str, str]]:
    """Normalize Ollama's input field into (kind, value) items."""
    if isinstance(value, str):
        return [("text", value)]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("`input` must be a non-empty string, image, or array.")
    items = []
    for item in value:
        if isinstance(item, str):
            items.append(("text", item))
        elif isinstance(item, dict) and isinstance(item.get("image"), str):
            items.append(("image", item["image"]))
        else:
            raise ValueError(
                '`input` items must be strings or {"image": <base64>} objects.'
            )
    return items


def embed_items(embed, items: list[tuple[str, str]]) -> list[list[float]]:
    """Embed parsed input items, preserving their order."""
    kinds = {kind for kind, _ in items}
    if kinds == {"text"}:
        array = embed(text=[value for _, value in items])
        mx.eval(array)
        return array.astype(mx.float32).tolist()
    if kinds == {"image"}:
        images = [[_decode_image(value)] for _, value in items]
        array = embed(images=images)
        mx.eval(array)
        return array.astype(mx.float32).tolist()

    vectors = []
    for kind, value in items:
        if kind == "text":
            array = embed(text=[value])
        else:
            array = embed(images=[[_decode_image(value)]])
        mx.eval(array)
        vectors.append(array.astype(mx.float32)[0].tolist())
    return vectors


def run_embedding_server(embed, port: int = PORT) -> None:
    """Serve POST /api/embed with an Ollama-shaped request and response."""
    jobs: queue.Queue = queue.Queue()

    def run_on_main(func):
        box: dict[str, object] = {"done": threading.Event()}
        jobs.put((func, box))
        box["done"].wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    class EmbeddingHandler(BaseHTTPRequestHandler):
        # Embedding an image can take a while; allow long-lived clients to wait.
        timeout = 60
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/embed":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self.send_json({"error": "Invalid Content-Length"}, 400)
                return
            body = self.rfile.read(length) if length else b""

            try:
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                items = parse_input(payload.get("input"))
                started = time.perf_counter()
                vectors = run_on_main(lambda: embed_items(embed, items))
                elapsed = time.perf_counter() - started
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
                return
            except Exception as error:  # noqa: BLE001 - report to the caller
                print(f"Embedding request failed: {error}", file=sys.stderr)
                self.send_json({"error": str(error)}, 500)
                return

            self.send_json(
                {
                    "model": MODEL,
                    "embeddings": vectors,
                    "total_duration": int(elapsed * 1_000_000_000),
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                }
            )

        def do_GET(self) -> None:
            self.send_error(404)

        def send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    server = ThreadingHTTPServer(("0.0.0.0", port), EmbeddingHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"Serving Ollama embedding API on http://0.0.0.0:{port}/api/embed",
        file=sys.stderr,
    )
    try:
        while True:
            # Poll with a timeout so the main thread periodically returns to the
            # interpreter and stays responsive to Ctrl+C.
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
