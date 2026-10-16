"""Scanning image directories and persisting their embeddings."""

import hashlib
import itertools
import sys
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
from PIL import Image

from gmbd.db import ImageDB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_BATCH_SIZE = 32


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
    db = ImageDB()

    for batch in itertools.batched(paths, IMAGE_BATCH_SIZE):
        ids = [file_id(path) for path in batch]
        stored_rows = db.fetch_ids(ids)
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
            db.insert(batch_paths, batch_ids, vectors)
            inserted += len(pending)

        if copies:
            batch_paths = [path for path, _ in copies]
            batch_ids = [record_id for _, record_id in copies]
            vectors = [stored_vectors[record_id] for record_id in batch_ids]
            db.insert(batch_paths, batch_ids, vectors)
            copied += len(copies)

        skipped += len(batch) - len(pending) - len(copies)

        print(
            f"Inserted {inserted}, copied {copied}, skipped {skipped} from {count} images ({(inserted + copied + skipped) * 100 // count}%)",
            end="\r",
            file=sys.stderr,
            flush=True,
        )

    if inserted or copied:
        db.optimize()

    print(file=sys.stderr)
