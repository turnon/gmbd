import argparse
import hashlib
import itertools
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import lancedb
import mlx.core as mx
from mlx_vlm import load
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_BATCH_SIZE = 8
DB_DIR = Path.home() / ".gmbd" / "db"
TABLE_NAME = "images"


def load_images(
    directory: Path,
) -> tuple[int, Iterator[tuple[Path, Image.Image]]]:
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    def generate() -> Iterator[tuple[Path, Image.Image]]:
        for path in paths:
            with Image.open(path) as image:
                yield path, image.convert("RGB")

    return len(paths), generate()


def upsert_db(paths: list[Path], embeddings: mx.array) -> None:
    mx.eval(embeddings)
    embeddings = embeddings.astype(mx.float32)

    records = []
    for path, embedding in zip(paths, embeddings):
        resolved = path.resolve()
        records.append(
            {
                "id": hashlib.md5(
                    resolved.read_bytes(), usedforsecurity=False
                ).hexdigest(),
                "name": path.name,
                "dir": str(resolved.parent),
                "vector": embedding.tolist(),
            }
        )

    db = lancedb.connect(DB_DIR)
    if TABLE_NAME in db.list_tables().tables:
        table = db.open_table(TABLE_NAME)
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )
    else:
        db.create_table(TABLE_NAME, data=records)


def query_db(query_embedding: mx.array) -> list[tuple[Path, float]]:
    """Rank images by similarity against embeddings persisted in LanceDB."""
    db = lancedb.connect(DB_DIR)
    if TABLE_NAME not in db.list_tables().tables:
        table = None
    else:
        table = db.open_table(TABLE_NAME)
    if table is None or table.count_rows() == 0:
        print(
            f"No embeddings found in {DB_DIR}; run with --dir DIR --persist first.",
            file=sys.stderr,
        )
        return []

    results = (
        table.search(
            query_embedding.astype(mx.float32)[0].tolist(),
            vector_column_name="vector",
        )
        .metric("dot")
        .limit(table.count_rows())
        .to_arrow()
        .to_pylist()
    )
    # LanceDB reports _distance = 1 - dot for the "dot" metric.
    return [(Path(row["dir"]) / row["name"], 1.0 - row["_distance"]) for row in results]


def output_ranked(ranked: list[tuple[Path, float]]) -> None:
    tmp_dir = Path("/tmp") / str(uuid.uuid7())
    tmp_dir.mkdir()
    print(tmp_dir)

    for rank, (path, score) in enumerate(ranked):
        if not path.is_file():
            print(f"skipping missing image {path}", file=sys.stderr)
            continue
        os.link(path.resolve(), tmp_dir / f"{rank}{path.suffix}")
        print(f"{score:.4f} | {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("--query", default="", help="The question (search query).")
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        type=Path,
        help="Directory containing images. If omitted, query persisted embeddings "
        f"in {DB_DIR} instead.",
    )
    args = parser.parse_args()

    model, processor = load("mlx-community/embeddinggemma-2-bf16")

    def embed(text=None, images=None) -> mx.array:
        inputs = processor(text=text, images=images, return_tensors="np")
        inputs = {key: mx.array(value) for key, value in inputs.items()}
        return model(**inputs).text_embeds

    query_embedding = embed(text=[f"task: search result | query: {args.query}"])

    if args.directory is not None and args.directory.is_dir():
        count, loaded = load_images(args.directory)

        ranked = []
        for batch in itertools.batched(loaded, IMAGE_BATCH_SIZE):
            batch_paths = [path for path, _ in batch]
            images = [image for _, image in batch]
            batch_embeddings = embed(images=[[image] for image in images])

            upsert_db(batch_paths, batch_embeddings)

            scores = (query_embedding @ batch_embeddings.T)[0]
            ranked.extend(
                (path, scores[i].item()) for i, path in enumerate(batch_paths)
            )

            print(
                f"Embedded {len(ranked)}/{count} images ({len(ranked) * 100 // count}%)",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
        if ranked:
            print(file=sys.stderr)
            ranked.sort(key=lambda item: item[1], reverse=True)
        else:
            print(f"No images found in {args.directory}", file=sys.stderr)
    else:
        ranked = query_db(query_embedding)

    if ranked:
        output_ranked(ranked)


if __name__ == "__main__":
    main()
