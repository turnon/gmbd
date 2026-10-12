import argparse
import hashlib
import os
import sys
import uuid
from pathlib import Path

import lancedb
import mlx.core as mx
from mlx_vlm import load
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
DB_DIR = Path.home() / ".gmbd" / "db"
TABLE_NAME = "images"


def load_images(directory: Path) -> tuple[list[Path], list[Image.Image]]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    images = [Image.open(path).convert("RGB") for path in paths]
    return paths, images


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
    print(f"Persisted {len(records)} embeddings to {DB_DIR}", file=sys.stderr)


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
        paths, images = load_images(args.directory)
        image_embeddings = embed(images=[[image] for image in images])

        upsert_db(paths, image_embeddings)

        scores = (query_embedding @ image_embeddings.T)[0]
        ranked_idx = mx.argsort(scores).tolist()[::-1]
        ranked = [(paths[i], scores[i].item()) for i in ranked_idx]
    else:
        ranked = query_db(query_embedding)

    if ranked:
        output_ranked(ranked)


if __name__ == "__main__":
    main()
