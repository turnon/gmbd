import argparse
import hashlib
import html
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


def write_gallery_html(
    tmp_dir: Path, entries: list[tuple[str, int, float, Path]]
) -> Path:
    """Write an index.html thumbnail gallery for the ranked results."""
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
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 1rem; }}
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
</style>
</head>
<body>
  <h1>{len(entries)} results</h1>
  <div class="grid">{''.join(cards)}
  </div>
</body>
</html>
"""
    index = tmp_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def output_ranked(ranked: list[tuple[Path, float]]) -> None:
    tmp_dir = Path("/tmp") / str(uuid.uuid7())
    tmp_dir.mkdir()

    entries = []
    for rank, (path, score) in enumerate(ranked):
        if not path.is_file():
            print(f"skipping missing image {path}", file=sys.stderr)
            continue
        link_name = f"{rank}{path.suffix}"
        os.link(path.resolve(), tmp_dir / link_name)
        entries.append((link_name, rank, score, path))

    index = write_gallery_html(tmp_dir, entries)
    print(index)


def main():
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("--query", default=None, help="The question (search query).")
    parser.add_argument(
        "--dir",
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

    query_embedding = (
        embed(text=[f"task: search result | query: {args.query}"])
        if args.query is not None
        else None
    )

    ranked = []
    if args.dir is not None and args.dir.is_dir():
        count, loaded = load_images(args.dir)

        embedded = 0
        for batch in itertools.batched(loaded, IMAGE_BATCH_SIZE):
            batch_paths = [path for path, _ in batch]
            images = [image for _, image in batch]
            batch_embeddings = embed(images=[[image] for image in images])

            upsert_db(batch_paths, batch_embeddings)

            if query_embedding is not None:
                scores = (query_embedding @ batch_embeddings.T)[0]
                ranked.extend(
                    (path, scores[i].item()) for i, path in enumerate(batch_paths)
                )

            embedded += len(batch_paths)
            print(
                f"Embedded {embedded}/{count} images ({embedded * 100 // count}%)",
                end="\r",
                file=sys.stderr,
                flush=True,
            )
        if loaded:
            print(file=sys.stderr)
        else:
            print(f"No images found in {args.dir}", file=sys.stderr)
        ranked.sort(key=lambda item: item[1], reverse=True)
    elif query_embedding is not None:
        ranked = query_db(query_embedding)

    if ranked:
        output_ranked(ranked)


if __name__ == "__main__":
    main()
