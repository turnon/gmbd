import argparse
import hashlib
import html
import itertools
import os
import sys
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import lancedb
import mlx.core as mx
import pyarrow as pa
from lancedb.index import BTree
from mlx_vlm import load
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
IMAGE_BATCH_SIZE = 32
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


def upsert_db(paths: list[Path], embeddings: mx.array) -> lancedb.table.LanceTable:
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
    if TABLE_NAME not in db.list_tables().tables:
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("dir", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), embeddings.shape[1])),
            ]
        )
        table = db.create_table(TABLE_NAME, schema=schema)
        table.create_index("id", config=BTree())
    else:
        table = db.open_table(TABLE_NAME)

    (
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(records)
    )
    return table


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


def output_ranked(ranked: list[tuple[Path, float]]) -> None:
    tmp_dir = Path("/tmp") / str(uuid.uuid7())
    tmp_dir.mkdir()

    entries = []
    skipped = 0
    for rank, (path, score) in enumerate(ranked):
        if not path.is_file():
            skipped += 1
            continue
        link_name = f"{rank}{path.suffix}"
        os.link(path.resolve(), tmp_dir / link_name)
        entries.append((link_name, rank, score, path))

    if skipped:
        print(f"skipped {skipped} missing image(s)", file=sys.stderr)

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
    args = parser.parse_args()

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

    query_embedding = (
        embed(text=[f"task: search result | query: {args.query}"])
        if args.query is not None
        else None
    )

    ranked = []
    if args.dir is not None and args.dir.is_dir():
        count, loaded = load_images(args.dir)

        embedded = 0
        table = None
        for batch in itertools.batched(loaded, IMAGE_BATCH_SIZE):
            batch_paths = [path for path, _ in batch]
            images = [image for _, image in batch]
            batch_embeddings = embed(images=[[image] for image in images])

            table = upsert_db(batch_paths, batch_embeddings)

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
        if table is not None:
            table.optimize(cleanup_older_than=timedelta(milliseconds=1))
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
