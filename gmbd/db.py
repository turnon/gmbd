"""Everything that touches LanceDB lives in this module.

ImageDB owns the connection, the schema and all queries; the tree/clean
maintenance reports at the bottom are built on top of it. No other module
imports lancedb or knows about table columns and SQL.
"""

import itertools
import os
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import lancedb
import mlx.core as mx
import pyarrow as pa
from lancedb.index import BTree

DB_DIR = Path.home() / ".gmbd" / "db"
TABLE_NAME = "images"


class ImageDB:
    """Read/write access to the stored image embeddings."""

    def __init__(self, db_dir: Path = DB_DIR) -> None:
        self.db_dir = db_dir
        self._db = lancedb.connect(db_dir)

    @property
    def exists(self) -> bool:
        """True once the images table has been created."""
        return self._open() is not None

    def _open(self):
        """Open the images table, or None when nothing has been indexed yet."""
        if TABLE_NAME not in self._db.list_tables().tables:
            return None
        return self._db.open_table(TABLE_NAME)

    def _open_or_create(self, vector_dim: int):
        table = self._open()
        if table is not None:
            return table

        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("dir", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            ]
        )
        table = self._db.create_table(TABLE_NAME, schema=schema)
        table.create_index("id", config=BTree())
        return table

    def fetch_ids(self, ids: list[str]) -> list[dict]:
        """Return the stored records matching the given ids."""
        table = self._open()
        if table is None:
            return []

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

    def insert(
        self, paths: list[Path], ids: list[str], vectors: list[list[float]]
    ) -> None:
        """Store one record per path, creating the table on first use."""
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

        self._open_or_create(len(vectors[0])).add(records)

    def search(
        self,
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
        table = self._open()
        if table is None:
            return [], 0

        filter = None
        if scope is not None:
            quoted_scope = scope.replace("'", "''")
            filter = f"regexp_like(dir, '{quoted_scope}')"

        total = table.count_rows(filter=filter)
        if not total:
            # Nothing matches; a limit of 0 would be rejected by LanceDB.
            return [], 0
        if per_page:
            page = min(max(1, page), self.page_count(total, per_page))
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

    def page_count(self, total: int, per_page: int) -> int:
        return max(1, (total + per_page - 1) // per_page)

    def directory_counts(self) -> Counter[str]:
        """Number of stored records per directory."""
        counts: Counter[str] = Counter()
        table = self._open()
        if table is None:
            return counts
        for batch in table.search().select(["dir"]).to_batches():
            counts.update(batch.column("dir").to_pylist())
        return counts

    def clean_missing(self) -> tuple[int, int]:
        """Delete records whose image files no longer exist on disk.

        Returns (number of removed records, total records before cleaning).
        """
        table = self._open()
        total = table.count_rows()
        missing_ids = []
        for batch in table.search().select(["id", "dir", "name"]).to_batches():
            for row in batch.to_pylist():
                if not (Path(row["dir"]) / row["name"]).is_file():
                    missing_ids.append(row["id"])

        for chunk in itertools.batched(missing_ids, 500):
            ids = ", ".join(f"'{record_id}'" for record_id in chunk)
            table.delete(f"id IN ({ids})")

        self.optimize()
        return len(missing_ids), total

    def optimize(self) -> None:
        """Compact the table after a batch of inserts or deletes."""
        table = self._open()
        if table is not None:
            table.optimize(cleanup_older_than=timedelta(milliseconds=1))


def tree_db() -> None:
    """Print the directories of all persisted records in tree style."""
    db = ImageDB()
    if not db.exists:
        print(
            f"No embeddings found in {db.db_dir}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    counts = db.directory_counts()
    if not counts:
        print(f"No records found in {db.db_dir}", file=sys.stderr)
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
    db = ImageDB()
    if not db.exists:
        print(
            f"No embeddings found in {db.db_dir}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    removed, total = db.clean_missing()
    print(f"Removed {removed} of {total} record(s) with missing files")
