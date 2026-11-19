"""Everything that touches LanceDB lives in this module.

LanceDB owns the connection, the schema and all queries; the tree/clean
maintenance reports in gmbd.cli are built on top of it. The SQLite backend in
gmbd.sqlit exposes the same interface, so cli/indexing/server stay backend
agnostic.
"""

import itertools
from collections import Counter
from datetime import timedelta
from pathlib import Path

import lancedb
import mlx.core as mx
import pyarrow as pa
from lancedb.index import BTree

DB_DIR = Path.home() / ".gmbd" / "db"
TABLE_NAME = "images"


class LanceDB:
    """Read/write access to the stored image embeddings."""

    kind = "lancedb"

    def __init__(self, db_dir: Path = DB_DIR) -> None:
        self.db_dir = db_dir
        self._db = lancedb.connect(db_dir)

    @property
    def location(self) -> Path:
        """Where this backend persists its data."""
        return self.db_dir

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

