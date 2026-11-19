"""Everything that touches SQLite (sqlite-vec) lives in this module.

SQLiteDB exposes the same interface as gmbd.lance.LanceDB, so the rest of the
package can use either backend without knowing about tables or SQL. Each row
stores the image metadata in a single sqlite-vec `vec0` table.
"""

import itertools
import re
import sqlite3
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
import sqlite_vec

DB_PATH = Path.home() / ".gmbd" / "db.sqlite"
TABLE_NAME = "images"
VECTOR_DIM = 768


def connect(db_path: Path) -> sqlite3.Connection:
    """Open db_path with sqlite-vec loaded and a regexp() SQL function."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.create_function(
        "regexp", 2, lambda pattern, value: bool(re.search(pattern, value or ""))
    )
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the vec0 table used by both indexing and migration."""
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME} USING vec0(
            id     INTEGER PRIMARY KEY,
            name   TEXT,
            dir    TEXT,
            digest TEXT,
            vector FLOAT[{VECTOR_DIM}] distance_metric=cosine
        )
        """
    )
    conn.commit()


def vector_blob(vector) -> bytes:
    """Pack an embedding (list or array) into the BLOB vec0 expects."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def vector_list(blob: bytes) -> list[float]:
    """Unpack a stored float32 BLOB back into a plain Python list."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


class SQLiteDB:
    """Read/write access to the stored image embeddings."""

    kind = "sqlite"

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def location(self) -> Path:
        """Where this backend persists its data."""
        return self.db_path

    @property
    def exists(self) -> bool:
        """True once the images table has been created."""
        return self._open() is not None

    def _open(self) -> sqlite3.Connection | None:
        """Open the database, or return None when nothing has been indexed yet."""
        if self._conn is not None:
            return self._conn
        if not self.db_path.is_file():
            return None
        conn = connect(self.db_path)
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()
        if table is None:
            conn.close()
            return None
        self._conn = conn
        return conn

    def _open_or_create(self) -> sqlite3.Connection:
        conn = self._open()
        if conn is not None:
            return conn
        conn = connect(self.db_path)
        create_schema(conn)
        self._conn = conn
        return conn

    def fetch_ids(self, ids: list[str]) -> list[dict]:
        """Return the stored records matching the given digests."""
        conn = self._open()
        if conn is None:
            return []

        stored = []
        for chunk in itertools.batched(sorted(ids), 500):
            placeholders = ", ".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT dir, name, digest, vector FROM images "
                f"WHERE digest IN ({placeholders})",
                chunk,
            )
            stored.extend(
                {
                    "id": digest,
                    "dir": directory,
                    "name": name,
                    "vector": vector_list(vector),
                }
                for directory, name, digest, vector in rows
            )
        return stored

    def insert(
        self, paths: list[Path], ids: list[str], vectors: list[list[float]]
    ) -> None:
        """Store one record per path, creating the table on first use."""
        conn = self._open_or_create()
        rows = []
        for path, record_id, vector in zip(paths, ids, vectors):
            resolved = path.resolve()
            rows.append(
                (path.name, str(resolved.parent), record_id, vector_blob(vector))
            )
        conn.executemany(
            "INSERT INTO images (name, dir, digest, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    def search(
        self,
        embedding: mx.array,
        scope: str | None = None,
        page: int = 1,
        per_page: int | None = None,
    ) -> tuple[list[tuple[Path, float]], int]:
        """Rank images by cosine similarity against embeddings in SQLite.

        When per_page is given, only the requested page of the ranking is
        returned, and page is clamped to the page count. Returns
        (ranked results, total matching rows).
        """
        conn = self._open()
        if conn is None:
            return [], 0

        if scope is not None:
            total = conn.execute(
                "SELECT count(*) FROM images WHERE dir REGEXP ?", (scope,)
            ).fetchone()[0]
        else:
            total = conn.execute("SELECT count(*) FROM images").fetchone()[0]
        if not total:
            return [], 0

        if per_page:
            page = min(max(1, page), self.page_count(total, per_page))
            offset = (page - 1) * per_page
            k = offset + per_page
        else:
            offset = 0
            k = total

        query = "SELECT dir, name, distance FROM images WHERE vector MATCH ? AND k = ?"
        params: list = [
            vector_blob(embedding.astype(mx.float32)[0].tolist()),
            k,
        ]
        if scope is not None:
            query += " AND id IN (SELECT id FROM images WHERE dir REGEXP ?)"
            params.append(scope)

        rows = conn.execute(query, params).fetchall()
        if per_page:
            rows = rows[offset:]

        # vec0 reports cosine distance; 1 - distance is the similarity score.
        ranked = [
            (Path(directory) / name, 1.0 - distance)
            for directory, name, distance in rows
        ]
        return ranked, total

    def page_count(self, total: int, per_page: int) -> int:
        return max(1, (total + per_page - 1) // per_page)

    def directory_counts(self) -> Counter[str]:
        """Number of stored records per directory."""
        counts: Counter[str] = Counter()
        conn = self._open()
        if conn is None:
            return counts
        rows = conn.execute(
            f"SELECT dir, count(*) FROM {TABLE_NAME} GROUP BY dir"
        )
        for directory, count in rows:
            counts[directory] = count
        return counts

    def clean_missing(self) -> tuple[int, int]:
        """Delete records whose image files no longer exist on disk.

        Returns (number of removed records, total records before cleaning).
        """
        conn = self._open()
        if conn is None:
            return 0, 0

        total = conn.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
        missing_ids = [
            record_id
            for record_id, directory, name in conn.execute(
                f"SELECT id, dir, name FROM {TABLE_NAME}"
            )
            if not (Path(directory) / name).is_file()
        ]

        for chunk in itertools.batched(missing_ids, 500):
            placeholders = ", ".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM {TABLE_NAME} WHERE id IN ({placeholders})", chunk
            )
        conn.commit()

        self.optimize()
        return len(missing_ids), total

    def optimize(self) -> None:
        """Commit pending work; vec0 has no index to compact."""
        conn = self._open()
        if conn is not None:
            conn.commit()
