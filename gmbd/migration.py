"""Migrate embeddings from LanceDB to the SQLite (sqlite-vec) backend.

Target schema (created by gmbd.sqlit.create_schema):

    CREATE VIRTUAL TABLE images USING vec0(
        id     INTEGER PRIMARY KEY,
        name   TEXT,
        dir    TEXT,
        digest TEXT,
        vector FLOAT[768] distance_metric=cosine
    )

Column mapping:
    LanceDB (none)               -> images.id       (auto-generated 1..N)
    LanceDB name      (TEXT)     -> images.name
    LanceDB dir       (TEXT)     -> images.dir
    LanceDB id        (TEXT)     -> images.digest
    LanceDB vector    (list<f32>)-> images.vector   (BLOB, 768 floats)

Called from the CLI with:

    uv run main.py --migrate lancedb.sqlite
"""

from pathlib import Path

import lancedb
import numpy as np

from gmbd import lance, sqlit


def migrate(
    lancedb_path: Path = lance.DB_DIR,
    table_name: str = lance.TABLE_NAME,
    sqlite_path: Path = sqlit.DB_PATH,
    batch_size: int = 1000,
    wipe: bool = True,
) -> int:
    """Copy every row of a LanceDB table into a sqlite-vec table."""
    # ---- 1. read from LanceDB ----
    ldb = lancedb.connect(lancedb_path)
    if table_name not in ldb.list_tables().tables:
        raise ValueError(f"No '{table_name}' table found in {lancedb_path}")
    table = ldb.open_table(table_name)
    arrow = table.to_arrow()
    total = arrow.num_rows
    print(f"[lancedb] read {total:,} rows from '{table_name}'")

    missing = {"id", "dir", "name", "vector"} - set(arrow.column_names)
    if missing:
        raise ValueError(f"LanceDB table is missing columns: {missing}")

    # ---- 2. open (or recreate) SQLite ----
    out = Path(sqlite_path).expanduser().resolve()
    if wipe:
        for suffix in ("", "-wal", "-shm"):
            stale = out.with_name(out.name + suffix)
            if stale.exists():
                print(f"[sqlite] removing existing {stale}")
                stale.unlink()
    conn = sqlit.connect(out)
    sqlit.create_schema(conn)
    cur = conn.cursor()

    if not wipe:
        cur.execute(f"DELETE FROM {sqlit.TABLE_NAME}")
        conn.commit()

    # ---- 3. batch insert ----
    inserted = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        rows = []
        for local_i, record in enumerate(arrow.slice(start, end - start).to_pylist()):
            new_id = start + local_i + 1  # 1-based PK
            vector = np.asarray(record["vector"], dtype=np.float32)
            if vector.shape != (sqlit.VECTOR_DIM,):
                raise ValueError(
                    f"Row {new_id}: expected shape ({sqlit.VECTOR_DIM},), "
                    f"got {vector.shape}"
                )
            rows.append(
                (
                    new_id,
                    record["name"],
                    record["dir"],
                    record["id"],  # LanceDB id -> SQLite digest
                    vector.tobytes(),
                )
            )

        cur.executemany(
            "INSERT INTO images (id, name, dir, digest, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        inserted += len(rows)
        print(f"[sqlite]  inserted rows {start + 1:,}..{end:,}")

    # ---- 4. sanity check ----
    n = cur.execute(f"SELECT COUNT(*) FROM {sqlit.TABLE_NAME}").fetchone()[0]
    print(f"[check] images={n:,}  expected={total:,}")
    if n != total or inserted != total:
        raise RuntimeError("Row-count mismatch between LanceDB and SQLite")

    conn.close()
    print(f"[done] wrote {total:,} rows into {out}")
    return total
