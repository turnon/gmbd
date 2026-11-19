# gmbd

Embed a question and rank images by similarity (MLX embeddinggemma + LanceDB or SQLite).

## Usage

```sh
uv run main.py --dir DIR            # embed every image under DIR into the db
uv run main.py --query "a red car"  # rank stored images and write a gallery html
uv run main.py --query ... --scope 'regex'   # restrict to matching directories
uv run main.py --server             # interactive search UI on http://0.0.0.0:8500
uv run main.py --tree               # show indexed directories as a tree
uv run main.py --clean              # drop records whose files no longer exist

uv run main.py --db sqlite ...      # use SQLite/sqlite-vec instead of LanceDB
uv run main.py --migrate lancedb.sqlite  # copy ~/.gmbd/db into ~/.gmbd/db.sqlite
```

`--db` accepts `lancedb` (default) or `sqlite`; it applies to indexing, queries,
the server, `--tree` and `--clean`.

## HTTP API

The server (`--server`, default `http://0.0.0.0:8500`) exposes:

- `GET /search?query=...&scope=...&page=1` — result gallery HTML, or JSON when the
  request sends `Accept: application/json` (or `Content-Type: application/json`).
- `POST /dir?path=/abs/dir` — index a directory in the background; returns a job id.
- `GET /jobs/<job_id>` — last progress line of that indexing job.

```sh
# JSON search results
curl -s 'http://localhost:8500/search?query=cat&scope=&page=1' \
  -H 'Accept: application/json'

# {"list": [{"path": "/abs/path/pic.jpg", "score": 0.83}], "total": 42}

# Index a directory
curl -s -X POST 'http://localhost:8500/dir?path=/abs/dir'
```
