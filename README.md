# gmbd

Embed a question and rank images by similarity (MLX embeddinggemma + LanceDB or SQLite).

## Usage

```sh
uv run main.py --dir DIR            # embed every image under DIR into the db
uv run main.py --query "a red car"  # rank stored images and write a gallery html
uv run main.py --query ... --scope 'regex'   # restrict to matching directories
uv run main.py --server             # interactive search UI on http://0.0.0.0:8500
uv run main.py --embedding          # Ollama-compatible embedding API on http://0.0.0.0:11434
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

## Ollama embedding API

`--embedding` serves `POST /api/embed` on `http://0.0.0.0:11434`, mimicking
Ollama's embed API. `input` may be a string, an array of strings, or an array of
`{"image": "<base64>"}` objects (a raw base64 string or a `data:image/...` URI).
The request's `model` field is ignored; the response always reports
`mlx-community/embeddinggemma-2-bf16` and includes `embeddings`.

```sh
curl -s http://localhost:11434/api/embed \
  -d '{
    "model": "mlx-community/embeddinggemma-2-bf16",
    "input": [
      { "image": "<base64 image 1>" },
      { "image": "<base64 image 2>" }
    ]
  }'

curl -s http://localhost:11434/api/embed -d '{"input": "a red car"}'
curl -s http://localhost:11434/api/embed -d '{"input": ["a red car", "a blue car"]}'
```
