# gmbd

Embed a question and rank images by similarity (MLX embeddinggemma + LanceDB).

## Usage

```sh
uv run main.py --dir DIR            # embed every image under DIR into the db
uv run main.py --query "a red car"  # rank stored images and write a gallery html
uv run main.py --query ... --scope 'regex'   # restrict to matching directories
uv run main.py --server             # interactive search UI on http://0.0.0.0:8500
uv run main.py --tree               # show indexed directories as a tree
uv run main.py --clean              # drop records whose files no longer exist
```
