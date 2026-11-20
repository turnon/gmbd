"""gmbd — embed a question and rank images by similarity.

Modules:
    cli         argument parsing and command dispatch
    embed       load the embedding model
    indexing    scan image directories and persist embeddings
    lance       everything LanceDB: connection, schema, queries
    sqlit       everything SQLite/sqlite-vec: connection, schema, queries
    migration   copy embeddings from LanceDB to SQLite
    gallery     pagination controls, index.html, temp-dir result galleries
    server      interactive HTTP search UI
    ollama      Ollama-compatible embedding API (POST /api/embed)
"""

__version__ = "0.1.0"
