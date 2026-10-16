"""gmbd — embed a question and rank images by similarity.

Modules:
    cli         argument parsing and command dispatch
    embed       load the embedding model
    indexing    scan image directories and persist embeddings
    db          everything LanceDB: connection, schema, queries, tree/clean
    gallery     pagination controls, index.html, temp-dir result galleries
    server      interactive HTTP search UI
"""

__version__ = "0.1.0"
