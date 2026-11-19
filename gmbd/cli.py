"""Command-line interface: index images, query them, and maintenance commands."""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from gmbd import lance, sqlit
from gmbd.embed import load_embedder
from gmbd.gallery import output_ranked
from gmbd.indexing import index_directory
from gmbd.migration import migrate
from gmbd.server import run_server

DB_CHOICES = ("lancedb", "sqlite")


def open_db(kind: str):
    """Return the selected storage backend (same interface for both)."""
    if kind == "sqlite":
        return sqlit.SQLiteDB()
    return lance.LanceDB()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("--query", default=None, help="The question (search query).")
    parser.add_argument(
        "--db",
        choices=DB_CHOICES,
        default="lancedb",
        help="Storage backend (default: lancedb).",
    )
    parser.add_argument(
        "--dir",
        default=None,
        type=Path,
        help="Directory containing images. Images whose content is already in "
        "the database are skipped. If omitted, query persisted embeddings "
        f"in {lance.DB_DIR} or {sqlit.DB_PATH} instead.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Restrict query results to images whose stored directory (the "
        "images.dir column) matches this regular expression.",
        metavar="REGEX",
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
    parser.add_argument(
        "--server",
        action="store_true",
        help="Serve the result gallery over HTTP on 0.0.0.0:8500 with the built-in "
        "http server, so queries can be run interactively through the search form.",
    )
    parser.add_argument(
        "--migrate",
        choices=("lancedb.sqlite",),
        default=None,
        help="Migrate all rows from the LanceDB database into SQLite.",
    )
    return parser.parse_args()


def tree_db(db) -> None:
    """Print the directories of all persisted records in tree style."""
    if not db.exists:
        print(
            f"No embeddings found in {db.location}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    counts: Counter[str] = db.directory_counts()
    if not counts:
        print(f"No records found in {db.location}", file=sys.stderr)
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


def clean_db(db) -> None:
    """Remove persisted records whose image files no longer exist on disk."""
    if not db.exists:
        print(
            f"No embeddings found in {db.location}; run with --dir DIR first.",
            file=sys.stderr,
        )
        return

    removed, total = db.clean_missing()
    print(f"Removed {removed} of {total} record(s) with missing files")


def main() -> None:
    args = parse_args()

    if args.migrate == "lancedb.sqlite":
        migrate()
        return

    db = open_db(args.db)

    if args.tree:
        tree_db(db)
        return

    if args.clean:
        clean_db(db)
        return

    embed = load_embedder()

    if args.server:
        run_server(embed, db, query=args.query, scope=args.scope)
        return

    if args.dir is not None:
        if not args.dir.is_dir():
            print(f"Not a directory: {args.dir}", file=sys.stderr)
        else:
            index_directory(args.dir, embed, db)

    if args.query is not None:
        query_embedding = embed(text=[f"task: search result | query: {args.query}"])
        ranked, _ = db.search(query_embedding, scope=args.scope)
        if ranked:
            output_ranked(ranked)
