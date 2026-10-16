"""Command-line interface: index images, query them, and maintenance commands."""

import argparse
import sys
from pathlib import Path

from gmbd.db import DB_DIR, ImageDB, clean_db, tree_db
from gmbd.embed import load_embedder
from gmbd.gallery import output_ranked
from gmbd.indexing import index_directory
from gmbd.server import run_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("--query", default=None, help="The question (search query).")
    parser.add_argument(
        "--dir",
        default=None,
        type=Path,
        help="Directory containing images. Images whose content is already in "
        "the database are skipped. If omitted, query persisted embeddings "
        f"in {DB_DIR} instead.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tree:
        tree_db()
        return

    if args.clean:
        clean_db()
        return

    embed = load_embedder()

    if args.server:
        run_server(embed, query=args.query, scope=args.scope)
        return

    if args.dir is not None:
        if not args.dir.is_dir():
            print(f"Not a directory: {args.dir}", file=sys.stderr)
        else:
            index_directory(args.dir, embed)

    if args.query is not None:
        query_embedding = embed(text=[f"task: search result | query: {args.query}"])
        ranked, _ = ImageDB().search(query_embedding, scope=args.scope)
        if ranked:
            output_ranked(ranked)
