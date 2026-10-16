"""Result galleries: pagination controls, index.html, and the temp-dir building.

page_count() is also used by gmbd.db to clamp the requested page of a search.
"""

import html
import os
import sys
import uuid
from pathlib import Path


def page_count(total: int, per_page: int) -> int:
    return max(1, (total + per_page - 1) // per_page)


def pagination_nav(page: int, pages: int) -> str:
    """Render prev/next and page-number submit buttons for the search form."""
    if pages <= 1:
        return ""

    def jump(target: int, label: str) -> str:
        return f'<button type="submit" name="page" value="{target}">{label}</button>'

    parts = []
    if page > 1:
        parts.append(jump(page - 1, "Prev"))
    else:
        parts.append('<button type="submit" disabled>Prev</button>')

    shown = [
        number
        for number in range(1, pages + 1)
        if number in (1, pages) or abs(number - page) <= 2
    ]
    previous = 0
    for number in shown:
        if previous and number - previous > 1:
            parts.append('<span class="gap">…</span>')
        if number == page:
            parts.append(f'<span class="current">{number}</span>')
        else:
            parts.append(jump(number, str(number)))
        previous = number

    if page < pages:
        parts.append(jump(page + 1, "Next"))
    else:
        parts.append('<button type="submit" disabled>Next</button>')

    return f'<nav class="pages">{"".join(parts)}</nav>'


def write_gallery_html(
    tmp_dir: Path,
    entries: list[tuple[str, int, float, Path]],
    query: str = "",
    scope: str = "",
    total: int | None = None,
    page: int = 1,
    pages: int = 1,
    per_page: int = 0,
) -> Path:
    """Write an index.html search form and thumbnail gallery for ranked results."""
    if total is None:
        total = len(entries)
    if pages > 1 and per_page:
        lo = (page - 1) * per_page + 1
        heading = f"{lo}\u2013{min(page * per_page, total)} of {total} results"
    elif pages > 1:
        heading = f"0 of {total} results"
    else:
        heading = f"{len(entries)} results"
    nav = pagination_nav(page, pages)

    cards = []
    for link_name, rank, score, path in entries:
        cards.append(
            f"""
      <figure class="card">
        <a href="{html.escape(link_name)}" target="_blank">
          <img loading="lazy" src="{html.escape(link_name)}" alt="rank {rank}">
        </a>
        <figcaption>
          <span class="rank">#{rank + 1}</span>
          <span class="score">{score:.4f}</span>
          <span class="path" title="{html.escape(str(path))}">{html.escape(str(path))}</span>
        </figcaption>
      </figure>"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gmbd results</title>
<style>
  body {{ margin: 0; padding: 1rem; background: #111; color: #eee;
         font: 14px/1.4 -apple-system, system-ui, sans-serif; }}
  .top {{ position: sticky; top: 0; z-index: 10; background: #111;
         margin: -1rem -1rem 1rem; padding: 1rem;
         border-bottom: 1px solid #2a2a2c; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0; white-space: nowrap; }}
  .search {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0; align-items: center; }}
  .status {{ display: flex; flex-wrap: nowrap; gap: 8px 12px; align-items: center;
            flex: 1 1 100%; justify-content: center; }}
  .search input {{ flex: 1 1 220px; padding: 8px 10px; border: 1px solid #3a3a3c;
                  border-radius: 8px; background: #1c1c1e; color: #eee; font: inherit; }}
  .search input:focus {{ outline: 2px solid #ffd60a; outline-offset: -1px; }}
  .search .go {{ padding: 8px 18px; border: 0; border-radius: 8px; background: #ffd60a;
                color: #111; font: inherit; font-weight: 600; cursor: pointer; }}
  .search .go:hover {{ background: #ffe14d; }}
  .grid {{ display: grid; gap: 12px;
          grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }}
  .card {{ margin: 0; background: #1c1c1e; border-radius: 10px; overflow: hidden;
          display: flex; flex-direction: column; }}
  .card img {{ width: 100%; height: 220px; object-fit: contain; display: block;
              background: #2a2a2c; }}
  figcaption {{ padding: 8px 10px; display: flex; flex-wrap: wrap; gap: 6px 10px;
               align-items: baseline; }}
  .rank {{ font-weight: 700; color: #ffd60a; }}
  .score {{ font-variant-numeric: tabular-nums; color: #7dd3fc; }}
  .path {{ flex: 1 1 100%; color: #98989d; font-size: 11px; overflow-wrap: anywhere;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .pages {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0; }}
  .pages button, .pages span {{ padding: 8px 10px; border: 0; border-radius: 8px;
                              color: #eee; background: #1c1c1e; font: inherit; }}
  .pages button {{ cursor: pointer; }}
  .pages button:hover {{ background: #2a2a2c; }}
  .pages button:disabled {{ opacity: 0.35; cursor: default; }}
  .pages button:disabled:hover {{ background: #1c1c1e; }}
  .pages .current {{ background: #ffd60a; color: #111; font-weight: 700; }}
  .pages .gap {{ background: transparent; color: #98989d; }}
</style>
</head>
<body>
  <header class="top">
    <form class="search" method="get" action="/">
      <input type="search" name="query" placeholder="query" value="{html.escape(query)}" autofocus>
      <input type="search" name="scope" placeholder="scope (directory regex)" value="{html.escape(scope)}">
      <button class="go" type="submit">Search</button>
      <div class="status">
        {nav}
        <h1>{heading}</h1>
      </div>
    </form>
  </header>
  <div class="grid">{"".join(cards)}
  </div>
</body>
</html>
"""
    index = tmp_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def build_gallery(
    ranked: list[tuple[Path, float]],
    query: str = "",
    scope: str = "",
    page: int = 1,
    per_page: int | None = None,
    total: int | None = None,
    prefix_links: bool = False,
) -> tuple[Path, Path]:
    """Hard-link ranked results into a fresh temp dir and write the gallery html.

    ranked is expected to already hold just one page of results when per_page is
    given (as returned by query_db); total then carries the full result count for
    the pagination controls. When prefix_links is set, thumbnail URLs are made
    relative to the gallery id (used by the server, which serves each gallery
    under its own path). Returns (tmp_dir, index.html path).
    """
    tmp_dir = Path("/tmp") / str(uuid.uuid7())
    tmp_dir.mkdir()
    link_prefix = f"{tmp_dir.name}/" if prefix_links else ""

    if total is None:
        total = len(ranked)
    if per_page:
        pages = page_count(total, per_page)
        page = min(max(1, page), pages)
        start = (page - 1) * per_page
    else:
        pages, page, per_page, start = 1, 1, 0, 0

    entries = []
    skipped = 0
    for rank, (path, score) in enumerate(ranked, start=start):
        if not path.is_file():
            skipped += 1
            continue
        filename = f"{rank}{path.suffix}"
        os.link(path.resolve(), tmp_dir / filename)
        entries.append((f"{link_prefix}{filename}", rank, score, path))

    if skipped:
        print(f"skipped {skipped} missing image(s)", file=sys.stderr)

    index = write_gallery_html(
        tmp_dir,
        entries,
        query=query,
        scope=scope,
        total=total,
        page=page,
        pages=pages,
        per_page=per_page,
    )
    return tmp_dir, index


def output_ranked(ranked: list[tuple[Path, float]]) -> None:
    _, index = build_gallery(ranked)
    print(index)
