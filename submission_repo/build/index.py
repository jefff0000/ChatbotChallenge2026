"""Indexer. STUB. Workshop 1 block 4.

Chunk the scraped text and write it to the store. Run this file
directly, after build/scrape.py, to (re)build data/chroma.

Store metadata now. Level 4 questions need to filter by year and page
type, and adding a field later means rebuilding everything.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.store import add_to_store, get_store

CHUNK_SIZE, OVERLAP = 800, 100


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunks with overlap.

    Overlap exists because a fact split across a boundary is lost: a
    date at char 998 and its event name at char 1002 land in different
    chunks and neither answers the question.

    TODO try a better strategy once this works: split on headings
    TODO first, fall back to characters last, and prepend the section
    TODO heading to each chunk.
    """
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def build_index(pages: list[dict], reset: bool = True):
    texts, metas, ids = [], [], []
    page_to_ids = defaultdict(list)
    cid = 0
    total_pages = len(pages)
    print(f"[index] building chunks from {total_pages} pages")
    for p_i, page in enumerate(pages, start=1):
        pieces = chunk(page.get("text", ""))
        for i, piece in enumerate(pieces):
            chunk_id = f"c{cid}"
            cid += 1
            texts.append(piece)
            meta = {
                "url":      page["url"],
                "title":    page.get("title", ""),
                "position": i,
                "cid":      chunk_id,
                "kind":     "text",
                # TODO add "year" and "page_type" here. Level 4 needs them.
            }
            metas.append(meta)
            ids.append(chunk_id)
            page_to_ids[page["url"]].append(chunk_id)
        print(
            f"[index] page {p_i:4d}/{total_pages} | "
            f"chunks this page={len(pieces):3d} | total chunks={len(texts):5d} | "
            f"{page.get('url', '?')}"
        )
    store = get_store(reset=reset)
    print(f"[index] writing {len(texts)} chunks to Chroma")
    add_to_store(store, texts, metas, ids=ids)
    _write_graphrag(texts, metas, page_to_ids)
    print(f"indexed {len(texts)} chunks")
    return store


def _write_graphrag(texts: list[str], metas: list[dict], page_to_ids: dict) -> None:
    nodes = {}
    neighbors = defaultdict(list)
    for t, m in zip(texts, metas):
        nodes[m["cid"]] = {"text": t, "metadata": m}

    for ids in page_to_ids.values():
        for i, cid in enumerate(ids):
            if i > 0:
                neighbors[cid].append(ids[i - 1])
            if i + 1 < len(ids):
                neighbors[cid].append(ids[i + 1])

    Path("data").mkdir(exist_ok=True)
    Path("data/graphrag.json").write_text(json.dumps({
        "nodes": nodes,
        "neighbors": neighbors,
    }))


if __name__ == "__main__":
    pages = json.loads(Path("data/pages.json").read_text())
    build_index(pages)
