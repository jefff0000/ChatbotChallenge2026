"""Image pipeline. STUB. Workshop 2 blocks 2 and 3.

Describe every collected image once, cache the result, and add the
descriptions to the same store as the scraped text. Run this file
directly, after build/scrape.py and build/index.py, to add images to
data/chroma.

Never describe an image while answering a question. Ingestion time is
unlimited; runtime is 30 seconds.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.llm import describe_image
from bot.store import add_to_store, get_store

DESCRIPTION_PROMPT = """
TODO [W2 b3] Write this.

A caption written for a human ("a room with modern furniture and
students working") cannot answer "how many tables are in Makerspace A"
or "what three words are on the wall behind the brainstorming area".

What would this prompt have to ask for so that both are answerable?
"""

CACHE_PATH = Path("data/descriptions.json")
GRAPH_PATH = Path("data/graphrag.json")


def describe_all(images: list[dict]) -> dict:
    """Describe every image once, cache the result, never regenerate.

    Ingestion is free, but not if you redo it every time you change a
    line downstream. Key the cache by image URL.
    """
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    total = len(images)
    print(f"[images] describing {total} images")

    for i, im in enumerate(images, start=1):
        src = im["src"]
        if src in cache:
            print(f"[images] describe {i:4d}/{total} cache hit | {src}")
            continue
        try:
            tmp = Path("data/_tmp_image")
            tmp.write_bytes(requests.get(src, timeout=30).content)
            cache[src] = describe_image(str(tmp), DESCRIPTION_PROMPT)
            print(f"[images] describe {i:4d}/{total} done      | {src}")
        except Exception as exc:
            print("failed:", src[:70], exc)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    print(f"{len(cache)} descriptions cached")
    return cache


def index_descriptions(descriptions: dict, images: list[dict]) -> None:
    """Add descriptions to the same store as the scraped text."""
    by_src = {im["src"]: im for im in images}
    texts, metas, ids = [], [], []
    page_to_image_ids = defaultdict(list)
    total = len(descriptions)
    for i, (src, text) in enumerate(descriptions.items(), start=1):
        im = by_src.get(src, {})
        # Alt text and captions are already text and cost nothing to index.
        full = " ".join(filter(None, [im.get("alt"), im.get("caption"), text]))
        image_id = f"img_{len(ids)}"
        texts.append(full)
        meta = {"url": im.get("page", src), "image": src, "kind": "image", "cid": image_id}
        metas.append(meta)
        ids.append(image_id)
        page_to_image_ids[meta["url"]].append(image_id)
        print(
            f"[images] index {i:4d}/{total} prepared | "
            f"total prepared={len(ids):4d} | {meta['url']}"
        )
    store = get_store(reset=False)
    print(f"[images] writing {len(texts)} image descriptions to Chroma")
    add_to_store(store, texts, metas, ids=ids)
    _attach_images_to_graph(texts, metas, page_to_image_ids)
    print(f"indexed {len(texts)} image descriptions")


def _attach_images_to_graph(texts: list[str], metas: list[dict], page_to_image_ids: dict) -> None:
    graph = {"nodes": {}, "neighbors": {}}
    if GRAPH_PATH.exists():
        graph = json.loads(GRAPH_PATH.read_text())
    nodes = graph.setdefault("nodes", {})
    neighbors = graph.setdefault("neighbors", {})

    page_to_text_ids = defaultdict(list)
    for cid, node in nodes.items():
        meta = node.get("metadata", {})
        if meta.get("kind") == "text":
            page_to_text_ids[meta.get("url", "")].append(cid)

    for i, (t, m) in enumerate(zip(texts, metas), start=1):
        nodes[m["cid"]] = {"text": t, "metadata": m}
        neighbors.setdefault(m["cid"], [])
        print(f"[graph] node {i:4d}/{len(metas)} added | {m['cid']}")

    for page, image_ids in page_to_image_ids.items():
        text_ids = page_to_text_ids.get(page, [])
        for iid in image_ids:
            for tid in text_ids:
                if tid not in neighbors[iid]:
                    neighbors[iid].append(tid)
                neighbors.setdefault(tid, [])
                if iid not in neighbors[tid]:
                    neighbors[tid].append(iid)
        print(
            f"[graph] linked page nodes | page={page} | "
            f"images={len(image_ids)} | text_chunks={len(text_ids)}"
        )

    GRAPH_PATH.parent.mkdir(exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph))


if __name__ == "__main__":
    images = json.loads(Path("data/images.json").read_text())
    descriptions = describe_all(images)
    index_descriptions(descriptions, images)
