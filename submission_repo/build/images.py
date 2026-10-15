"""Image pipeline. STUB. Workshop 2 blocks 2 and 3.

Describe every collected image once, cache the result, and add the
descriptions to the same store as the scraped text. Run this file
directly, after build/scrape.py and build/index.py, to add images to
data/chroma.

Never describe an image while answering a question. Ingestion time is
unlimited; runtime is 30 seconds.
"""
import json
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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
CLIP_INDEX_PATH = Path("data/clip_image_index.json")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def describe_all(images: list[dict]) -> dict:
    """Describe every image once, cache the result, never regenerate.

    Ingestion is free, but not if you redo it every time you change a
    line downstream. Key the cache by image URL.
    """
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    for im in images:
        src = im["src"]
        if src in cache:
            continue
        try:
            tmp = Path("data/_tmp_image")
            tmp.write_bytes(requests.get(src, timeout=30).content)
            cache[src] = describe_image(str(tmp), DESCRIPTION_PROMPT)
        except Exception as exc:
            print("failed:", src[:70], exc)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    print(f"{len(cache)} descriptions cached")
    return cache


def index_descriptions(descriptions: dict, images: list[dict]) -> None:
    """Add descriptions to the same store as the scraped text."""
    by_src = {im["src"]: im for im in images}
    texts, metas = [], []
    for src, text in descriptions.items():
        im = by_src.get(src, {})
        # Alt text and captions are already text and cost nothing to index.
        full = " ".join(filter(None, [im.get("alt"), im.get("caption"), text]))
        texts.append(full)
        metas.append({"url": im.get("page", src), "image": src, "kind": "image"})
    store = get_store(reset=False)
    add_to_store(store, texts, metas, ids=[f"img_{i}" for i in range(len(texts))])
    print(f"indexed {len(texts)} image descriptions")


def build_clip_index(descriptions: dict, images: list[dict]) -> None:
    """Build CLIP embedding index for image retrieval at runtime."""
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    index = []
    for im in images:
        src = im["src"]
        try:
            img = Image.open(requests.get(src, timeout=30, stream=True).raw).convert("RGB")
            with torch.no_grad():
                inp = processor(images=img, return_tensors="pt")
                emb = model.get_image_features(**inp)[0].cpu().tolist()
            index.append({
                "src": src,
                "page": im.get("page", src),
                "alt": im.get("alt", ""),
                "caption": im.get("caption", ""),
                "description": descriptions.get(src, ""),
                "embedding": emb,
            })
        except Exception as exc:
            print("clip failed:", src[:70], exc)

    CLIP_INDEX_PATH.parent.mkdir(exist_ok=True)
    CLIP_INDEX_PATH.write_text(json.dumps(index))
    print(f"indexed {len(index)} CLIP image embeddings")


if __name__ == "__main__":
    images = json.loads(Path("data/images.json").read_text())
    descriptions = describe_all(images)
    index_descriptions(descriptions, images)
    build_clip_index(descriptions, images)
