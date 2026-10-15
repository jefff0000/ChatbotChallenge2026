"""CLIP-based image retrieval (optional)."""
import json
import math
from functools import lru_cache
from pathlib import Path

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # noqa: BLE001
    torch = None
    CLIPModel = None
    CLIPProcessor = None

INDEX_PATH = Path("data/clip_image_index.json")
MODEL_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def _load_model():
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        return None, None
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        model.eval()
        return model, processor
    except Exception:  # noqa: BLE001
        return None, None


@lru_cache(maxsize=1)
def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text())


def retrieve_images(question: str, top_k: int = 2) -> list[dict]:
    """Return top-k CLIP matched images for a question."""
    index = _load_index()
    model, processor = _load_model()
    if not index or model is None or processor is None:
        return []

    with torch.no_grad():
        t = processor(text=[question], return_tensors="pt", padding=True)
        q = model.get_text_features(**t)[0].cpu().tolist()

    qn = math.sqrt(sum(v * v for v in q)) + 1e-12
    ranked = []
    for item in index:
        emb = item.get("embedding", [])
        if not emb:
            continue
        en = math.sqrt(sum(v * v for v in emb)) + 1e-12
        score = float(sum(a * b for a, b in zip(q, emb)) / (qn * en))
        ranked.append((score, item))
    ranked.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, item in ranked[:top_k]:
        out.append({
            "score": score,
            "url": item.get("page") or item.get("src", ""),
            "src": item.get("src", ""),
            "alt": item.get("alt", ""),
            "caption": item.get("caption", ""),
            "description": item.get("description", ""),
        })
    return out
