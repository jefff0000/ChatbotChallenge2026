"""Lightweight graph expansion for retrieved chunks."""
import json
from functools import lru_cache
from pathlib import Path

GRAPH_PATH = Path("data/graphrag.json")


@lru_cache(maxsize=1)
def _graph() -> dict:
    if not GRAPH_PATH.exists():
        return {"nodes": {}, "neighbors": {}}
    return json.loads(GRAPH_PATH.read_text())


def expand_chunks(chunks: list[dict], max_extra: int = 3) -> list[dict]:
    """Add graph neighbors for retrieved chunks."""
    graph = _graph()
    nodes = graph.get("nodes", {})
    neighbors = graph.get("neighbors", {})

    out = list(chunks)
    seen = {c.get("metadata", {}).get("cid") for c in chunks}
    seen.discard(None)

    added = 0
    for chunk in chunks:
        cid = chunk.get("metadata", {}).get("cid")
        if not cid:
            continue
        for nid in neighbors.get(cid, []):
            if nid in seen:
                continue
            node = nodes.get(nid)
            if not node:
                continue
            out.append({
                "text": node.get("text", ""),
                "metadata": node.get("metadata", {}),
                "distance": chunk.get("distance", 1.0) + 0.05,
            })
            seen.add(nid)
            added += 1
            if added >= max_extra:
                return out
    return out
