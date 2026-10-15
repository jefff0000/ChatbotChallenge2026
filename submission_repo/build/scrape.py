"""Scraper. STUB. Workshop 1 block 4.

Crawl both websites, extract the readable text, and collect image
records in the same pass. Run this file directly to (re)build
data/pages.json and data/images.json.

Check for /sitemap.xml before writing a crawler. If it exists it lists
every page and you can skip the crawl entirely.
"""
import json
from collections import deque
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

SITES = [
    "https://innoacademy.engg.hku.hk/",
]


def _normalise(url: str) -> str:
    p = urlparse(url)
    clean = p._replace(fragment="", query="")
    u = urlunparse(clean)
    return u[:-1] if u.endswith("/") else u


def _allowed_host(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    return hostname == "engg.hku.hk" or hostname.endswith(".engg.hku.hk")


def _bad_page(html: str) -> bool:
    text = html.lower()
    return "wordpress error" in text or "page not found" in text


def _fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    ctype = r.headers.get("content-type", "").lower()
    if "text/html" not in ctype:
        return None
    if _bad_page(r.text):
        return None
    return r.text


def crawl(start_url: str, max_pages: int = 3000) -> list[str]:
    """Return every valid page URL under *.engg.hku.hk."""
    seen, out = set(), []
    queue = deque([_normalise(start_url)])

    while queue and len(out) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        if not _allowed_host(urlparse(url).hostname):
            continue
        html = _fetch_html(url)
        if not html:
            continue

        out.append(url)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            link = _normalise(urljoin(url, a["href"]))
            if urlparse(link).scheme not in ("http", "https"):
                continue
            if not _allowed_host(urlparse(link).hostname):
                continue
            if link not in seen:
                queue.append(link)

    return out


def extract(html: str, url: str) -> dict:
    """Return {"url", "title", "text", "images": [...]} for one page.

    soup.get_text() on the whole page returns the navigation menu and
    footer on every page. Those near-identical fragments become chunks
    that look moderately similar to every query and crowd real results
    out of your top five. Open the site, right-click the content, choose
    Inspect, and find the element that actually wraps it.
    """
    soup = BeautifulSoup(html, "html.parser")

    # TODO replace this with the element that holds the content, e.g.
    # TODO soup.select_one("main") or soup.select_one("#content")
    body = soup

    images = []
    for img in soup.select("img"):
        src = img.get("src")
        if not src:
            continue
        fig = img.find_parent("figure")
        images.append({
            "src":     urljoin(url, src),     # relative -> absolute
            "alt":     img.get("alt", ""),
            "caption": (fig.find("figcaption").get_text(strip=True)
                        if fig and fig.find("figcaption") else ""),
            "page":    url,
        })

    return {
        "url":    url,
        "title":  soup.title.get_text(strip=True) if soup.title else "",
        "text":   body.get_text(" ", strip=True),
        "images": images,
    }


if __name__ == "__main__":
    pages = []
    for site in SITES:
        for url in crawl(site):
            try:
                html = _fetch_html(url)
                if not html:
                    continue
                pages.append(extract(html, url))
            except Exception as exc:
                print("skipped", url, exc)

    Path("data").mkdir(exist_ok=True)
    Path("data/pages.json").write_text(json.dumps(pages, indent=1))

    images = [im for p in pages for im in p["images"]]
    Path("data/images.json").write_text(json.dumps(images, indent=1))

    print(f"{len(pages)} pages, {len(images)} images")
