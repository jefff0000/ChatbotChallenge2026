"""Scraper. STUB. Workshop 1 block 4.

Crawl both websites, extract the readable text, and collect image
records in the same pass. Run this file directly to (re)build
data/pages.json and data/images.json.

Check for /sitemap.xml before writing a crawler. If it exists it lists
every page and you can skip the crawl entirely.
"""
import argparse
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
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True).lower() if soup.title else "")
    heading = " ".join(h.get_text(" ", strip=True).lower() for h in soup.select("h1, h2")[:2])
    checks = [title, heading]
    needles = ("wordpress error", "page not found", "404 not found")
    return any(any(n in c for n in needles) for c in checks if c)


def _fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 ChatbotChallengeCrawler/1.0"})
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
        print(f"[crawl] {len(out):4d}/{max_pages} accepted | queue={len(queue):4d} | {url}")
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


def _args():
    parser = argparse.ArgumentParser(description="Crawl engg.hku.hk pages and extract text/images.")
    parser.add_argument(
        "max_pages_positional",
        nargs="?",
        type=int,
        help="Optional positional max pages (e.g. `python build/scrape.py 500`).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3000,
        help="Maximum accepted pages to crawl per seed site (e.g. 500).",
    )
    args = parser.parse_args()
    if args.max_pages_positional is not None:
        args.max_pages = args.max_pages_positional
    return args


if __name__ == "__main__":
    args = _args()
    pages = []
    failures = 0
    for site in SITES:
        print(f"[crawl] seed: {site} (max_pages={args.max_pages})")
        urls = crawl(site, max_pages=args.max_pages)
        total = len(urls)
        print(f"[crawl] accepted {total} pages from seed: {site}")
        for i, url in enumerate(urls, start=1):
            try:
                html = _fetch_html(url)
                if not html:
                    failures += 1
                    print(f"[extract] {i:4d}/{total} failed fetch | {url}")
                    continue
                pages.append(extract(html, url))
                print(f"[extract] {i:4d}/{total} done | pages={len(pages):4d} | {url}")
            except Exception as exc:
                print("skipped", url, exc)
                failures += 1

    Path("data").mkdir(exist_ok=True)
    Path("data/pages.json").write_text(json.dumps(pages, indent=1))

    images = [im for p in pages for im in p["images"]]
    Path("data/images.json").write_text(json.dumps(images, indent=1))

    print(f"{len(pages)} pages, {len(images)} images")
    print(f"failed fetch/extract: {failures}")
    if not pages:
        print(
            "No pages were collected. Check DNS/network access, then verify that "
            "the start URL is reachable and returns HTTP 200 HTML."
        )
