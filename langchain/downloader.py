"""
Downloads and parses text from URLs in cached_search_results.json
and saves the resulting text into cached_pages.json for zero-network benchmarking.
"""
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_CACHE = "cached_search_results.json"
OUTPUT_CACHE = "cached_pages.json"
MAX_WORKERS = 32
MAX_PAGES_PER_QUERY = 2

def fetch_and_parse_url(url: str, timeout: float = 10.0) -> str:
    """Fetch URL and extract clean text using BeautifulSoup."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except Exception:
        return None

def process_query_entry(query: str, urls: list) -> tuple:
    """Fetch up to MAX_PAGES_PER_QUERY valid page texts for a single query."""
    texts = []
    for url in urls:
        if len(texts) >= MAX_PAGES_PER_QUERY:
            break
        text = fetch_and_parse_url(url)
        if text and text.strip():
            texts.append(text)
    return query, texts

def main():
    if not os.path.exists(INPUT_CACHE):
        print(f"[ERROR] Could not find {INPUT_CACHE}")
        sys.exit(1)

    with open(INPUT_CACHE, "r", encoding="utf-8") as f:
        search_cache = json.load(f)

    print(f"[INFO] Loaded {len(search_cache)} queries from {INPUT_CACHE}")
    print(f"[INFO] Fetching and parsing web pages using {MAX_WORKERS} workers...")

    cached_pages = {}
    total_queries = len(search_cache)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_query = {
            pool.submit(process_query_entry, query, urls): query 
            for query, urls in search_cache.items()
        }

        for future in as_completed(future_to_query):
            query, page_texts = future.result()
            cached_pages[query] = page_texts
            completed += 1
            if completed % 10 == 0 or completed == total_queries:
                print(f"[PROGRESS] {completed}/{total_queries} queries processed...")

    with open(OUTPUT_CACHE, "w", encoding="utf-8") as f:
        json.dump(cached_pages, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Saved cached page texts to {OUTPUT_CACHE}")

if __name__ == '__main__':
    main()