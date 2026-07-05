import os
import json
import random
import time
import argparse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def load_all_dataset_questions(benchmark: str) -> list:
    """Loads all valid questions from the specified JSONL dataset."""
    file_path = Path(f"datasets/{benchmark.lower()}.jsonl")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            if "question" in data:
                q_text = data["question"]
                
                if "choices" in data and isinstance(data["choices"], list):
                    choices_str = " ".join([f"({c.get('label', '')}) {c.get('text', '')}" for c in data["choices"]])
                    q_text = f"{q_text} {choices_str}"
                    
                questions.append(q_text)
                
    if not questions:
        raise ValueError(f"No valid questions found in {file_path}")
    return questions

def build_cache(benchmark: str, total_needed: int, output_file: str):
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("[ERROR] Missing SERPER_API_KEY environment variable.")
        return

    # 1. Load and Pool Queries
    all_questions = load_all_dataset_questions(benchmark)
    if total_needed > len(all_questions):
        print(f"[WARNING] Requested {total_needed} queries, but dataset only has {len(all_questions)}. Wrapping.")
        query_pool = random.choices(all_questions, k=total_needed)
    else:
        query_pool = random.sample(all_questions, total_needed)

    # 2. Setup robust requests session to prevent 429 Too Many Requests
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    cache = {}
    print(f"[INFO] Fetching search results for {len(query_pool)} unique queries...")

    # 3. Fetch Data
    for i, q in enumerate(query_pool):
        print(f"[{i+1}/{len(query_pool)}] Fetching: {q[:50]}...")
        
        # Skip if somehow duplicated in our pool (rare, but safe)
        if q in cache:
            continue
            
        try:
            params = {"q": q, "apiKey": api_key, "num": 10}
            resp = session.get("https://google.serper.dev/search", params=params, timeout=10, verify=False)
            resp.raise_for_status()
            
            data = resp.json()
            items = data.get("organic", [])
            urls = [item["link"] for item in items if "link" in item]
            
            cache[q] = urls
            
            # Small sleep to be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch for '{q}': {e}")
            cache[q] = [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://www.bbc.com/news"
            ]

    # 4. Save Cache
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)
    print(f"\n✅ Done! Saved {len(cache)} results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch web searches to isolate CPU/GPU benchmarking")
    parser.add_argument('--benchmark', choices=["freshQA", "freshqa", "QASC", "qasc"], default="freshqa")
    parser.add_argument('--queries', type=int, default=255, help="Total unique queries to cache (1+2+4+8+16+32+64+128 = 255)")
    parser.add_argument('--output', type=str, default="cached_search_results.json")
    args = parser.parse_args()
    
    # Supress insecure request warnings if you are bypassing SSL
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    build_cache(args.benchmark, args.queries, args.output)