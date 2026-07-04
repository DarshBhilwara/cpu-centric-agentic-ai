"""
Batch web-LLM orchestrator using LangGraph batching with per-query NVTX markers.
Accepts multiple queries as CLI args, runs the full tool chain in a single batched graph invocation, and marks each node per query.
"""
import os
import sys
import time
import timeit
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from typing import List, Optional, TypedDict

import nvtx
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from langgraph.graph import StateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_community.llms import VLLMOpenAI
import matplotlib.pyplot as plt

# Global timing storage for statistics categorized by hardware/resource type
timing_stats = {
    'network': defaultdict(list),
    'cpu': defaultdict(list),
    'gpu': defaultdict(list)
}

# 1) Shared state schema
class GraphState(TypedDict):
    query: str
    urls: List[str]
    page_texts: List[str]
    summaries: List[str]
    final_response: str
    job_id: int
    skip_web_search: bool

# 2) Tool implementations
def web_search(state: GraphState) -> GraphState:
    marker = f"web_search: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()

    if not state["skip_web_search"]:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            nvtx.pop_range()
            raise RuntimeError("Missing SERPER_API_KEY")

        params = {
            "q": state["query"],
            "apiKey": api_key,
            "num": 10
        }

        resp = requests.get(
            "https://google.serper.dev/search",
            params=params,
            timeout=10
        )
        resp.raise_for_status()

        data = resp.json()
        items = data.get("organic", [])
        urls = [item["link"] for item in items if "link" in item]

    else:
        # Fallback list for offline testing
        urls = [
            "https://en.wikipedia.org/wiki/Spiel_des_Jahres",
            "https://boardgamegeek.com/wiki/page/Spiel_des_Jahres",
            "https://www.reddit.com/r/boardgames/comments/buwap5/are_previous_spiel_des_jahres_winners_now_too/",
            "https://boardgamegeek.com/thread/3282083/spiel-des-jahres-winners-1979-to-2023-and-who-do-y",
            "https://blog.recommend.games/posts/thoughts-on-spiel-des-jahres/",
            "https://www.spiel-des-jahres.de/en/award-winners-2024/",
            "https://www.facebook.com/groups/132851767828/posts/10162746926537829/",
            "https://www.tabletopgaming.co.uk/news/spiel-des-jahres-2024-winners-announced/",
            "https://therewillbe.games/board-game-lists-and-guides/6214-the-ten-greatest-spiel-des-jahres-winners",
            "https://www.dicebreaker.com/topics/spiel-des-jahres/best-games/overlooked-spiel-des-jahres-winners",
        ]

    elapsed = timeit.default_timer() - start_time
    timing_stats['network']["web_search"].append(elapsed)
    nvtx.pop_range()
    return {"urls": urls}


def _fetch_single(url: str, timeout: float = 10.0) -> Optional[str]:
    """Download one URL and return plain-text, or None on error."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except requests.RequestException:
        return None


def _fetch_url_single_state(state: GraphState) -> GraphState: 
    """Sequential download of up to two pages for *one* query."""
    texts: List[str] = []
    for url in state["urls"]:
        if len(texts) >= 2:
            break
        txt = _fetch_single(url)
        if txt:
            texts.append(txt)
    return {"page_texts": texts}


def fetch_url(state_or_states):  
    """Batched wrapper: handles either a single state or a list of states."""
    start_time = timeit.default_timer()
    
    if isinstance(state_or_states, list):
        marker = f"fetch_url_batch: {len(state_or_states)} queries"
        nvtx.push_range(marker)
        
        max_workers = max(min(len(state_or_states), os.cpu_count() or 1), 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_fetch_url_single_state, state_or_states))
            
        elapsed = timeit.default_timer() - start_time
        timing_stats['network']['fetch_url'].append(elapsed)
        nvtx.pop_range()
        return results
    else:
        marker = f"fetch_url: {state_or_states['query'][:30]}"
        nvtx.push_range(marker)
        
        result = _fetch_url_single_state(state_or_states)
        
        elapsed = timeit.default_timer() - start_time
        timing_stats['network']['fetch_url'].append(elapsed)
        nvtx.pop_range()
        return result


def _lexrank_one(text: str) -> str:
    """Run LexRank on a single document and return one-sentence summary."""
    summarizer = LexRankSummarizer()
    doc = PlaintextParser.from_string(text, Tokenizer("english")).document
    sentences = summarizer(doc, sentences_count=1)
    return " ".join(str(s) for s in sentences)


def summarize(state: GraphState) -> GraphState:
    marker = f"summarize_lexrank: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer() 
 
    max_workers = max(min(len(state["page_texts"]), os.cpu_count() or 1), 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        sums = list(pool.map(_lexrank_one, state["page_texts"]))
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['cpu']['summarize_lexrank'].append(elapsed)
    nvtx.pop_range()
    
    return {"summaries": sums}

 
def final_answer(state: GraphState) -> GraphState:
    marker = f"llm_inference_gpt_oss_20b: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()
    
    llm = VLLMOpenAI(
        base_url='http://localhost:5000/v1',
        model="openai/gpt-oss-20b",
        openai_api_key='EMPTY'
    )
    
    prompt = f"Based on these summaries, answer: {state['query']}\n\n" + "\n\n".join(state['summaries'])
    answer = llm.invoke([prompt])
    
    elapsed = timeit.default_timer() - start_time
    timing_stats['gpu']['llm_inference_gpt_oss_20b'].append(elapsed)
    nvtx.pop_range()
    
    return {'final_response': answer}
 
 
def get_timing_statistics_str() -> str:
    """Generate a formatted string of average, min, and max time for each stage."""
    output = []
    output.append("\n" + "="*70)
    output.append("TIMING STATISTICS (across all batches)")
    output.append("="*70)
    
    def format_category(category_name, stats_dict):
        output.append(f"\n[{category_name.upper()} BOUND TASKS]")
        output.append(f"{'Stage':<30} {'Count':<10} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10}")
        output.append("-" * 75)
        
        if not stats_dict:
            output.append("No tasks recorded in this category.")
            return

        for stage, times in stats_dict.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                count = len(times)
                output.append(f"{stage:<30} {count:<10} {avg_time:<10.4f} {min_time:<10.4f} {max_time:<10.4f}")
            else:
                output.append(f"{stage:<30} {'0':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    format_category('network', timing_stats['network'])
    format_category('cpu', timing_stats['cpu'])
    format_category('gpu', timing_stats['gpu'])
    output.append("\n" + "="*70 + "\n")
    
    return "\n".join(output)
 
 
# 3) Build and compile the graph
builder = StateGraph(GraphState)
builder.set_entry_point('web_search')
builder.add_node('web_search', web_search)
builder.add_node('fetch_url', fetch_url)
builder.add_node('summarize_lexrank', summarize)
builder.add_node('llm_inference_gpt_oss_20b', final_answer)

builder.add_edge('web_search', 'fetch_url')
builder.add_edge('fetch_url', 'summarize_lexrank')
builder.add_edge('summarize_lexrank', 'llm_inference_gpt_oss_20b')
builder.set_finish_point('llm_inference_gpt_oss_20b')

compiled_graph = builder.compile()
 

# 4) Dataset Loader
def load_random_questions(benchmark: str, num_tests: int) -> List[str]:
    """Loads a random sample of questions from the specified JSONL dataset."""
    benchmark = benchmark.lower()
    file_path = Path(f"datasets/{benchmark}.jsonl")
    
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
                
                # If it's a multiple choice dataset like QASC, append the choices
                if "choices" in data and isinstance(data["choices"], list):
                    choices_str = " ".join([f"({c.get('label', '')}) {c.get('text', '')}" for c in data["choices"]])
                    q_text = f"{q_text} {choices_str}"
                    
                questions.append(q_text)
                
    if not questions:
        raise ValueError(f"No valid questions found in {file_path}")
        
    if num_tests > len(questions):
        print(f"[WARNING] Requested {num_tests} tests, but only {len(questions)} are available. Using all.")
        num_tests = len(questions)

    # Shuffle and return the requested number of tests
    return random.sample(questions, num_tests)


# 5) Batch invocation and Result Logging
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LangChain batch orchestrator for latency/throughput visualization')
    parser.add_argument('--verbose', action='store_true', help='Enable output of per-stage latencies')
    parser.add_argument('--skip-web-search', action='store_true', help='Skip web search stage')
    parser.add_argument('--job-id', type=int, default=1, help="Job id for bash multiprocessing")
    parser.add_argument('--benchmark', choices=["freshQA", "freshqa", "QASC", "qasc"], default="freshqa", help="Dataset to load")
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    latencies = []
    throughputs = []
    all_batch_results = {}
    
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"[INFO] Running benchmarks for dataset '{args.benchmark}' on batch sizes: {batch_sizes}")
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        try:
            # We use `batch_size` as our `num_tests` parameter so the graph receives a list of exactly `batch_size` queries
            queries = load_random_questions(args.benchmark, batch_size)
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            sys.exit(1)

        initial_states = [
            {
                'query': q,
                'urls': [],
                'page_texts': [],
                'summaries': [],
                'final_response': '',
                'job_id': args.job_id,
                'skip_web_search': args.skip_web_search
            }
            for q in queries
        ]

        cfg = RunnableConfig(max_concurrency=batch_size)
        
        nvtx.push_range(f'batch_run_{batch_size}')
        start_time = timeit.default_timer()
        
        # Run batch invocation
        result_states = compiled_graph.batch(initial_states, config=cfg)
        
        end_time = timeit.default_timer()
        total_time = end_time - start_time
        nvtx.pop_range()
        
        throughput = batch_size / total_time
        latencies.append(total_time)
        throughputs.append(throughput)
        
        # Store results for this batch
        all_batch_results[batch_size] = {
            "latency_s": total_time,
            "throughput_qps": throughput,
            "states": result_states
        }
        
        print(f"Batch {batch_size} Latency: {total_time:.2f}s | Throughput: {throughput:.2f} queries/s")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, latencies, marker='D', color='#96CEB4', linewidth=3, markersize=8)
    plt.xlabel('Batch Size', fontsize=16, fontweight='bold')
    plt.ylabel('Latency (s)', fontsize=16, fontweight='bold')
    plt.title('LangChain Orchestrator Latency vs Batch Size', fontsize=18, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    latency_path = f"./results/langchain_latency_{timestamp}.png"
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, throughputs, marker='D', color='#96CEB4', linewidth=3, markersize=8)
    plt.xlabel('Batch Size', fontsize=16, fontweight='bold')
    plt.ylabel('Throughput (queries/sec)', fontsize=16, fontweight='bold')
    plt.title('LangChain Orchestrator Throughput vs Batch Size', fontsize=18, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    throughput_path = f"./results/langchain_throughput_{timestamp}.png"
    plt.savefig(throughput_path, dpi=300, bbox_inches='tight')
    plt.close()


    detailed_file = os.path.join("results", f"{args.benchmark.lower()}_{timestamp}.json")
    detailed_data = {
        "metadata": {
            "benchmark": args.benchmark.lower(),
            "batch_sizes_tested": batch_sizes,
            "skip_web_search": args.skip_web_search,
            "job_id": args.job_id
        },
        "timing_stats_raw": timing_stats,
        "batch_results": all_batch_results
    }
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=4)
        
    summary_text_blocks = []
    
    if args.verbose:
        stats_str = get_timing_statistics_str()
        print(stats_str)
        summary_text_blocks.append(stats_str)
        
    summary_text_blocks.append("\n" + "="*70 + "\nRESULTS PREVIEW\n" + "="*70)
    
    # We'll just print a preview of the last (largest) batch state to avoid terminal flood, 
    # but write everything to the text file.
    for batch, data in all_batch_results.items():
        summary_text_blocks.append(f"\n--- BATCH SIZE: {batch} ---")
        for state in data['states']:
            res_str = f"🧑 » {state['query']}\n🤖 » {state['final_response']}\n"
            summary_text_blocks.append(res_str)

    summary_file = os.path.join("results", f"{args.benchmark.lower()}_{timestamp}_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_text_blocks))
        
    print(f"\n✅ Saved latency plot to {latency_path}")
    print(f"✅ Saved throughput plot to {throughput_path}")
    print(f"✅ Detailed JSON results saved to: {detailed_file}")
    print(f"✅ Complete text summary saved to: {summary_file}")