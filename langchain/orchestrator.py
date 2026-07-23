"""
Batch web-LLM orchestrator using LangGraph batching with detailed per-query CPU/GPU profiling.
Accepts multiple queries as CLI args, runs the full tool chain in a single batched graph invocation.
Runs in two batching modes ("same" and "different" queries) and generates comprehensive plots.
"""
import os
import sys
import timeit
import argparse
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from typing import List, Optional, TypedDict, Dict, Any

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

# Global timing storage focused ONLY on CPU and GPU
timing_stats = {
    'cpu': {
        'summarize_total': defaultdict(list),
        'parse_doc': defaultdict(list),
        'lexrank_algo': defaultdict(list)
    },
    'gpu': {
        'llm_total': defaultdict(list),
        'ttft': defaultdict(list), # Time To First Token (Prefill)
        'decode_time': defaultdict(list) # Time spent generating tokens
    }
}

# LOAD CACHE
CACHE_FILE = "cached_search_results.json"
SEARCH_CACHE = {}
QUERY_POOL = []

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        SEARCH_CACHE = json.load(f)
        QUERY_POOL = list(SEARCH_CACHE.keys())
    print(f"[INFO] Loaded {len(SEARCH_CACHE)} cached search entries.")
else:
    print(f"[ERROR] {CACHE_FILE} not found! Please run cache_builder.py first.")
    sys.exit(1)


# 1) Shared state schema
class GraphState(TypedDict):
    query: str
    urls: List[str]
    page_texts: List[str]
    summaries: List[str]
    final_response: str
    job_id: int
    skip_web_search: bool

# 2) Tool implementations (Network profiling removed)
def web_search(state: GraphState) -> GraphState:
    query = state["query"]
    if not state["skip_web_search"] and query in SEARCH_CACHE:
        urls = SEARCH_CACHE[query]
    else:
        urls = [
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://www.bbc.com/news"
        ]
    return {"urls": urls}

def _fetch_single(url: str, timeout: float = 10.0) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(separator="\n")
    except requests.RequestException:
        return None

def _fetch_url_single_state(state: GraphState) -> GraphState: 
    texts: List[str] = []
    for url in state["urls"]:
        if len(texts) >= 2:
            break
        txt = _fetch_single(url)
        if txt:
            texts.append(txt)
    return {"page_texts": texts}

def fetch_url(state_or_states):  
    if isinstance(state_or_states, list):
        max_workers = max(min(len(state_or_states), os.cpu_count() or 1), 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_fetch_url_single_state, state_or_states))
        return results
    else:
        return _fetch_url_single_state(state_or_states)

def _lexrank_one_detailed(text: str, max_chars: int = 50000) -> Dict[str, Any]:
    """Run LexRank and return detailed CPU timing breakdown."""
    if len(text) > max_chars:
        text = text[:max_chars]
        
    # Phase 1: Parsing/Tokenization
    t_start = timeit.default_timer()
    doc = PlaintextParser.from_string(text, Tokenizer("english")).document
    t_parse = timeit.default_timer() - t_start
    
    if not doc.sentences:
        return {"summary": "", "parse_time": t_parse, "algo_time": 0.0}
        
    # Phase 2: LexRank Algorithm
    t_algo_start = timeit.default_timer()
    summarizer = LexRankSummarizer()
    sentences = summarizer(doc, sentences_count=1)
    summary = " ".join(str(s) for s in sentences)
    t_algo = timeit.default_timer() - t_algo_start
    
    return {"summary": summary, "parse_time": t_parse, "algo_time": t_algo}

def summarize(state: GraphState) -> GraphState:
    marker = f"summarize_lexrank: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer() 
 
    max_workers = max(min(len(state["page_texts"]), os.cpu_count() or 1), 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        detailed_results = list(pool.map(_lexrank_one_detailed, state["page_texts"]))
 
    elapsed = timeit.default_timer() - start_time
    timing_stats['cpu']['summarize_total'].append(elapsed)
    
    # Store granular CPU stats
    for res in detailed_results:
        timing_stats['cpu']['parse_doc'].append(res["parse_time"])
        timing_stats['cpu']['lexrank_algo'].append(res["algo_time"])
        
    nvtx.pop_range()
    
    sums = [res["summary"] for res in detailed_results]
    return {"summaries": sums}
 
def final_answer(state: GraphState) -> GraphState:
    marker = f"llm_inference_gpt_oss_20b: {state['query'][:30]}"
    nvtx.push_range(marker)
    
    llm = VLLMOpenAI(
        base_url='http://localhost:5000/v1',
        model="openai/gpt-oss-20b",
        openai_api_key='EMPTY',
        streaming=True # Enable streaming for TTFT extraction
    )
    
    prompt = f"Based on these summaries, answer: {state['query']}\n\n" + "\n\n".join(state['summaries'])
    
    t_start = timeit.default_timer()
    ttft = None
    response_chunks = []
    
    # Stream to calculate TTFT (GPU Prefill) and decode times
    for chunk in llm.stream(prompt):
        if ttft is None:
            ttft = timeit.default_timer() - t_start
        response_chunks.append(chunk)
        
    t_end = timeit.default_timer()
    total_time = t_end - t_start
    decode_time = total_time - (ttft if ttft else 0)
    
    timing_stats['gpu']['llm_total'].append(total_time)
    timing_stats['gpu']['ttft'].append(ttft if ttft else 0)
    timing_stats['gpu']['decode_time'].append(decode_time)
    
    nvtx.pop_range()
    
    return {'final_response': "".join(response_chunks)}
 
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

# 4) Batch invocation and Result Logging
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LangChain batch orchestrator for deep CPU/GPU visualization')
    parser.add_argument('--verbose', action='store_true', help='Enable output of per-stage latencies')
    parser.add_argument('--skip-web-search', action='store_true', help='Skip web search stage')
    parser.add_argument('--job-id', type=int, default=1, help="Job id for bash multiprocessing")
    parser.add_argument('--benchmark', default="detailed_hardware_stats", help="Dataset name for file logging")
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_modes = ["same", "different"]
    
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_batch_results = {}

    def calculate_stats(times):
        if not times:
            return {"min": 0, "max": 0, "avg": 0}
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times)
        }
    
    for mode in batch_modes:
        print(f"\n" + "="*50)
        print(f"RUNNING BATCH MODE: {mode.upper()} QUERIES")
        print("="*50)
        
        pool_cursor = 0
        all_batch_results[mode] = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing Mode: {mode} | Batch Size: {batch_size} ---")
            
            if mode == "different":
                queries = QUERY_POOL[pool_cursor : pool_cursor + batch_size]
                pool_cursor += batch_size
                if len(queries) < batch_size:
                    print(f"[WARNING] Ran out of unique queries. Skipping remaining batch sizes for 'different' mode.")
                    break
            else:
                queries = [QUERY_POOL[0]] * batch_size

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
            
            # Track list lengths to isolate this batch's stats
            idx_tracker = {
                'cpu_total': len(timing_stats['cpu']['summarize_total']),
                'cpu_parse': len(timing_stats['cpu']['parse_doc']),
                'cpu_algo': len(timing_stats['cpu']['lexrank_algo']),
                'gpu_total': len(timing_stats['gpu']['llm_total']),
                'gpu_ttft': len(timing_stats['gpu']['ttft']),
                'gpu_decode': len(timing_stats['gpu']['decode_time']),
            }

            nvtx.push_range(f'batch_run_{mode}_{batch_size}')
            start_time = timeit.default_timer()
            
            # Run batch invocation
            result_states = compiled_graph.batch(initial_states, config=cfg)
            
            total_time = timeit.default_timer() - start_time
            nvtx.pop_range()

            # Calculate granular stats for this batch
            b_stats = {
                "cpu_total": calculate_stats(timing_stats['cpu']['summarize_total'][idx_tracker['cpu_total']:]),
                "cpu_parse": calculate_stats(timing_stats['cpu']['parse_doc'][idx_tracker['cpu_parse']:]),
                "cpu_algo": calculate_stats(timing_stats['cpu']['lexrank_algo'][idx_tracker['cpu_algo']:]),
                "gpu_total": calculate_stats(timing_stats['gpu']['llm_total'][idx_tracker['gpu_total']:]),
                "gpu_ttft": calculate_stats(timing_stats['gpu']['ttft'][idx_tracker['gpu_ttft']:]),
                "gpu_decode": calculate_stats(timing_stats['gpu']['decode_time'][idx_tracker['gpu_decode']:]),
            }

            throughput = batch_size / total_time
            
            all_batch_results[mode][batch_size] = {
                "latency_s": total_time,
                "throughput_qps": throughput,
                "hardware_details": b_stats,
                "states": result_states
            }
            
            print(f"Overall Latency: {total_time:.2f}s | Throughput: {throughput:.2f} q/s")
            print(f"  -> CPU Breakdown (Avg): Total: {b_stats['cpu_total']['avg']:.3f}s | Parse: {b_stats['cpu_parse']['avg']:.3f}s | Algo: {b_stats['cpu_algo']['avg']:.3f}s")
            print(f"  -> GPU Breakdown (Avg): Total: {b_stats['gpu_total']['avg']:.3f}s | Prefill (TTFT): {b_stats['gpu_ttft']['avg']:.3f}s | Decode: {b_stats['gpu_decode']['avg']:.3f}s")

    # ==========================================
    # --- Plotting Section ---
    # ==========================================
    styles = {"same": {"color": "#1f77b4", "marker": "o"}, "different": {"color": "#ff7f0e", "marker": "D"}}
    
    # 1) Overall Latency Plot
    plt.figure(figsize=(10, 6))
    for mode in batch_modes:
        if mode in all_batch_results and all_batch_results[mode]:
            b_sizes = list(all_batch_results[mode].keys())
            lats = [all_batch_results[mode][b]["latency_s"] for b in b_sizes]
            plt.plot(b_sizes, lats, marker=styles[mode]["marker"], color=styles[mode]["color"], linewidth=2, markersize=8, label=f'{mode.capitalize()} Queries')
            
    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (s)', fontsize=14, fontweight='bold')
    plt.title('Overall Latency by Batch Mode', fontsize=16, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    plt.legend()
    latency_path = f"./results/latency_comparison_{timestamp}.png"
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Overall Throughput Plot
    plt.figure(figsize=(10, 6))
    for mode in batch_modes:
        if mode in all_batch_results and all_batch_results[mode]:
            b_sizes = list(all_batch_results[mode].keys())
            tps = [all_batch_results[mode][b]["throughput_qps"] for b in b_sizes]
            plt.plot(b_sizes, tps, marker=styles[mode]["marker"], color=styles[mode]["color"], linewidth=2, markersize=8, label=f'{mode.capitalize()} Queries')

    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (queries/sec)', fontsize=14, fontweight='bold')
    plt.title('Throughput by Batch Mode', fontsize=16, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    plt.legend()
    throughput_path = f"./results/throughput_comparison_{timestamp}.png"
    plt.savefig(throughput_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3) CPU Breakdown (Parse vs Algo)
    plt.figure(figsize=(12, 7))
    for mode in batch_modes:
        if mode in all_batch_results and all_batch_results[mode]:
            b_sizes = list(all_batch_results[mode].keys())
            parse_avg = [all_batch_results[mode][b]["hardware_details"]["cpu_parse"]["avg"] for b in b_sizes]
            algo_avg = [all_batch_results[mode][b]["hardware_details"]["cpu_algo"]["avg"] for b in b_sizes]
            
            plt.plot(b_sizes, parse_avg, marker='s', linestyle='--', color=styles[mode]["color"], label=f"Parse Time ({mode})")
            plt.plot(b_sizes, algo_avg, marker='^', linestyle='-', color=styles[mode]["color"], label=f"Algo Time ({mode})")

    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Average Stage Latency (s)', fontsize=14, fontweight='bold')
    plt.title('CPU Breakdown: Parsing vs LexRank Algorithm', fontsize=16, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    plt.legend()
    cpu_path = f"./results/cpu_breakdown_{timestamp}.png"
    plt.savefig(cpu_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4) GPU Breakdown (TTFT vs Decode)
    plt.figure(figsize=(12, 7))
    for mode in batch_modes:
        if mode in all_batch_results and all_batch_results[mode]:
            b_sizes = list(all_batch_results[mode].keys())
            ttft_avg = [all_batch_results[mode][b]["hardware_details"]["gpu_ttft"]["avg"] for b in b_sizes]
            decode_avg = [all_batch_results[mode][b]["hardware_details"]["gpu_decode"]["avg"] for b in b_sizes]
            
            plt.plot(b_sizes, ttft_avg, marker='s', linestyle='--', color=styles[mode]["color"], label=f"Prefill / TTFT ({mode})")
            plt.plot(b_sizes, decode_avg, marker='^', linestyle='-', color=styles[mode]["color"], label=f"Decode Generation ({mode})")

    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Average Stage Latency (s)', fontsize=14, fontweight='bold')
    plt.title('GPU Breakdown: Prefill (TTFT) vs Token Decode', fontsize=16, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    plt.legend()
    gpu_path = f"./results/gpu_breakdown_{timestamp}.png"
    plt.savefig(gpu_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # --- File Output Generation ---
    # ==========================================
    
    # 1. Detailed JSON output
    detailed_file = os.path.join("results", f"{args.benchmark.lower()}_{timestamp}_detailed.json")
    detailed_data = {
        "metadata": {
            "benchmark": args.benchmark.lower(),
            "modes_tested": batch_modes,
            "batch_sizes_tested": batch_sizes
        },
        "timing_stats_raw": timing_stats,
        "batch_results": all_batch_results
    }
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=4)
        
    # 2. Text Summary Output
    summary_file = os.path.join("results", f"{args.benchmark.lower()}_{timestamp}_summary.txt")
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("HARDWARE BENCHMARK SUMMARY")
    summary_lines.append("="*70)

    for mode in batch_modes:
        if mode in all_batch_results and all_batch_results[mode]:
            summary_lines.append(f"\n\n{'='*30}\nMODE: {mode.upper()} QUERIES\n{'='*30}")
            for batch, data in all_batch_results[mode].items():
                summary_lines.append(f"\n--- BATCH SIZE: {batch} ---")
                summary_lines.append(f"• Total Latency:          {data['latency_s']:.2f}s")
                summary_lines.append(f"• Throughput:             {data['throughput_qps']:.2f} queries/s")
                
                hd = data['hardware_details']
                summary_lines.append("\n  [CPU Breakdown - Min / Avg / Max]")
                summary_lines.append(f"  • Doc Parse:      {hd['cpu_parse']['min']:.3f}s / {hd['cpu_parse']['avg']:.3f}s / {hd['cpu_parse']['max']:.3f}s")
                summary_lines.append(f"  • LexRank Algo:   {hd['cpu_algo']['min']:.3f}s / {hd['cpu_algo']['avg']:.3f}s / {hd['cpu_algo']['max']:.3f}s")
                
                summary_lines.append("\n  [GPU Breakdown - Min / Avg / Max]")
                summary_lines.append(f"  • Prefill (TTFT): {hd['gpu_ttft']['min']:.3f}s / {hd['gpu_ttft']['avg']:.3f}s / {hd['gpu_ttft']['max']:.3f}s")
                summary_lines.append(f"  • Decode (Gen):   {hd['gpu_decode']['min']:.3f}s / {hd['gpu_decode']['avg']:.3f}s / {hd['gpu_decode']['max']:.3f}s")
                
                summary_lines.append("\n  [Responses Sample]")
                # Just show the first response to keep summary manageable
                if data['states']:
                    state = data['states'][0]
                    summary_lines.append(f"  🧑 » {state['query']}")
                    summary_lines.append(f"  🤖 » {state['final_response'][:200]}...")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print("\n" + "="*50)
    print("✅ BENCHMARK COMPLETE")
    print("="*50)
    print(f"📈 Latency Plot:    {latency_path}")
    print(f"📈 Throughput Plot: {throughput_path}")
    print(f"📈 CPU Plot:        {cpu_path}")
    print(f"📈 GPU Plot:        {gpu_path}")
    print(f"📁 Detailed JSON:   {detailed_file}")
    print(f"📝 Text Summary:    {summary_file}")