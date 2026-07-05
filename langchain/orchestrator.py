"""
Batch web-LLM orchestrator using LangGraph batching with per-query NVTX markers.
Accepts multiple queries as CLI args, runs the full tool chain in a single batched graph invocation, and marks each node per query.
"""
import os
import sys
import timeit
import argparse
import json
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

# LOAD CACHE (Replaces live dataset and web search parsing)
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

# 2) Tool implementations
def web_search(state: GraphState) -> GraphState:
    marker = f"web_search: {state['query'][:30]}"
    nvtx.push_range(marker)
    start_time = timeit.default_timer()

    query = state["query"]
    
    # Fast local dictionary lookup
    if not state["skip_web_search"] and query in SEARCH_CACHE:
        urls = SEARCH_CACHE[query]
    else:
        urls = [
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://www.bbc.com/news"
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
 

# 4) Batch invocation and Result Logging
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LangChain batch orchestrator for latency/throughput visualization')
    parser.add_argument('--verbose', action='store_true', help='Enable output of per-stage latencies')
    parser.add_argument('--skip-web-search', action='store_true', help='Skip web search stage')
    parser.add_argument('--job-id', type=int, default=1, help="Job id for bash multiprocessing")
    parser.add_argument('--benchmark', default="cached_dataset", help="Dataset name for file logging")
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    total_needed_queries = sum(batch_sizes) 
    
    if total_needed_queries > len(QUERY_POOL):
        print(f"[ERROR] Need {total_needed_queries} queries, but cache only has {len(QUERY_POOL)}. Re-run cache_builder.py")
        sys.exit(1)

    latencies = []
    throughputs = []
    
    # Lists to store the 3 core stages for plotting (Min, Max, Avg)
    stage_stats_network = {"min": [], "max": [], "avg": []}
    stage_stats_summarize = {"min": [], "max": [], "avg": []}
    stage_stats_llm = {"min": [], "max": [], "avg": []}
    
    all_batch_results = {}
    
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pool_cursor = 0
    
    def calculate_stats(times):
        if not times:
            return {"min": 0, "max": 0, "avg": 0, "total": 0}
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "total": sum(times)
        }
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing Batch Size: {batch_size} ---")
        
        # Slice out completely unique queries for this batch size
        queries = QUERY_POOL[pool_cursor : pool_cursor + batch_size]
        pool_cursor += batch_size

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
        
        # Track lengths of ALL timing lists before batch run
        ws_idx = len(timing_stats['network']['web_search'])
        fetch_idx = len(timing_stats['network']['fetch_url'])
        sum_idx = len(timing_stats['cpu']['summarize_lexrank'])
        llm_idx = len(timing_stats['gpu']['llm_inference_gpt_oss_20b'])

        nvtx.push_range(f'batch_run_{batch_size}')
        start_time = timeit.default_timer()
        
        # Run batch invocation
        result_states = compiled_graph.batch(initial_states, config=cfg)
        
        end_time = timeit.default_timer()
        total_time = end_time - start_time
        nvtx.pop_range()

        # Extract latencies for just this batch run
        batch_ws_times = timing_stats['network']['web_search'][ws_idx:]
        batch_fetch_times = timing_stats['network']['fetch_url'][fetch_idx:]
        batch_sum_times = timing_stats['cpu']['summarize_lexrank'][sum_idx:]
        batch_llm_times = timing_stats['gpu']['llm_inference_gpt_oss_20b'][llm_idx:]
        
        # Calculate deep statistics
        ws_stats = calculate_stats(batch_ws_times)
        fetch_stats = calculate_stats(batch_fetch_times)
        sum_stats = calculate_stats(batch_sum_times)
        llm_stats = calculate_stats(batch_llm_times)
        
        # Combined network approximations per query
        network_min = ws_stats["min"] + fetch_stats["min"]
        network_max = ws_stats["max"] + fetch_stats["max"]
        network_avg = ws_stats["avg"] + fetch_stats["avg"]

        throughput = batch_size / total_time
        latencies.append(total_time)
        throughputs.append(throughput)
        
        # Store for detailed plotting
        stage_stats_network["min"].append(network_min)
        stage_stats_network["max"].append(network_max)
        stage_stats_network["avg"].append(network_avg)
        
        stage_stats_summarize["min"].append(sum_stats["min"])
        stage_stats_summarize["max"].append(sum_stats["max"])
        stage_stats_summarize["avg"].append(sum_stats["avg"])
        
        stage_stats_llm["min"].append(llm_stats["min"])
        stage_stats_llm["max"].append(llm_stats["max"])
        stage_stats_llm["avg"].append(llm_stats["avg"])
        
        # Store comprehensive results for this batch
        all_batch_results[batch_size] = {
            "latency_s": total_time,
            "throughput_qps": throughput,
            "details": {
                "network": {"min": network_min, "max": network_max, "avg": network_avg},
                "summarization": sum_stats,
                "llm_inference": llm_stats,
            },
            "states": result_states
        }
        
        # Calculate accounted time vs overhead using max (since parallel tasks bottleneck at the max)
        accounted_time = network_max + sum_stats["max"] + llm_stats["max"]
        overhead = total_time - accounted_time

        print(f"Batch {batch_size} Overall Latency: {total_time:.2f}s | Throughput: {throughput:.2f} q/s")
        print(f"  -> Pipeline Breakdown (Min / Avg / Max):")
        print(f"     1. Network:       {network_min:.2f}s / {network_avg:.2f}s / {network_max:.2f}s")
        print(f"     2. Summarization: {sum_stats['min']:.2f}s / {sum_stats['avg']:.2f}s / {sum_stats['max']:.2f}s")
        print(f"     3. LLM Inference: {llm_stats['min']:.2f}s / {llm_stats['avg']:.2f}s / {llm_stats['max']:.2f}s")
        print(f"     *. Graph Overhead:{overhead:.2f}s")

    # --- Plotting section ---

    # 1) Overall Latency Plot
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, latencies, marker='D', color='#96CEB4', linewidth=3, markersize=8)
    plt.xlabel('Batch Size', fontsize=16, fontweight='bold')
    plt.ylabel('Latency (s)', fontsize=16, fontweight='bold')
    plt.title('LangChain Orchestrator Overall Latency', fontsize=18, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.grid(True, alpha=0.3)
    latency_path = f"./results/langchain_latency_{timestamp}.png"
    plt.savefig(latency_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Overall Throughput Plot
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

    # 3) Component Stage Latency Plot (Network vs Summarization vs LLM) with Min/Max Bands
    plt.figure(figsize=(12, 7))
    
    # Summarization Plot (Avg line + Min/Max fill)
    plt.plot(batch_sizes, stage_stats_summarize["avg"], marker='o', color='#FF9999', linewidth=3, markersize=8, label='Summarization Avg (CPU)')
    plt.fill_between(batch_sizes, stage_stats_summarize["min"], stage_stats_summarize["max"], color='#FF9999', alpha=0.2, label='Summarization Min/Max')
    
    # LLM Plot (Avg line + Min/Max fill)
    plt.plot(batch_sizes, stage_stats_llm["avg"], marker='s', color='#66B2FF', linewidth=3, markersize=8, label='LLM Inference Avg (GPU)')
    plt.fill_between(batch_sizes, stage_stats_llm["min"], stage_stats_llm["max"], color='#66B2FF', alpha=0.2, label='LLM Min/Max')

    plt.xlabel('Batch Size', fontsize=16, fontweight='bold')
    plt.ylabel('Stage Latency (s)', fontsize=16, fontweight='bold')
    plt.title('Pipeline Stage Latencies (Avg with Min/Max Bands)', fontsize=18, fontweight='bold')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(x) for x in batch_sizes])
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    stage_latency_path = f"./results/stage_latency_{timestamp}.png"
    plt.savefig(stage_latency_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Output Logging ---
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
        
    summary_text_blocks.append("\n" + "="*70 + "\nDETAILED RESULTS PREVIEW\n" + "="*70)

    for batch, data in all_batch_results.items():
        summary_text_blocks.append(f"\n--- BATCH SIZE: {batch} ---")
        summary_text_blocks.append(f"• Total Latency:          {data['latency_s']:.2f}s")
        summary_text_blocks.append(f"• Throughput:             {data['throughput_qps']:.2f} queries/s")
        
        d = data['details']
        summary_text_blocks.append("\n  [Stage Breakdown - Min / Avg / Max]")
        summary_text_blocks.append(f"  • Network:       {d['network']['min']:.3f}s / {d['network']['avg']:.3f}s / {d['network']['max']:.3f}s")
        summary_text_blocks.append(f"  • Summarization: {d['summarization']['min']:.3f}s / {d['summarization']['avg']:.3f}s / {d['summarization']['max']:.3f}s")
        summary_text_blocks.append(f"  • LLM Inference: {d['llm_inference']['min']:.3f}s / {d['llm_inference']['avg']:.3f}s / {d['llm_inference']['max']:.3f}s\n")
        
        for state in data['states']:
            res_str = f"🧑 » {state['query']}\n🤖 » {state['final_response']}\n"
            summary_text_blocks.append(res_str)

    summary_file = os.path.join("results", f"{args.benchmark.lower()}_{timestamp}_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_text_blocks))
        
    print(f"\n✅ Saved overall latency plot to {latency_path}")
    print(f"✅ Saved overall throughput plot to {throughput_path}")
    print(f"✅ Saved component stage latency plot (with Min/Max bands) to {stage_latency_path}")
    print(f"✅ Detailed JSON results saved to: {detailed_file}")
    print(f"✅ Complete text summary saved to: {summary_file}")