"""
LLM Inference Throughput Benchmarking Script
 
This script benchmarks LLM inference throughput (requests/second) vs batch size
for different combinations of input and output tokens using vLLM server.
 
Configuration:
- Batch sizes: [1, 2, 4, 8, 16, 32, 64, 128]
- Input tokens: [500, 1000, 1500, 2000]
- Output tokens: [500, 1000, 1500, 2000]
- Total runs: 4 * 4 * 8 = 128 combinations
"""
 
import json
import time
import requests
import asyncio
import aiohttp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
import logging
from pathlib import Path
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def thousands_formatter(x, pos):
    if x >= 1000:
        return f'{int(x/1000)}k'
    else:
        return f'{int(x)}'
 
class LLMBenchmark:
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.results = []
        
    def generate_prompt(self, target_tokens: int) -> str:
        """Generate a prompt with approximately target_tokens length."""
        # Approximate 4 characters per token
        base_text = "Write a detailed analysis about artificial intelligence and machine learning technologies. "
        repeat_count = max(1, target_tokens * 4 // len(base_text))
        return base_text * repeat_count
    
    async def make_request(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int) -> Tuple[bool, float]:
        """Make a single request to the vLLM server."""
        payload = {
            "model": "openai/gpt-oss-20b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{self.server_url}/v1/completions",
                                  json=payload,
                                  timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    await response.json()
                    end_time = time.time()
                    return True, end_time - start_time
                else:
                    logger.error(f"Request failed with status {response.status}")
                    return False, 0
        except Exception as e:
            logger.error(f"Request failed with error: {e}")
            return False, 0
    
    async def benchmark_batch(self, batch_size: int, input_tokens: int, output_tokens: int) -> Dict:
        """Benchmark a specific combination of batch_size, input_tokens, and output_tokens."""
        logger.info(f"Benchmarking: batch_size={batch_size}, input_tokens={input_tokens}, output_tokens={output_tokens}")
        
        prompt = self.generate_prompt(input_tokens)
        
        # Single benchmark run
        logger.info(f"Running single benchmark batch...")
        
        async with aiohttp.ClientSession() as session:
            # Create batch of requests
            batch_start = time.time()
            tasks = []
            for _ in range(batch_size):
                task = self.make_request(session, prompt, output_tokens)
                tasks.append(task)
            
            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_end = time.time()
            
            # Check if all requests in batch succeeded
            batch_successful = all(
                isinstance(result, tuple) and result[0]
                for result in results if not isinstance(result, Exception)
            )
            
            if not batch_successful:
                logger.error("Batch failed!")
                return {
                    'batch_size': batch_size,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'throughput_tokens_per_sec': 0,
                    'batch_time': 0,
                    'successful': False
                }
            
            batch_time = batch_end - batch_start
            throughput_tok = batch_size * (input_tokens + output_tokens) / batch_time
            logger.info(f"Completed in {batch_time:.2f}s - {throughput_tok:.2f} tokens/s")
        
        result = {
            'batch_size': batch_size,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'throughput_tokens_per_sec': throughput_tok,
            'batch_time': batch_time,
            'successful': True
        }
        
        return result
    
    async def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        input_tokens = [500, 1000, 1500, 2000]
        output_tokens = [500, 1000, 1500, 2000]
        
        total_combinations = len(batch_sizes) * len(input_tokens) * len(output_tokens)
        logger.info(f"Starting benchmark with {total_combinations} combinations...")
        
        current_combination = 0
        
        for batch_size in batch_sizes:
            for input_tok in input_tokens:
                for output_tok in output_tokens:
                    current_combination += 1
                    logger.info(f"Progress: {current_combination}/{total_combinations}")
                    
                    result = await self.benchmark_batch(batch_size, input_tok, output_tok)
                    self.results.append(result)
                    
                    # Add small delay between combinations
                    await asyncio.sleep(1)
        
        logger.info("Benchmark completed!")
    
    def save_results(self, filename: str = "./results/throughput_results.json"):
        """Save results to JSON file."""
        timestamp = datetime.now().isoformat()
        data = {
            'timestamp': timestamp,
            'server_url': self.server_url,
            'model_path': 'openai/gpt-oss-20b',
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str = "./results/throughput_results.json"):
        """Load results from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.results = data.get('results', data)
        logger.info(f"Results loaded from {filename}")

    def print_analysis(self):
        """Print detailed analysis of the results to the terminal."""
        successful_results = [r for r in self.results if r.get('successful', True)]
        if not successful_results:
            logger.warning("No successful results to analyze.")
            return

        print("\n" + "=" * 90)
        print("🚀 TOKEN THROUGHPUT BENCHMARK ANALYSIS")
        print("=" * 90)
        print(f"{'Batch Size':<12} {'Input':<10} {'Output':<10} {'Batch Time':<12} {'Tokens/s':<15} {'Success':<10}")
        print("-" * 90)
     
        tokens_per_sec_list = []
        for r in successful_results:
            bs = r['batch_size']
            inp = r['input_tokens']
            out = r['output_tokens']
            bt = r['batch_time']
            # Fallback calculation if loading older JSON, otherwise use the stored metric
            tps = r.get('throughput_tokens_per_sec', bs * (inp + out) / bt)
            tokens_per_sec_list.append((tps, r))
            print(f"{bs:<12} {inp:<10} {out:<10} {bt:<12.2f} {tps:<15.2f} ✓")
     
        if tokens_per_sec_list:
            max_tps, max_r = max(tokens_per_sec_list, key=lambda x: x[0])
            min_tps, _ = min(tokens_per_sec_list, key=lambda x: x[0])
            avg_tps = sum(x[0] for x in tokens_per_sec_list) / len(tokens_per_sec_list)
     
            print(f"\n🎯 Key Insights:")
            print(f"   • Best throughput: {max_tps:.2f} tokens/s (Batch size: {max_r['batch_size']}, In/Out: {max_r['input_tokens']}/{max_r['output_tokens']})")
            print(f"   • Worst throughput: {min_tps:.2f} tokens/s")
            print(f"   • Average throughput: {avg_tps:.2f} tokens/s")
     
        batch_sizes = sorted(set(r['batch_size'] for r in successful_results))
        print(f"\n📊 Throughput by Batch Size:")
        for bs in batch_sizes:
            bs_tps = [tps for tps, r in tokens_per_sec_list if r['batch_size'] == bs]
            if bs_tps:
                print(f"   • Batch size {bs}: {sum(bs_tps)/len(bs_tps):.2f} tokens/s average")

    def plot_token_throughput(self, save_path: str):
        """Create Token Throughput vs Batch Size plot."""
        successful_results = [r for r in self.results if r.get('successful', True)]
        if not successful_results:
            logger.error("No results to plot!")
            return
            
        batch_sizes = [r['batch_size'] for r in successful_results]
        input_tokens = [r['input_tokens'] for r in successful_results]
        output_tokens = [r['output_tokens'] for r in successful_results]
        
        tokens_per_sec = [
            r.get('throughput_tokens_per_sec', r['batch_size'] * (r['input_tokens'] + r['output_tokens']) / r['batch_time'])
            for r in successful_results
        ]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        unique_input_tokens = sorted(set(input_tokens))
        unique_output_tokens = sorted(set(output_tokens))
     
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_input_tokens)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(unique_output_tokens)]
        color_map = {inp: colors[i] for i, inp in enumerate(unique_input_tokens)}
        marker_map = {out: markers[i] for i, out in enumerate(unique_output_tokens)}
     
        token_combinations = sorted(set(zip(input_tokens, output_tokens)))
     
        for inp, out in token_combinations:
            indices = [i for i in range(len(successful_results))
                       if input_tokens[i] == inp and output_tokens[i] == out]
     
            x_vals = [batch_sizes[i] for i in indices]
            y_vals = [tokens_per_sec[i] for i in indices]
     
            sorted_pairs = sorted(zip(x_vals, y_vals))
            x_vals_sorted = [x for x, y in sorted_pairs]
            y_vals_sorted = [y for x, y in sorted_pairs]
     
            ax.plot(x_vals_sorted, y_vals_sorted, marker=marker_map[out], linestyle='-',
                    linewidth=2, markersize=8, color=color_map[inp], alpha=0.7)
     
        ax.set_xlabel('Batch Size', fontsize=18, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        
        input_legend = [Line2D([0], [0], color=color_map[inp], linewidth=2, label=f'Input: {inp}')
                        for inp in unique_input_tokens]
        output_legend = [Line2D([0], [0], marker=marker_map[out], color='gray', linestyle='None', 
                                markersize=8, label=f'Output: {out}')
                         for out in unique_output_tokens]
     
        ax.legend(handles=input_legend + output_legend, loc='best',
                  fontsize=12, ncol=2, framealpha=0.5, columnspacing=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
 
async def main():
    parser = argparse.ArgumentParser(description='LLM Inference Benchmarking Tool')
    parser.add_argument('--server-url', default='http://localhost:5000', help='vLLM server URL')
    parser.add_argument('--output', default=str(Path("./results/throughput_results.json")), help='Output JSON file')
    parser.add_argument('--load-results', type=str, help='Load results from JSON instead of running benchmark')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark = LLMBenchmark(args.server_url)
    
    if args.load_results:
        benchmark.load_results(args.load_results)
    else:
        try:
            response = requests.get(f"{args.server_url}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"vLLM server not accessible at {args.server_url}")
                return
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            return
        
        await benchmark.run_full_benchmark()
        benchmark.save_results(args.output)
    
    benchmark.print_analysis()
    
    plot_path = str(output_dir / "throughput_results.png")
    benchmark.plot_token_throughput(save_path=plot_path)
 
if __name__ == "__main__":
    asyncio.run(main())
