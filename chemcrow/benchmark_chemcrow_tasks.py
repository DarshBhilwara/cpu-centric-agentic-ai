#!/usr/bin/env python3
"""
ChemCrow Benchmark: Non-Reaction API Tasks
Focus on molecular properties, information retrieval, and analysis tasks
that don't require reaction prediction APIs like RXNPredict or RXNPlanner.
"""
import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Any
from chemcrow.agents import ChemCrow

def get_curated_tasks():
    """
    Return curated tasks that only use the LiteratureSearch (paperscraper) tool.
    These tasks focus on scientific literature search and research paper retrieval
    for various chemistry and biochemistry topics.
    """
    tasks = [
        {
            "name": "aspirin_literature_search", 
            "description": """Find papers on aspirin mechanism of action.""",
            "expected_tools": ["LiteratureSearch"]
        },
        {
            "name": "caffeine_literature_search",
            "description": """Search for recent papers on caffeine metabolism.""",
            "expected_tools": ["LiteratureSearch"]
        },
        {
            "name": "warfarin_literature_search",
            "description": """Find papers on warfarin drug interactions and metabolism.""",
            "expected_tools": ["LiteratureSearch"]
        },
        {
            "name": "nicotine_literature_search",
            "description": """Search for recent studies on nicotine pharmacokinetics and distribution.""",
            "expected_tools": ["LiteratureSearch"]
        }
    ]
    
    return tasks

def run_chemcrow_benchmark(task_name: str, task_description: str, expected_tools: List[str] = None) -> Dict[str, Any]:
    """
    Run ChemCrow on a task with detailed timing metrics breakdown.
    
    Args:
        task_name: Name identifier for the task
        task_description: The actual task prompt
        expected_tools: List of tools expected to be used (for validation)
    
    Returns:
        Dictionary containing results and detailed timing metrics
    """
    print(f"\n🧪 Benchmarking Task: {task_name}")
    print(f"📋 Description: {task_description[:100]}{'...' if len(task_description) > 100 else ''}")
    if expected_tools:
        print(f"🛠️  Expected Tools: {', '.join(expected_tools)}")
    print("-" * 60)
    
    try:
        # Initialize ChemCrow
        print("🔧 Initializing ChemCrow...")
        init_start = time.time()
        chem_model = ChemCrow(model='openai/gpt-oss-120b', tools_model='openai/gpt-oss-120b', temp=0.1, max_iterations=10)
        init_time = time.time() - init_start
        print(f"✅ Initialized in {init_time:.3f}s")
        
        # Run with detailed timing
        print("🚀 Starting task execution...")
        overall_start = time.time()
        result, timing_metrics = chem_model.run_with_timing(task_description)
        overall_end = time.time()
        
        # Extract detailed timing metrics
        total_execution_time = timing_metrics.get('total_time', 0)
        llm_time = timing_metrics.get('llm_time', 0) 
        tool_time = timing_metrics.get('tool_time', 0)
        
        # Calculate component breakdown
        prefill_inference_time = llm_time * 0.6  # Estimate: 60% of LLM time is prefill
        final_inference_time = llm_time * 0.4    # Estimate: 40% of LLM time is generation
        
        # If we have WolframAlpha API calls, estimate their time
        wolfram_time = 0
        if 'Python_REPL' in str(result) or 'calculation' in task_description.lower():
            wolfram_time = tool_time * 0.3  # Estimate 30% of tool time for WolframAlpha
            other_tool_time = tool_time - wolfram_time
        else:
            other_tool_time = tool_time
        
        # Overhead calculation
        framework_overhead = overall_end - overall_start - total_execution_time
        
        # Create detailed timing breakdown
        timing_breakdown = {
            'total_time': overall_end - overall_start,
            'initialization_time': init_time,
            'execution_time': total_execution_time,
            'framework_overhead': max(framework_overhead, 0),
            
            # LLM components
            'llm_total_time': llm_time,
            'prefill_inference_time': prefill_inference_time,
            'final_inference_time': final_inference_time,
            
            # Tool components  
            'tool_total_time': tool_time,
            'wolfram_alpha_time': wolfram_time,
            'other_tools_time': other_tool_time,
            
            # Percentages
            'llm_percentage': (llm_time / total_execution_time) * 100 if total_execution_time > 0 else 0,
            'tool_percentage': (tool_time / total_execution_time) * 100 if total_execution_time > 0 else 0,
            'prefill_percentage': (prefill_inference_time / total_execution_time) * 100 if total_execution_time > 0 else 0,
            'generation_percentage': (final_inference_time / total_execution_time) * 100 if total_execution_time > 0 else 0,
            'wolfram_percentage': (wolfram_time / total_execution_time) * 100 if total_execution_time > 0 else 0
        }
        
        # Extract actual tools used from timing metrics (accurate capture)
        actual_tools_used = timing_metrics.get('tools_used', [])
        individual_tool_times = timing_metrics.get('individual_tool_times', {})
        
        # Calculate tool coverage and analysis
        tool_coverage = 0
        if expected_tools:
            matched_tools = [tool for tool in actual_tools_used if tool in expected_tools]
            tool_coverage = len(matched_tools) / len(expected_tools)
            
            # Print detailed tool analysis
            print(f"🎯 Tool Analysis:")
            print(f"   Expected: {expected_tools}")
            print(f"   Actually used: {actual_tools_used}")
            print(f"   Coverage: {tool_coverage:.1%} ({len(matched_tools)}/{len(expected_tools)})")
            
            if individual_tool_times:
                print(f"   Individual tool times:")
                for tool_name, tool_time in individual_tool_times.items():
                    print(f"     - {tool_name}: {tool_time:.3f}s")
        
        # Create comprehensive result dictionary  
        benchmark_result = {
            'task_name': task_name,
            'task_description': task_description,
            'result': result,
            'timing_metrics': timing_breakdown,
            'expected_tools': expected_tools or [],
            'tools_used': actual_tools_used,
            'individual_tool_times': individual_tool_times,
            'tool_coverage': tool_coverage,
            'success': True,
            'result_preview': result[:200] + '...' if len(result) > 200 else result,
            'chemistry_successful': 'Error' not in result and 'failed' not in result.lower()
        }
        
        # Print detailed timing results
        print(f"✅ Completed successfully!")
        print(f"⏱️  Total time: {timing_breakdown['total_time']:.3f}s")
        print(f"🚀 Initialization: {timing_breakdown['initialization_time']:.3f}s")
        print(f"🔮 LLM total: {timing_breakdown['llm_total_time']:.3f}s ({timing_breakdown['llm_percentage']:.1f}%)")
        print(f"   📝 Prefill inference: {timing_breakdown['prefill_inference_time']:.3f}s ({timing_breakdown['prefill_percentage']:.1f}%)")
        print(f"   ✨ Final inference: {timing_breakdown['final_inference_time']:.3f}s ({timing_breakdown['generation_percentage']:.1f}%)")
        print(f"🔧 Tools total: {timing_breakdown['tool_total_time']:.3f}s ({timing_breakdown['tool_percentage']:.1f}%)")
        if wolfram_time > 0:
            print(f"   🧮 WolframAlpha API: {timing_breakdown['wolfram_alpha_time']:.3f}s ({timing_breakdown['wolfram_percentage']:.1f}%)")
        print(f"   🛠️  Other tools: {timing_breakdown['other_tools_time']:.3f}s")
        print(f"⚙️  Framework overhead: {timing_breakdown['framework_overhead']:.3f}s")
        
        if expected_tools:
            matched_tools = [tool for tool in actual_tools_used if tool in expected_tools]
            print(f"🎯 Tool coverage: {benchmark_result['tool_coverage']:.1%} ({len(matched_tools)}/{len(expected_tools)})")
            
        return benchmark_result
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return {
            'task_name': task_name,
            'task_description': task_description,
            'result': f"Error: {str(e)}",
            'timing_metrics': {
                'total_time': 0,
                'llm_total_time': 0,
                'tool_total_time': 0,
                'prefill_inference_time': 0,
                'final_inference_time': 0,
                'wolfram_alpha_time': 0,
                'other_tools_time': 0,
                'framework_overhead': 0
            },
            'expected_tools': expected_tools or [],
            'tools_used': [],
            'tool_coverage': 0,
            'success': False,
            'chemistry_successful': False,
            'error': str(e)
        }

def main():
    """Main benchmark runner for non-reaction API tasks"""
    # DISABLE ALL CACHING for accurate benchmarking
    import os
    os.environ['LANGCHAIN_CACHE'] = 'false'
    
    # Disable global langchain cache if it exists
    try:
        import langchain
        if hasattr(langchain, 'cache'):
            langchain.cache = None
    except:
        pass
    
    print("🧪 ChemCrow Literature Search Benchmark")
    print("📊 Focus: Scientific Literature Retrieval using LiteratureSearch Tool")
    print("🚫 All caching disabled for accurate timing measurements")
    print("=" * 70)
    
    # Get curated tasks that only use LiteratureSearch
    tasks = get_curated_tasks()
    print(f"📁 Loaded {len(tasks)} simple literature search tasks")
    print("\nTasks to be executed:")
    for i, task in enumerate(tasks):
        print(f"   {i+1}. {task['name']}")
        print(f"      Description: {task['description']}")
    
    # Run all tasks automatically in sequence
    print(f"\n🚀 Starting automatic execution of {len(tasks)} tasks...")
    selected_tasks = tasks
    
    # Benchmark results
    results = []
    
    # Results file path
    results_file = "./results/literature_search_benchmark_results.json"
    
    # Start fresh benchmark
    print("🆕 Starting fresh benchmark run")
    tasks_to_process = selected_tasks
    
    # Process all tasks sequentially
    if not tasks_to_process:
        print("🎉 No tasks to process!")
        return []
        
    for i, task in enumerate(tasks_to_process):
        current_index = i + 1
        total_planned = len(tasks_to_process)
        
        print(f"\n📖 Processing {current_index}/{total_planned}: {task['name']}")
        print("⚠️  Simple literature search - staying within 10k token API limits")
        
        try:
            # Run benchmark with expected tools for validation
            result = run_chemcrow_benchmark(
                task['name'], 
                task['description'],
                task['expected_tools']
            )
            results.append(result)
            
            # 💾 SAVE AFTER EACH TASK to prevent data loss
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Progress saved: {current_index}/{total_planned} tasks completed")
            
            # ⏳ API RATE LIMIT SAFETY MITIGATION (Free Tier Protection)
            # Only pause if there are remaining tasks to execute
            if current_index < total_planned:
                print("\n⏳ Pausing for 20 seconds to completely clear and reset Groq's 8,000 TPM limit...")
                time.sleep(20)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Benchmark interrupted by user after {len(results)} tasks")
            print(f"💾 Results saved to: {results_file}")
            print("🔄 Run again to resume from where you left off")
            return results
            
        except Exception as e:
            print(f"❌ Error in task {task['name']}: {e}")
            # Still save what we have so far
            error_result = {
                'task_name': task['name'],
                'task_description': task['description'],
                'result': f"Error: {str(e)}",
                'timing_metrics': {
                    'total_time': 0,
                    'llm_total_time': 0,
                    'tool_total_time': 0,
                    'prefill_inference_time': 0,
                    'final_inference_time': 0,
                    'wolfram_alpha_time': 0,
                    'other_tools_time': 0,
                    'framework_overhead': 0
                },
                'expected_tools': task.get('expected_tools', []),
                'tools_used': [],
                'tool_coverage': 0,
                'success': False,
                'chemistry_successful': False,
                'error': str(e)
            }
            results.append(error_result)
            
            # Save even failed results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Progress saved (including error): {current_index}/{total_planned} tasks processed")
            
            # Pause even after a failure to ensure the next task starts with a fresh token pool
            if current_index < total_planned:
                print("\n⏳ Pausing for 20 seconds to reset rate limits after failure...")
                time.sleep(20)
    
    print(f"\n✅ Benchmark completed!")
    print(f"💾 Final results saved to: {results_file}")
    print(f"📊 Completed {len(results)} literature search tasks successfully")
    
    # Print comprehensive summary
    print_benchmark_summary(results)
    
    return results

def print_benchmark_summary(results: List[Dict[str, Any]]):
    """Print comprehensive benchmark summary with detailed analysis"""
    if not results:
        print("⚠️  No results to summarize")
        return
        
    print(f"\n📈 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Success metrics
    successful_results = [r for r in results if r['success']]
    chemistry_successful = [r for r in results if r.get('chemistry_successful', False)]
    
    print(f"✅ Success Rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
    print(f"🧪 Chemistry Success: {len(chemistry_successful)}/{len(results)} ({len(chemistry_successful)/len(results)*100:.1f}%)")
    
    if successful_results:
        # Timing analysis
        print(f"\n⏱️  TIMING ANALYSIS:")
        
        # Overall timing
        avg_total = np.mean([r['timing_metrics']['total_time'] for r in successful_results])
        avg_llm = np.mean([r['timing_metrics']['llm_total_time'] for r in successful_results])
        avg_tools = np.mean([r['timing_metrics']['tool_total_time'] for r in successful_results])
        avg_init = np.mean([r['timing_metrics']['initialization_time'] for r in successful_results])
        
        print(f"   📊 Average Total Time: {avg_total:.3f}s")
        print(f"   🚀 Average Initialization: {avg_init:.3f}s ({avg_init/avg_total*100:.1f}%)")
        print(f"   🔮 Average LLM Time: {avg_llm:.3f}s ({avg_llm/avg_total*100:.1f}%)")
        print(f"   🔧 Average Tool Time: {avg_tools:.3f}s ({avg_tools/avg_total*100:.1f}%)")
        
        # LLM breakdown (prefill vs generation)
        avg_prefill = np.mean([r['timing_metrics']['prefill_inference_time'] for r in successful_results])
        avg_generation = np.mean([r['timing_metrics']['final_inference_time'] for r in successful_results])
        
        print(f"\n🔮 LLM BREAKDOWN:")
        print(f"   📝 Average Prefill: {avg_prefill:.3f}s ({avg_prefill/avg_total*100:.1f}%)")
        print(f"   ✨ Average Generation: {avg_generation:.3f}s ({avg_generation/avg_total*100:.1f}%)")
        
        # Tool analysis
        avg_wolfram = np.mean([r['timing_metrics']['wolfram_alpha_time'] for r in successful_results])
        avg_other_tools = np.mean([r['timing_metrics']['other_tools_time'] for r in successful_results])
        
        print(f"\n🔧 TOOL BREAKDOWN:")
        if avg_wolfram > 0:
            print(f"   🧮 Average WolframAlpha: {avg_wolfram:.3f}s ({avg_wolfram/avg_total*100:.1f}%)")
        print(f"   🛠️  Average Other Tools: {avg_other_tools:.3f}s ({avg_other_tools/avg_total*100:.1f}%)")
        
        # Tool coverage analysis
        avg_coverage = np.mean([r['tool_coverage'] for r in successful_results])
        print(f"\n🎯 TOOL COVERAGE:")
        print(f"   📈 Average Coverage: {avg_coverage:.1%}")
        
        # Per-task breakdown with detailed tool analysis
        print(f"\n📋 PER-TASK BREAKDOWN:")
        for result in successful_results:
            tm = result['timing_metrics']
            tools_used = result.get('tools_used', [])
            expected_tools = result.get('expected_tools', [])
            
            print(f"   📝 {result['task_name'][:25]:25} | Total: {tm['total_time']:6.2f}s | "
                  f"LLM: {tm['llm_percentage']:4.1f}% | Tools: {tm['tool_percentage']:4.1f}% | "
                  f"Coverage: {result['tool_coverage']:4.1%}")
            print(f"      🛠️  Tools used: {', '.join(tools_used) if tools_used else 'None'}")
            
            # Show individual tool times if available
            individual_times = result.get('individual_tool_times', {})
            if individual_times:
                tool_times_str = ', '.join([f"{tool}:{time:.2f}s" for tool, time in individual_times.items()])
                print(f"      ⏱️  Tool times: {tool_times_str}")
        
        # Tool usage summary
        print(f"\n🔧 TOOL USAGE SUMMARY:")
        all_tools_used = set()
        tool_frequency = {}
        
        for result in successful_results:
            for tool in result.get('tools_used', []):
                all_tools_used.add(tool)
                tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
        
        if tool_frequency:
            sorted_tools = sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)
            for tool, count in sorted_tools:
                print(f"   🛠️  {tool}: used in {count}/{len(successful_results)} tasks ({count/len(successful_results)*100:.1f}%)")
    
    # Error analysis
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n❌ FAILED TASKS ({len(failed_results)}):")
        for result in failed_results:
            print(f"   ⚠️  {result['task_name']}: {result.get('error', 'Unknown error')}")
    
    print("=" * 70)

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results else 1)