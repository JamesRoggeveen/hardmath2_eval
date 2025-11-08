import json
import os
import sys
import argparse
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from typing import Dict, Any, List
import re
from collections import defaultdict

def generate_summary(results_dir):
    # Detect which results file to use
    results_path_pass = os.path.join(results_dir, 'results.json')
    results_path_llm  = os.path.join(results_dir, 'results_llm.json')
    metric_mode = 'pass'  # default – classic is_equivalent metric
    if os.path.exists(results_path_pass):
        results_file = results_path_pass
    elif os.path.exists(results_path_llm):
        results_file = results_path_llm
        metric_mode = 'llm'  # rubric scores between 0-1
    else:
        raise FileNotFoundError(f"Neither results.json nor results_llm.json found in {results_dir}")

    summary_file = os.path.join(results_dir, 'summary.json')
    
    with open(results_file, 'r') as f:
        results = json.load(f)

    # If we are in LLM-judge mode, fabricate a boolean pass flag so that the
    # downstream logic that references `is_equivalent` does not blow up.  We
    # still preserve the original numeric score for later averaging.
    if metric_mode == 'llm':
        SCORE_PASS_THRESHOLD = 0.5  # feel free to adjust per-rubric
        for row in results:
            # Inject a synthetic binary flag if missing
            raw_score = row.get('score', 0)
            if raw_score is None:
                raw_score = 0.0  # treat missing score as zero
            row['score'] = raw_score  # normalise value
            row['is_equivalent'] = raw_score >= SCORE_PASS_THRESHOLD

    # Group results by model name and prompt_idx
    results_by_model = defaultdict(lambda: defaultdict(list))
    for result in results:
        model_name = result['model_name']
        prompt_idx = result['prompt_idx']
        results_by_model[model_name][prompt_idx].append(result)

    # Get unique models and question types
    models = sorted(list(results_by_model.keys()))
    question_types = sorted(set(result['type'] for result in results))

    # Get all unique prompt indices
    all_prompt_indices = set()
    for model_results in results_by_model.values():
        all_prompt_indices.update(model_results.keys())

    # Calculate dataset distribution statistics - only count each unique problem once
    problem_types = defaultdict(int)
    seen_problems = set()  # Track unique problems by their prompt_idx
    for result in results:
        prob_type = result['type']
        prompt_idx = result['prompt_idx']
        # Only count each problem once, regardless of model or query
        if prompt_idx not in seen_problems:
            problem_types[prob_type] += 1
            seen_problems.add(prompt_idx)

    # Convert to percentages
    total_problems = sum(problem_types.values())
    problem_type_percentages = {k: (v/total_problems)*100 for k,v in problem_types.items()}

    # Calculate success rates for each model and prompt
    prompt_success_rates = defaultdict(dict)
    for model_name, prompt_results in results_by_model.items():
        for prompt_idx, queries in prompt_results.items():
            successful_queries = sum(1 for query in queries if query['is_equivalent'])
            success_rate = (successful_queries / len(queries)) * 100
            prompt_success_rates[model_name][prompt_idx] = success_rate

    # Calculate and save summary statistics
    summary_stats = {
        "models": models,
        "question_types": question_types,
        "dataset_distribution": {
            "total_problems": total_problems,
            "problem_type_counts": dict(problem_types),
            "problem_type_percentages": problem_type_percentages
        },
        "prompt_success_rates": {
            model: dict(rates) for model, rates in prompt_success_rates.items()
        }
    }
    
    # Calculate evaluation success statistics per model
    model_eval_stats = {}
    for model_name, prompt_results in results_by_model.items():
        total_queries = sum(len(queries) for queries in prompt_results.values())
        total_eval_success = sum(
            sum(1 for query in queries if query.get('eval_success', False))
            for queries in prompt_results.values()
        )
        eval_success_rate = (total_eval_success / total_queries) * 100 if total_queries > 0 else 0
        
        model_eval_stats[model_name] = {
            "total_queries": total_queries,
            "total_eval_success": total_eval_success,
            "eval_success_rate": eval_success_rate
        }
    
    summary_stats["model_eval_stats"] = model_eval_stats
    
    for model_name, prompt_results in results_by_model.items():
        total_prompts = len(prompt_results)
        total_queries = sum(len(queries) for queries in prompt_results.values())
        total_successful_queries = sum(
            sum(1 for query in queries if query['is_equivalent'])
            for queries in prompt_results.values()
        )
        
        # When using LLM-judge scores we also need to accumulate the raw
        # numeric values so we can report averages.
        if metric_mode == 'llm':
            total_score_sum = sum(
                sum(query.get('score', 0) for query in queries)
                for queries in prompt_results.values()
            )

        # Calculate success rate for each prompt
        prompt_stats = {}
        for prompt_idx in all_prompt_indices:  # Use all prompt indices
            if prompt_idx in prompt_results:
                queries = prompt_results[prompt_idx]
                # Sort queries by query_idx to ensure first query is at index 0
                queries.sort(key=lambda x: x.get('query_idx', 0))
                
                # Get the problem type from the first query
                problem_type = queries[0]['type'] if queries else 'Unknown'
                
                # Calculate pass@1 (using only first query)
                pass_at_1 = queries[0]['is_equivalent'] if queries else False
                
                # Calculate mean rubric score for this prompt (LLM mode)
                if metric_mode == 'llm':
                    mean_score = sum(q.get('score', 0) for q in queries) / len(queries)
                
                # Calculate pass@n (all queries must be equivalent)
                pass_at_n = all(query['is_equivalent'] for query in queries)
                
                successful_queries = sum(1 for query in queries if query['is_equivalent'])
                prompt_stats[f"prompt_{prompt_idx}"] = {
                    "problem_type": problem_type,
                    "total_queries": len(queries),
                    "successful_queries": successful_queries,
                    "success_rate": (successful_queries/len(queries))*100 if queries else 0,
                    "pass_at_1": pass_at_1,
                    "pass_at_n": pass_at_n,
                    **({"mean_score": mean_score} if metric_mode == 'llm' else {})
                }
            else:
                # For prompts that this model hasn't run, add an entry with 0 success
                prompt_stats[f"prompt_{prompt_idx}"] = {
                    "problem_type": "Unknown",  # We don't know the type since it wasn't run
                    "total_queries": 0,
                    "successful_queries": 0,
                    "success_rate": 0,
                    "pass_at_1": False,
                    "pass_at_n": False
                }

        # Calculate overall pass@1 and pass@n rates
        total_pass_at_1 = sum(1 for stats in prompt_stats.values() if stats["pass_at_1"])
        total_pass_at_n = sum(1 for stats in prompt_stats.values() if stats["pass_at_n"])
        overall_pass_at_1_rate = (total_pass_at_1 / total_prompts) * 100 if total_prompts > 0 else 0
        overall_pass_at_n_rate = (total_pass_at_n / total_prompts) * 100 if total_prompts > 0 else 0

        # Calculate success rates per question type
        question_type_stats = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "total_prompts": 0,
            "pass_at_1_prompts": 0,
            "pass_at_n_prompts": 0
        })
        
        # First pass: collect all queries and their success status
        for queries in prompt_results.values():
            # Sort queries by query_idx
            queries.sort(key=lambda x: x.get('query_idx', 0))
            
            # Get the question type from the first query
            if queries:
                q_type = queries[0]['type']
                question_type_stats[q_type]["total_prompts"] += 1
                
                # Check pass@1
                if queries[0]['is_equivalent']:
                    question_type_stats[q_type]["pass_at_1_prompts"] += 1
                
                # Check pass@n
                if all(query['is_equivalent'] for query in queries):
                    question_type_stats[q_type]["pass_at_n_prompts"] += 1
                
                # Count all queries
                for query in queries:
                    question_type_stats[q_type]["total"] += 1
                    if query['is_equivalent']:
                        question_type_stats[q_type]["successful"] += 1

        # Convert question type stats to percentages
        question_type_rates = {}
        for q_type, stats in question_type_stats.items():
            success_rate = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            pass_at_1_rate = (stats["pass_at_1_prompts"] / stats["total_prompts"]) * 100 if stats["total_prompts"] > 0 else 0
            pass_at_n_rate = (stats["pass_at_n_prompts"] / stats["total_prompts"]) * 100 if stats["total_prompts"] > 0 else 0
            
            if metric_mode == 'llm':
                # Compute mean rubric score for this question type
                score_values = []
                for queries in prompt_results.values():
                    if queries and queries[0]['type'] == q_type:
                        score_values.extend([q.get('score', 0) for q in queries])
                mean_type_score = sum(score_values)/len(score_values) if score_values else 0
                question_type_rates[q_type] = {
                    "total_queries": stats["total"],
                    "successful_queries": stats["successful"],
                    "success_rate": success_rate,
                    "total_prompts": stats["total_prompts"],
                    "pass_at_1_rate": pass_at_1_rate,
                    "pass_at_n_rate": pass_at_n_rate,
                    "mean_score": mean_type_score
                }
            else:
                question_type_rates[q_type] = {
                    "total_queries": stats["total"],
                    "successful_queries": stats["successful"],
                    "success_rate": success_rate,
                    "total_prompts": stats["total_prompts"],
                    "pass_at_1_rate": pass_at_1_rate,
                    "pass_at_n_rate": pass_at_n_rate
                }

        # Assemble the per-model summary block
        model_block = {
            "total_prompts": total_prompts,
            "total_queries": total_queries,
            "total_successful_queries": total_successful_queries,
            "overall_success_rate": (total_successful_queries/total_queries)*100 if total_queries > 0 else 0,
            "overall_pass_at_1_rate": (total_pass_at_1 / total_prompts) * 100 if total_prompts > 0 else 0,
            "overall_pass_at_n_rate": (total_pass_at_n / total_prompts) * 100 if total_prompts > 0 else 0,
            "prompt_breakdown": prompt_stats,
            "question_type_breakdown": question_type_rates
        }
        if metric_mode == 'llm':
            model_block["overall_mean_score"] = total_score_sum / total_queries if total_queries else 0
        summary_stats[model_name] = model_block

    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=4)

    print(f"Summary statistics saved as {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary statistics from evaluation results")
    parser.add_argument("--results-dir", type=str, default="results", 
                        help="Directory containing results.json and where summary.json will be saved")
    args = parser.parse_args()
    
    generate_summary(args.results_dir)
