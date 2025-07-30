import json
import asyncio
import os
import hashlib
import datetime
import shutil
import argparse

from benchmark_evaluator import (
    bulk_query_with_progress,
    evaluate_solution,
    evaluate_with_llm_judge,
)
from datasets import load_dataset
from benchmark_evaluator.query import (
    SUPPORTED_MODELS,
    SUPPORTED_MODELS_OPENAI,
    SUPPORTED_MODELS_GEMINI,
    SUPPORTED_MODELS_ANTHROPIC,
    SUPPORTED_MODELS_TOGETHER,
    SUPPORTED_MODELS_DEEPSEEK,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"
QUERY_LOG_FILE = "results/query_log.json"

def load_config():
    """Load configuration from config.json file."""
    config_path = "scripts/eval_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
        print("Please create an eval_config.json file with a 'models' dictionary mapping model names to boolean values.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in eval_config.json at {config_path}")
        exit(1)

def validate_api_keys(selected_models):
    """Validate that we have API keys for the model classes we're using."""
    # Check which model classes we're using
    using_openai = any(model in SUPPORTED_MODELS_OPENAI for model in selected_models)
    using_gemini = any(model in SUPPORTED_MODELS_GEMINI for model in selected_models)
    using_anthropic = any(model in SUPPORTED_MODELS_ANTHROPIC for model in selected_models)
    using_together = any(model in SUPPORTED_MODELS_TOGETHER for model in selected_models)
    
    # Validate API keys
    missing_keys = []
    if using_openai and not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if using_gemini and not os.environ.get("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")
    if using_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if using_together and not os.environ.get("TOGETHER_API_KEY"):
        missing_keys.append("TOGETHER_API_KEY")
    
    if missing_keys:
        print(f"Error: Missing required API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables in your .env file")
        exit(1)

def validate_models(config):
    """Validate that selected models are supported."""
    if 'models' not in config:
        print("Error: 'models' dictionary not found in config.json")
        exit(1)
    
    # Get selected models (those set to true)
    selected_models = [model for model, enabled in config['models'].items() if enabled]
    
    # Check if any models are selected
    if not selected_models:
        print("Error: No models selected. Please set at least one model to true in config.json")
        exit(1)
    
    # Check if all selected models are supported
    invalid_models = [model for model in selected_models if model not in SUPPORTED_MODELS]
    if invalid_models:
        print(f"Error: The following selected models are not supported: {invalid_models}")
        print(f"Supported models are: {list(SUPPORTED_MODELS.keys())}")
        exit(1)
    
    # Validate API keys for the model classes we're using
    validate_api_keys(selected_models)
    
    return selected_models

def get_prompt_hash(prompt):
    """Generate a hash for the prompt text."""
    return hashlib.md5(prompt.encode()).hexdigest()

def log_prompt(prompt, model_name, query_idx, log_file):
    """Log a completed prompt query."""
    log = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = set(json.load(f))
    
    prompt_hash = get_prompt_hash(prompt)
    log.add(prompt_hash)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(list(log), f, indent=2)

def process_results(query_results, prompt_list, solution_list, parameter_list, type_list, index_list, *, use_llm_judge: bool = False, judge_model: str = "gpt-4o-mini"):
    """Process query results and generate evaluation results."""
    full_results = []
    results = []
    
    # Map problem types to rubric files (extend as needed)
    RUBRIC_MAP = {
        "boundary_layers": "rubrics/boundary_rubric.txt",
        "wkb": "rubrics/wkb_rubric.txt",
        "nonlinear_pde": "rubrics/nonlinear_pde_rubric.txt"
    }

    for response, prompt_idx, model_name, error, query_idx in query_results:
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx} (query {query_idx}): {response}")
            continue
            
        if use_llm_judge:
            rubric_path = RUBRIC_MAP.get(type_list[prompt_idx])
            if rubric_path is None:
                from benchmark_evaluator.evaluator import EvaluationResult
                eval_result = EvaluationResult(success=False, error_message=f"No rubric for problem type {type_list[prompt_idx]}")
            else:
                eval_result = evaluate_with_llm_judge(
                    model_response=response,
                    solution_string=solution_list[prompt_idx],
                    rubric_path=rubric_path,
                    judge_model=judge_model,
                )
        else:
            eval_result = evaluate_solution(response, solution_list[prompt_idx], parameter_list[prompt_idx])
        eval_result_serialized = eval_result.to_dict()
        
        full_results.append({
            "prompt_idx": prompt_idx,
            "query_idx": query_idx,
            "prompt": prompt_list[prompt_idx],
            "model_name": model_name,
            "model_response": response,
            "eval_result": eval_result_serialized,
            "type": type_list[prompt_idx],
            "index": index_list[prompt_idx]
        })
        
        try:
            if use_llm_judge:
                results.append({
                    "prompt_idx": prompt_idx,
                    "query_idx": query_idx,
                    "prompt": prompt_list[prompt_idx],
                    "model_name": model_name,
                    "model_response": response,
                    "type": type_list[prompt_idx],
                    "index": index_list[prompt_idx],
                    "eval_success": eval_result.success,
                    "score": eval_result.score,
                    "judge_reasoning": eval_result.judge_reasoning,
                })
            else:
                results.append({
                    "prompt_idx": prompt_idx,
                    "query_idx": query_idx,
                    "prompt": prompt_list[prompt_idx],
                    "model_name": model_name,
                    "model_response": response,
                    "type": type_list[prompt_idx],
                    "index": index_list[prompt_idx],
                    "eval_success": eval_result.success,
                    "is_equivalent": eval_result.is_equivalent,
                    "model_latex_solution": getattr(eval_result.model_result, 'extracted_solutions', None),
                    "solution_latex": getattr(eval_result.solution_result, 'extracted_solutions', None),
                    "model_eval_result": getattr(eval_result.model_result, 'evaluation_results', None),
                    "solution_eval_result": getattr(eval_result.solution_result, 'evaluation_results', None),
                })
        except Exception as e:
            print(f"Error serializing eval result for prompt {prompt_idx} and model {model_name} (query {query_idx}): {e}")
            continue
    
    return full_results, results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run model evaluation on benchmark data")
    parser.add_argument("--limit", type=int, help="Limit evaluation to the first N prompts", default=0)
    parser.add_argument("--no-save", action="store_true", help="Do not save results, print them to the console instead.")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM-as-a-judge rubric based grading instead of numeric/symbolic evaluator.")
    parser.add_argument("--models", nargs="*", default=None, help="Override model list (e.g. --models 'Gemini 2.5 Flash')")
    parser.add_argument("--problem-type", nargs="*", default=None, help="Filter dataset to these problem types (e.g. --problem-type boundary_layers wkb)")
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config()

    # Pick models: CLI override > config file
    if args.models:
        selected_models = args.models
        validate_api_keys(selected_models)
    else:
        selected_models = validate_models(config)
    
    # Get number of queries per prompt from config, default to 1
    num_queries = config.get('num_queries', 1)
    print(f"Running evaluation with models: {selected_models}", flush=True)
    print(f"Number of queries per prompt: {num_queries}", flush=True)
    
    if not args.no_save:
        # Create a timestamp for unique results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory with timestamp
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Update results paths with timestamp
        suffix = "_llm" if args.use_llm_judge else ""

        query_log_file = f"{results_dir}/query_log.json"  # unchanged
        query_results_file = f"{results_dir}/query_results.json"  # unchanged
        full_results_file = f"{results_dir}/full_results{suffix}.json"
        results_file = f"{results_dir}/results{suffix}.json"
        
        print(f"Results will be saved to directory: {results_dir}", flush=True)
    else:
        print("--- Running in no-save mode. Results will be printed to the console. ---", flush=True)
        query_log_file = None
        query_results_file = None
        full_results_file = None
        results_file = None
    
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]
    
    # Optional filtering by problem type
    if args.problem_type:
        types_str = ", ".join(args.problem_type)
        print(f"Filtering dataset to problem types: {types_str}", flush=True)
        allowed = set(args.problem_type)
        filtered = [
            (p, s, par, t, idx)
            for p, s, par, t, idx in zip(prompt_list, solution_list, parameter_list, type_list, index_list)
            if t in allowed
        ]
        if not filtered:
            print("No problems found with specified problem types. Exiting.")
            return
        prompt_list, solution_list, parameter_list, type_list, index_list = map(list, zip(*filtered))

    # Diagnostic summary of remaining prompt types
    from collections import Counter
    type_counts = Counter(type_list)
    summary = ", ".join(f"{t}:{c}" for t, c in type_counts.items())
    print(f"Prompts kept after filtering: {len(prompt_list)} (by type → {summary})", flush=True)

    # Limit the number of prompts if requested (after filtering)
    if args.limit > 0:
        print(f"Limiting evaluation to the first {args.limit} prompts", flush=True)
        prompt_list = prompt_list[:args.limit]
        solution_list = solution_list[:args.limit]
        parameter_list = parameter_list[:args.limit]
        type_list = type_list[:args.limit]
        index_list = index_list[:args.limit]

    # Create a list of (prompt, prompt_idx, query_idx) tuples for each query
    prompt_tuples = []
    for idx, prompt in enumerate(prompt_list):
        for query_idx in range(num_queries):
            prompt_tuples.append((prompt, idx, query_idx))

    # Run queries with repeated prompts
    query_results = asyncio.run(bulk_query_with_progress([p[0] for p in prompt_tuples], selected_models))
    print(f"Number of query results: {len(query_results)}", flush=True)


    # Update prompt_idx in results to match original indices and add query_idx
    updated_results = []
    for i, (response, _, model_name, error) in enumerate(query_results):
        # Calculate prompt_idx and query_idx based on the result's position
        total_queries_per_prompt = len(selected_models) * num_queries
        prompt_idx = i // total_queries_per_prompt
        query_idx = (i % total_queries_per_prompt) // len(selected_models)
        updated_results.append((response, prompt_idx, model_name, error, query_idx))
        
        # Log successful query
        if not error and not args.no_save:
            # Update the query log function to use the new path
            log_prompt(prompt_list[prompt_idx], model_name, query_idx, query_log_file)
    
    print(f"Number of updated results: {len(updated_results)}", flush=True)

    if not args.no_save:
        try:
            serializable_results = [
                {
                    "prompt_idx": prompt_idx,
                    "query_idx": query_idx,
                    "model_name": model_name,
                    "response": response,
                    "error": error
                }
                for response, prompt_idx, model_name, error, query_idx in updated_results
            ]
            with open(query_results_file, "w") as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            print(f"Error saving query results: {e}", flush=True)
    # Process results
    full_results, results = process_results(
        updated_results,
        prompt_list,
        solution_list,
        parameter_list,
        type_list,
        index_list,
        use_llm_judge=args.use_llm_judge,
        judge_model="gpt-4o-mini",
    )

    if args.no_save:
        print("\n--- Evaluation Results (Dry Run) ---")
        print(json.dumps(results, indent=2))
    else:
        # Save results
        with open(full_results_file, "w") as f:
            json.dump(full_results, f, indent=2)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved {len(results)} results to {results_file}", flush=True)
        print(f"Saved {len(full_results)} results to {full_results_file}", flush=True)

if __name__ == "__main__":
    main()
    