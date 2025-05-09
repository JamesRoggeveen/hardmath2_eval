import json
import asyncio
import os
import hashlib

from benchmark_evaluator import bulk_query_with_progress, evaluate_solution
from datasets import load_dataset
from benchmark_evaluator.query import SUPPORTED_MODELS, SUPPORTED_MODELS_OPENAI, SUPPORTED_MODELS_GEMINI
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
    
    # Validate API keys
    missing_keys = []
    if using_openai and not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if using_gemini and not os.environ.get("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")
    
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

def log_prompt(prompt, model_name, query_idx):
    """Log a completed prompt query."""
    log = set()
    if os.path.exists(QUERY_LOG_FILE):
        with open(QUERY_LOG_FILE, 'r') as f:
            log = set(json.load(f))
    
    prompt_hash = get_prompt_hash(prompt)
    log.add(prompt_hash)
    
    os.makedirs(os.path.dirname(QUERY_LOG_FILE), exist_ok=True)
    with open(QUERY_LOG_FILE, 'w') as f:
        json.dump(list(log), f, indent=2)

def process_results(query_results, prompt_list, solution_list, parameter_list, type_list, index_list):
    """Process query results and generate evaluation results."""
    full_results = []
    results = []
    
    # Process each result
    for response, prompt_idx, model_name, error, query_idx in query_results:
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx} (query {query_idx}): {response}")
            continue
            
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
                "model_latex_solution": eval_result.model_result.extracted_solutions,
                "solution_latex": eval_result.solution_result.extracted_solutions,
                "model_eval_result": eval_result.model_result.evaluation_results,
                "solution_eval_result": eval_result.solution_result.evaluation_results
            })
        except Exception as e:
            print(f"Error serializing eval result for prompt {prompt_idx} and model {model_name} (query {query_idx}): {e}")
            continue
    
    return full_results, results

def main():
    # Load and validate configuration
    config = load_config()
    selected_models = validate_models(config)
    
    # Get number of queries per prompt from config, default to 1
    num_queries = config.get('num_queries', 1)
    print(f"Running evaluation with models: {selected_models}")
    print(f"Number of queries per prompt: {num_queries}")
    
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]

    # Create a list of (prompt, prompt_idx, query_idx) tuples for each query
    prompt_tuples = []
    for idx, prompt in enumerate(prompt_list):
        for query_idx in range(num_queries):
            prompt_tuples.append((prompt, idx, query_idx))

    # Run queries with repeated prompts
    query_results = asyncio.run(bulk_query_with_progress([p[0] for p in prompt_tuples], selected_models))
    print(f"Number of query results: {len(query_results)}")


    # Update prompt_idx in results to match original indices and add query_idx
    updated_results = []
    for i, (response, _, model_name, error) in enumerate(query_results):
        # Calculate prompt_idx and query_idx based on the result's position
        total_queries_per_prompt = len(selected_models) * num_queries
        prompt_idx = i // total_queries_per_prompt
        query_idx = (i % total_queries_per_prompt) // len(selected_models)
        updated_results.append((response, prompt_idx, model_name, error, query_idx))
        
        # Log successful query
        if not error:
            log_prompt(prompt_list[prompt_idx], model_name, query_idx)
    
    print(f"Number of updated results: {len(updated_results)}")

    try:
        os.makedirs("results", exist_ok=True)
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
        with open("results/query_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
    except Exception as e:
        print(f"Error saving query results: {e}")
    # Process results
    full_results, results = process_results(
        updated_results, 
        prompt_list, 
        solution_list, 
        parameter_list, 
        type_list, 
        index_list
    )

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save results
    with open("results/full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to results.json")
    print(f"Saved {len(full_results)} results to full_results.json")

if __name__ == "__main__":
    main()
    