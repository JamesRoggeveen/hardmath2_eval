import json
import asyncio
import os

from benchmark_evaluator import bulk_query_with_progress, evaluate_solution
from datasets import load_dataset
from benchmark_evaluator.query import SUPPORTED_MODELS, SUPPORTED_MODELS_OPENAI, SUPPORTED_MODELS_GEMINI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"

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

def main():
    # Load and validate configuration
    config = load_config()
    selected_models = validate_models(config)
    
    print(f"Running evaluation with models: {selected_models}")
    
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]

    query_results = asyncio.run(bulk_query_with_progress(prompt_list, selected_models))

    full_results = []
    results = []
    for response, prompt_idx, model_name, error in query_results:
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx}: {error}")
            continue
        eval_result = evaluate_solution(response, solution_list[prompt_idx], parameter_list[prompt_idx])
        eval_result_serialized = eval_result.to_dict()
        full_results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt_list[prompt_idx],
            "model_name": model_name,
            "model_response": response,
            "eval_result": eval_result_serialized,
            "type": type_list[prompt_idx],
            "index": index_list[prompt_idx]
        })
        try:
            results.append(
                {
                    "prompt_idx": prompt_idx,
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
                }
            )
        except Exception as e:
            print(f"Error serializing eval result for prompt {prompt_idx} and model {model_name}: {e}")
            continue
        

    with open("results/full_results.json", "w") as f:
        json.dump(full_results, f)

    with open("results/results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
    