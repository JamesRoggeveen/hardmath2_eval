import json
import asyncio

from benchmark_evaluator import bulk_query_with_progress, evaluate_solution
from datasets import load_dataset

HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"

MODELS = ["GPT-4o", "Gemini 2.0 Flash Thinking", "Gemini 2.0 Flash"]

def main():
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]

    query_results = asyncio.run(bulk_query_with_progress(prompt_list, MODELS))

    results = []
    for response, prompt_idx, model_name, error in query_results:
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx}: {error}")
            continue

        eval_result = evaluate_solution(response, solution_list[prompt_idx], parameter_list[prompt_idx])
        eval_result_serialized = eval_result.to_dict()
        results.append({
            "prompt_idx": prompt_idx,
            "model_name": model_name,
            "model_response": response,
            "eval_result": eval_result_serialized,
            "type": type_list[prompt_idx],
            "index": index_list[prompt_idx]
        })

    with open("results/results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
    