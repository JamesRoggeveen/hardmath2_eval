import json
from benchmark_evaluator import evaluate_solution
from datasets import load_dataset
import os
import tqdm

def process_results(query_results, prompt_list, solution_list, parameter_list, type_list, index_list, skip_list):
    """Process query results and generate evaluation results."""
    full_results = []
    results = []
    
    # Process each result
    for i,(response, prompt_idx, model_name, error, query_idx) in enumerate(query_results):
        print(f"Begin processing number {i}, model {model_name} at prompt_idx {prompt_idx}")
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx} (query {query_idx}): {response}")
            continue

        if prompt_idx in skip_list:
            print(f"Skipping prompt {prompt_idx}")
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
            model_eval_results = eval_result.model_result.evaluation_results
            model_eval_results_serialized = []
            for value in model_eval_results:
                if isinstance(value,complex):
                    model_eval_results_serialized.append(str(value))
                else:
                    model_eval_results_serialized.append(value)
            solution_eval_results = eval_result.solution_result.evaluation_results
            solution_eval_results_serialized = []
            for value in solution_eval_results:
                if isinstance(value,complex):
                    solution_eval_results_serialized.append(str(value))
                else:
                    solution_eval_results_serialized.append(value)

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
                "model_eval_result": model_eval_results_serialized,
                "solution_eval_result": solution_eval_results_serialized
            })
        except Exception as e:
            print(f"Error serializing for prompt {prompt_idx} and model {model_name} (query {query_idx}): {e}")
            continue
    
    return full_results, results


if __name__ == "__main__":
    with open('results/query_results.json', 'r') as f:
        results = json.load(f)

    updated_results = []
    for result in results:
        prompt_idx = result["prompt_idx"]
        model_name = result["model_name"]
        response = result["response"]
        query_idx = result["query_idx"]
        error = result["error"]
        updated_results.append((response, prompt_idx, model_name, error, query_idx))

    print(f"Number of results loaded: {len(updated_results)}")

    HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train", cache_dir=".cache")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]

    skip_indicies = [0,56,168,195,205]
    # bad_sols = []
    # start_idx = 0
    # for i in range(start_idx, len(prompt_list)):
    #     if i in skip_indicies:
    #         continue
    #     print(f"Evaluating solution {i}")
    #     eval_sol = evaluate_solution(solution_list[i],parameter_list[i])
    #     if eval_sol.success == False:
    #         bad_sols.append(i)
    #     print(f"Solution success: {eval_sol.success}")
    # print(bad_sols)
    full_results, results = process_results(
        updated_results, 
        prompt_list, 
        solution_list, 
        parameter_list, 
        type_list, 
        index_list,
        skip_indicies
    )

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save results
    with open("results/full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to results.json", flush=True)
    print(f"Saved {len(full_results)} results to full_results.json", flush=True)

    # model_evals = []
    # start_idx = 1850
    # for i in range(start_idx, len(results)):
    #     item = results[i]
    #     if item["error"] == True:
    #         continue
    #     model = item["model_name"]
    #     idx = item["prompt_idx"]
    #     print(f"Evaluating {model} on prompt {idx} at index {i}")
    #     response = item["response"]
    #     parameter_str = parameter_list[idx]
    #     eval_sol = evaluate_solution(response,parameter_str)
    #     eval_sol = eval_sol.to_dict()
    #     model_evals.append({
    #         "prompt": idx,
    #         "model": model,
    #         "eval": eval_sol
    #     })