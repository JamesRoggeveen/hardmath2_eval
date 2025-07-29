import json
from benchmark_evaluator import evaluate_solution, evaluate_with_llm_judge
from datasets import load_dataset
import os
import tqdm
import argparse

def process_results(query_results, prompt_list, solution_list, parameter_list, type_list, index_list, skip_list, *, use_llm_judge: bool = False, judge_model: str = "gpt-4o-mini"):
    """Process query results and generate evaluation results."""
    full_results = []
    results = []
    
    # Map problem types to rubric files (extend as needed)
    RUBRIC_MAP = {
        "boundary_layers": "rubrics/boundary_rubric.txt",
        "wkb": "rubrics/wkb_rubric.txt",
        "nonlinear_pde": "rubrics/nonlinear_pdes_rubric.txt"
    }

    # Process each result
    for i,(response, prompt_idx, model_name, error, query_idx) in enumerate(query_results):
        print(f"Begin processing number {i}, model {model_name} at prompt_idx {prompt_idx}")
        if error:
            print(f"Error querying {model_name} for prompt {prompt_idx} (query {query_idx}): {response}")
            continue

        if prompt_idx in skip_list:
            print(f"Skipping prompt {prompt_idx}")
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
    parser = argparse.ArgumentParser(
        description="Process query_results.json and produce evaluation outputs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing query_results.json and where outputs will be saved"
    )
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM-as-a-judge rubric scoring instead of numeric/symbolic evaluator")
    parser.add_argument("--problem-type", nargs="*", default=None, help="Filter to these problem types before evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of evaluations after filtering")
    args = parser.parse_args()
    results_dir = args.results_dir

    # Load the raw query results
    input_path = os.path.join(results_dir, "query_results.json")
    with open(input_path, "r") as f:
        raw_results = json.load(f)

    updated_results = []
    for result in raw_results:
        response = result["response"]
        prompt_idx = result["prompt_idx"]
        model_name = result["model_name"]
        error = result["error"]
        query_idx = result["query_idx"]
        updated_results.append((response, prompt_idx, model_name, error, query_idx))

    print(f"Number of results loaded: {len(updated_results)}")

    HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train", cache_dir=".cache")

    prompt_list = dataset["prompt"]
    solution_list = dataset["solution"]
    parameter_list = dataset["parameters"]
    type_list = dataset["type"]
    index_list = dataset["index"]

    skip_indices = [0, 56, 168, 195, 205]

    # Optional filtering by problem type
    if args.problem_type:
        types_str = ", ".join(args.problem_type)
        print(f"Filtering to problem types: {types_str}", flush=True)
        allowed = set(args.problem_type)
        mask = [t in allowed for t in type_list]
        if not any(mask):
            print("No entries with that problem type – exiting.")
            exit(0)
        # Re-filter all lists
        prompt_list  = [p for p, m in zip(prompt_list, mask) if m]
        solution_list= [s for s, m in zip(solution_list, mask) if m]
        parameter_list=[p for p, m in zip(parameter_list, mask) if m]
        type_list    = [t for t, m in zip(type_list, mask) if m]
        index_list   = [i for i, m in zip(index_list, mask) if m]
        # Also filter updated_results to matching prompt_idx
        allowed_idx = {i for i, m in enumerate(mask) if m}
        updated_results = [r for r in updated_results if r[1] in allowed_idx]

    # Apply limit to UNIQUE problems (prompt_idx) after filtering
    if args.limit > 0:
        seen = set()
        limited_results = []
        for item in updated_results:
            prompt_idx = item[1]
            if prompt_idx not in seen:
                if len(seen) >= args.limit:
                    break
                seen.add(prompt_idx)
            limited_results.append(item)
        updated_results = limited_results

    full_results, results = process_results(
        updated_results,
        prompt_list,
        solution_list,
        parameter_list,
        type_list,
        index_list,
        skip_indices,
        use_llm_judge=args.use_llm_judge,
        judge_model="gpt-4o-mini",
    )

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)

    suffix = "_llm" if args.use_llm_judge else ""

    full_path = os.path.join(results_dir, f"full_results{suffix}.json")
    res_path  = os.path.join(results_dir, f"results{suffix}.json")

    with open(full_path, "w") as f:
        json.dump(full_results, f, indent=2)

    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {res_path}", flush=True)
    print(f"Saved {len(full_results)} results to {full_path}", flush=True)

# if __name__ == "__main__":
#     with open('results/query_results.json', 'r') as f:
#         results = json.load(f)

#     updated_results = []
#     for result in results:
#         prompt_idx = result["prompt_idx"]
#         model_name = result["model_name"]
#         response = result["response"]
#         query_idx = result["query_idx"]
#         error = result["error"]
#         updated_results.append((response, prompt_idx, model_name, error, query_idx))

#     print(f"Number of results loaded: {len(updated_results)}")

#     HUGGINGFACE_DATASET_NAME = "AnonBenchmark5727/benchmark_data"
#     dataset = load_dataset(HUGGINGFACE_DATASET_NAME, split="train", cache_dir=".cache")

#     prompt_list = dataset["prompt"]
#     solution_list = dataset["solution"]
#     parameter_list = dataset["parameters"]
#     type_list = dataset["type"]
#     index_list = dataset["index"]

#     skip_indicies = [0,56,168,195,205]
#     # bad_sols = []
#     # start_idx = 0
#     # for i in range(start_idx, len(prompt_list)):
#     #     if i in skip_indicies:
#     #         continue
#     #     print(f"Evaluating solution {i}")
#     #     eval_sol = evaluate_solution(solution_list[i],parameter_list[i])
#     #     if eval_sol.success == False:
#     #         bad_sols.append(i)
#     #     print(f"Solution success: {eval_sol.success}")
#     # print(bad_sols)
#     full_results, results = process_results(
#         updated_results, 
#         prompt_list, 
#         solution_list, 
#         parameter_list, 
#         type_list, 
#         index_list,
#         skip_indicies
#     )

#     # Create results directory if it doesn't exist
#     os.makedirs("results", exist_ok=True)

#     # Save results
#     with open("results/full_results.json", "w") as f:
#         json.dump(full_results, f, indent=2)

#     with open("results/results.json", "w") as f:
#         json.dump(results, f, indent=2)

#     print(f"Saved {len(results)} results to results.json", flush=True)
#     print(f"Saved {len(full_results)} results to full_results.json", flush=True)

   