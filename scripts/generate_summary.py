import json
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from typing import Dict, Any, List
import re
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np

def main():
    with open('results/results.json', 'r') as f:
        results = json.load(f)

    # Group results by model name and prompt_idx
    results_by_model = defaultdict(lambda: defaultdict(list))
    for result in results:
        model_name = result['model_name']
        prompt_idx = result['prompt_idx']
        results_by_model[model_name][prompt_idx].append(result)

    # Calculate success rates for each model and prompt
    prompt_success_rates = defaultdict(dict)
    for model_name, prompt_results in results_by_model.items():
        for prompt_idx, queries in prompt_results.items():
            successful_queries = sum(1 for query in queries if query['is_equivalent'])
            success_rate = (successful_queries / len(queries)) * 100
            prompt_success_rates[model_name][prompt_idx] = success_rate

    # Create a heatmap of success rates
    models = list(results_by_model.keys())
    prompt_indices = sorted(set().union(*[set(r.keys()) for r in prompt_success_rates.values()]))
    
    # Create the data matrix for the heatmap
    data = np.zeros((len(models), len(prompt_indices)))
    for i, model in enumerate(models):
        for j, prompt_idx in enumerate(prompt_indices):
            data[i, j] = prompt_success_rates[model].get(prompt_idx, 0)

    plt.figure(figsize=(15, 8))
    plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(label='Success Rate (%)')
    
    # Add labels
    plt.xticks(range(len(prompt_indices)), [f'Prompt {idx}' for idx in prompt_indices], rotation=45)
    plt.yticks(range(len(models)), models)
    plt.xlabel('Prompt Index')
    plt.ylabel('Model')
    plt.title('Success Rate by Prompt and Model')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(prompt_indices)):
            plt.text(j, i, f'{data[i, j]:.1f}%', 
                    ha='center', va='center',
                    color='black' if data[i, j] < 50 else 'white')

    plt.tight_layout()
    plt.savefig('results/success_rate_heatmap.png')
    plt.close()

    print("Success rate heatmap saved as success_rate_heatmap.png")

    # Calculate and save summary statistics
    summary_stats = {}
    for model_name, prompt_results in results_by_model.items():
        total_prompts = len(prompt_results)
        total_queries = sum(len(queries) for queries in prompt_results.values())
        total_successful_queries = sum(
            sum(1 for query in queries if query['is_equivalent'])
            for queries in prompt_results.values()
        )
        
        # Calculate success rate for each prompt
        prompt_stats = {}
        for prompt_idx, queries in prompt_results.items():
            successful_queries = sum(1 for query in queries if query['is_equivalent'])
            prompt_stats[f"prompt_{prompt_idx}"] = {
                "total_queries": len(queries),
                "successful_queries": successful_queries,
                "success_rate": f"{(successful_queries/len(queries))*100:.1f}"
            }

        summary_stats[model_name] = {
            "total_prompts": total_prompts,
            "total_queries": total_queries,
            "total_successful_queries": total_successful_queries,
            "overall_success_rate": f"{(total_successful_queries/total_queries)*100:.1f}%",
            "prompt_breakdown": prompt_stats
        }

    with open('results/summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)

    # Create pie chart of problem types
    problem_types = defaultdict(int)
    for result in results:
        prob_type = result['type']
        problem_types[prob_type] += 1

    # Convert to percentages
    total_problems = sum(problem_types.values())
    problem_type_percentages = {k: (v/total_problems)*100 for k,v in problem_types.items()}

    plt.figure(figsize=(10, 8))
    plt.pie(problem_type_percentages.values(), 
            labels=[t.replace('_', ' ').title() for t in problem_type_percentages.keys()],
            autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.tight_layout(pad=1.5)  # Add padding around the plot
    plt.savefig('results/problem_distribution.png', bbox_inches='tight')  # Save with tight bounding box
    plt.close()

    print("Summary statistics saved as summary.json")
    print("Problem distribution chart saved as problem_distribution.png")

if __name__ == "__main__":
    main()
