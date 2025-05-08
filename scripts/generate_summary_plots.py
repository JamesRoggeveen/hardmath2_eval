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

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use serif font (optional)
plt.rcParams['font.size'] = 18

def generate_plots(summary_stats: Dict[str, Any]):
    """Generate all plots for the evaluation results using data from summary.json."""
    models = summary_stats["models"]
    prompt_success_rates = summary_stats["prompt_success_rates"]
    dataset_distribution = summary_stats["dataset_distribution"]

    # Calculate the number of queries per prompt (n)
    queries_per_prompt = max(
        int(round(summary_stats[model]["total_queries"] / summary_stats[model]["total_prompts"]))
        for model in models
    )

    # Create a heatmap of success rates
    prompt_indices = sorted(set().union(*[set(rates.keys()) for rates in prompt_success_rates.values()]))
    
    # Create the data matrix for the heatmap
    data = np.zeros((len(models), len(prompt_indices)))
    for i, model in enumerate(models):
        for j, prompt_idx in enumerate(prompt_indices):
            data[i, j] = prompt_success_rates[model].get(prompt_idx, 0)

    plt.figure(figsize=(15, 8))
    plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(label='Success Rate (\%)')
    
    # Add labels
    plt.xticks(range(len(prompt_indices)), [f'Prompt {idx}' for idx in prompt_indices], rotation=45)
    plt.yticks(range(len(models)), models)
    plt.xlabel('Prompt Index')
    plt.ylabel('Model')
    plt.title('Success Rate by Prompt and Model')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(prompt_indices)):
            plt.text(j, i, f'{data[i, j]:.1f}\%', 
                    ha='center', va='center',
                    color='black' if data[i, j] < 50 else 'white')

    plt.tight_layout()
    plt.savefig('results/success_rate_heatmap.png')
    plt.close()

    print("Success rate heatmap saved as success_rate_heatmap.png")

    # Create pie chart of problem types using data from summary
    plt.figure(figsize=(10, 8))
    plt.pie(dataset_distribution["problem_type_percentages"].values(), 
            labels=[t.replace('_', ' ').title() for t in dataset_distribution["problem_type_percentages"].keys()],
            autopct='%1.1f\%%',
            startangle=90)
    plt.axis('equal')
    plt.tight_layout(pad=1.5)  # Add padding around the plot
    plt.savefig('results/problem_distribution.png', bbox_inches='tight')  # Save with tight bounding box
    plt.close()

    print("Problem distribution chart saved as problem_distribution.png")

    # Create success rates chart
    plt.figure(figsize=(15, 8))
    
    # Get question types and add "Overall" as the first type
    question_types = ["Overall"] + sorted(dataset_distribution["problem_type_counts"].keys())
    x = np.arange(len(models))
    width = 0.8 / len(question_types)
    
    # Plot bars for each question type
    for i, q_type in enumerate(question_types):
        if q_type == "Overall":
            rates = [summary_stats[model]["overall_success_rate"] for model in models]
        else:
            rates = [summary_stats[model]["question_type_breakdown"][q_type]["success_rate"] 
                    for model in models]
        plt.bar(x + i*width - 0.4 + width/2, rates, width, 
                label=q_type.replace('_', ' ').title())
    
    plt.ylabel('Success Rate (\%)')
    plt.title('Success Rate by Model and Question Type')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/success_rates.png', bbox_inches='tight')
    plt.close()

    print("Success rates chart saved as success_rates.png")

    # Create pass@1 rates chart
    plt.figure(figsize=(15, 8))
    
    # Plot bars for each question type
    for i, q_type in enumerate(question_types):
        if q_type == "Overall":
            rates = [summary_stats[model]["overall_pass_at_1_rate"] for model in models]
        else:
            rates = [summary_stats[model]["question_type_breakdown"][q_type]["pass_at_1_rate"] 
                    for model in models]
        plt.bar(x + i*width - 0.4 + width/2, rates, width, 
                label=q_type.replace('_', ' ').title())
    
    plt.ylabel('Pass@1 Rate (\%)')
    plt.title('Pass@1 Rate by Model and Question Type')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pass_at_1_rates.png', bbox_inches='tight')
    plt.close()

    print("Pass@1 rates chart saved as pass_at_1_rates.png")

    # Create pass@n rates chart
    plt.figure(figsize=(15, 8))
    
    # Plot bars for each question type
    for i, q_type in enumerate(question_types):
        if q_type == "Overall":
            rates = [summary_stats[model]["overall_pass_at_n_rate"] for model in models]
        else:
            rates = [summary_stats[model]["question_type_breakdown"][q_type]["pass_at_n_rate"] 
                    for model in models]
        plt.bar(x + i*width - 0.4 + width/2, rates, width, 
                label=q_type.replace('_', ' ').title())
    
    plt.ylabel(f'Pass@{queries_per_prompt} Rate (\%)')
    plt.title(f'Pass@{queries_per_prompt} Rate by Model and Question Type')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pass_at_n_rates.png', bbox_inches='tight')
    plt.close()

    print(f"Pass@{queries_per_prompt} rates chart saved as pass_at_n_rates.png")

def main():
    # Check if summary.json exists, if not generate it first
    if not os.path.exists('results/summary.json'):
        from generate_summary import generate_summary
        generate_summary()
    
    # Read summary from JSON file
    with open('results/summary.json', 'r') as f:
        summary_stats = json.load(f)

    # Generate all plots
    generate_plots(summary_stats)

if __name__ == "__main__":
    main() 