import json
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from typing import Dict, Any, List
import re

from matplotlib import pyplot as plt

def main():
    with open('results/results.json', 'r') as f:
        results = json.load(f)

    # Group results by model name
    results_by_model = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in results_by_model:
            results_by_model[model_name] = []
        results_by_model[model_name].append(result)

    # Calculate success rates for each model
    success_rates = {}
    for model_name, results in results_by_model.items():
        success_count = sum(1 for result in results if result['is_equivalent'])
        total_count = len(results)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        success_rates[model_name] = success_rate

    # Create bar chart
    models = list(results_by_model.keys())
    success_rates = [success_rates[model] for model in models]

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'  # Use serif font (optional)
    plt.rcParams['font.size'] = 18
    colors = ['#64CC33','#33BACC','#FF405E']
    plt.figure(figsize=(7, 5))
    plt.bar(models, success_rates, color='#33BACC')
    plt.ylabel('Success Rate (\%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 100)
    # Add percentage labels on top of each bar
    for i, v in enumerate(success_rates):
        plt.text(i, v + 1, f'{v:.1f}%', 
                ha='center', va='bottom')

    plt.savefig('results/success_rate_chart.png')
    plt.close()

    print("Success rate chart saved as success_rate_chart.png")

    # Calculate and save summary statistics
    summary_stats = {}
    for model_name, model_results in results_by_model.items():
        total_questions = len(model_results)
        correct_answers = sum(1 for r in model_results if r['is_equivalent'])
        summary_stats[model_name] = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "success_rate": f"{(correct_answers/total_questions)*100:.1f}%"
        }

    with open('results/summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)

    # Create pie chart of problem types
    problem_types = {}
    for result in results:
        prob_type = result['type']
        if prob_type not in problem_types:
            problem_types[prob_type] = 0
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
