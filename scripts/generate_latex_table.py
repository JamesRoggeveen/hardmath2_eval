import json
import os
import sys
import argparse
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from typing import Dict, Any, List

def format_problem_type(problem_type: str) -> str:
    """Convert problem type from snake_case to Title Case with spaces."""
    return problem_type.replace('_', ' ').title()

def generate_pass_at_1_with_counts(summary_stats: Dict[str, Any]) -> str:
    """Generate a LaTeX table showing pass@1 rates with problem counts for each model and question type."""
    models = summary_stats["models"]
    if not models:
        return "No model results found."
    
    # Get all question types including "Overall"
    question_types = ["Overall"] + sorted(summary_stats["question_types"])
    
    latex = "\\documentclass{article}\n"
    latex += "\\usepackage{amsmath}\n"
    latex += "\\usepackage{booktabs}\n"
    latex += "\\usepackage[table]{xcolor}\n"
    latex += "\\usepackage{geometry}\n"
    latex += "\\usepackage{graphicx}\n"  # For resizebox
    latex += "\\geometry{margin=1in}\n"
    latex += "\\begin{document}\n\n"
    
    # Generate pass@1 rates table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"  # Resize table to fit page width
    
    # Table header
    latex += "\\begin{tabular}{|l|" + "|c" * (len(question_types) * 2) + "|}\n"
    latex += "\\hline\n"
    
    # Main header row
    latex += "\\textbf{Model}"
    for qt in question_types:
        latex += f" & \\multicolumn{{2}}{{c|}}{{\\textbf{{{format_problem_type(qt)}}}}}"
    latex += " \\\\\n"
    
    # Subheader row with Rate and Count for each question type
    latex += "\\cline{2-" + str(len(question_types) * 2 + 1) + "}\n"
    latex += " "
    for _ in question_types:
        latex += " & \\textbf{Rate} & \\textbf{Count}"
    latex += " \\\\\n"
    latex += "\\hline\n"
    
    # Generate rows for each model
    for model in models:
        # Fix model display names
        display_model = model
        
        # Special handling for o1, o3, o3-mini, o4-mini models - remove "GPT-" prefix
        if model.startswith("GPT-o"):
            # For o1, o3, etc. models, remove the GPT- prefix
            display_model = model[4:]  # Remove "GPT-" prefix
        
        row = f"{display_model}"
        
        # Add overall pass@1 rate and count
        overall_rate = summary_stats[model]["overall_pass_at_1_rate"]
        total_prompts = summary_stats[model]["total_prompts"]
        row += f" & {overall_rate:.1f} & {total_prompts}"
        
        # Add pass@1 rates and counts for each question type
        for q_type in question_types[1:]:  # Skip "Overall" as we already handled it
            q_type_stats = summary_stats[model]["question_type_breakdown"].get(q_type, {})
            pass_at_1_rate = q_type_stats.get("pass_at_1_rate", 0)
            total_prompts = q_type_stats.get("total_prompts", 0)
            row += f" & {pass_at_1_rate:.1f} & {total_prompts}"
        
        row += " \\\\\n\\hline\n"
        latex += row
    
    latex += "\\end{tabular}\n"
    latex += "}\n"  # Close resizebox
    latex += "\\caption{Pass@1 Rates and Problem Counts by Model and Question Type}\n"
    latex += "\\end{table}\n"
    latex += "\\end{document}\n"
    
    return latex


def main():

    parser = argparse.ArgumentParser(
    description="Generate a LaTeX pass@1 table from summary statistics."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Folder holding summary.json (default: results). "
             "You can pass a sibling like 'results_test' or an absolute path."
    )
    args = parser.parse_args()

    # Read summary statistics
    results_dir = (
        args.results_dir                                    # absolute path → use as-is
        if os.path.isabs(args.results_dir)
        else os.path.join(project_root, args.results_dir)   # relative → under project_root
    )
    
    summary_path = os.path.join(results_dir, "summary.json")
    
    # Check if summary.json exists, if not generate it first
    if not os.path.exists(summary_path):
        from generate_summary import generate_summary
        generate_summary()
    
    # Read summary statistics
    with open(summary_path, 'r') as f:
        summary_stats = json.load(f)
    
    # Generate the LaTeX table
    latex = generate_pass_at_1_with_counts(summary_stats)
    
    # Write to file
    output_path = os.path.join(results_dir, "pass_at_1_with_counts.tex")
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Generated pass@1 rates table with counts at: {output_path}")

if __name__ == "__main__":
    main() 