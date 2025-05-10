import json
import os
import sys
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
    latex += "\\usepackage{pdflscape}\n"  # For landscape orientation
    latex += "\\usepackage{array}\n"  # For better table formatting
    latex += "\\geometry{landscape,margin=0.75in}\n"  # Landscape orientation with smaller margins
    latex += "\\begin{document}\n\n"
    latex += "\\small\n"  # Smaller font size for the whole document
    
    # Generate pass@1 rates table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\setlength{\\tabcolsep}{4pt}\n"  # Reduce column separation
    
    # Table header
    latex += "\\begin{tabular}{|l|" + "|c" * (len(question_types) * 2) + "|}\n"
    latex += "\\hline\n"
    
    # Main header row with shorter problem type names
    latex += "\\textbf{Model}"
    for qt in question_types:
        # Shorter names for column headers
        short_name = format_problem_type(qt)
        if short_name.lower() != "overall":
            # Abbreviate longer names
            short_name = short_name.replace("Asympytotic", "Asymp.")
            short_name = short_name.replace("Boundary", "Bound.")
            short_name = short_name.replace("Nonlinear", "Nonlin.")
        latex += f" & \\multicolumn{{2}}{{c|}}{{\\textbf{{{short_name}}}}}"
    latex += " \\\\\n"
    
    # Subheader row with Rate and Count for each question type
    latex += "\\cline{2-" + str(len(question_types) * 2 + 1) + "}\n"
    latex += " "
    for _ in question_types:
        latex += " & \\textbf{Rate} & \\textbf{Count}"  # Shorter column headers
    latex += " \\\\\n"
    latex += "\\hline\n"
    
    # Generate rows for each model
    for model in models:
        # Shorten model names for better fit
        display_model = model.replace("Gemini 2.5 Flash Thinking", "Gem 2.5 Flash Think")
        display_model = display_model.replace("Gemini 2.5 Pro Preview", "Gem 2.5 Pro")
        display_model = display_model.replace("Gemini 2.0 Flash", "Gem 2.0 Flash")
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
    latex += "\\caption{Pass@1 Rates and Problem Counts by Model and Question Type}\n"
    latex += "\\end{table}\n"
    latex += "\\end{document}\n"
    
    return latex

def main():
    # Read summary statistics
    results_dir = os.path.join(project_root, "results")
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