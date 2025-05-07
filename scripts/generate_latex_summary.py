import json
import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from typing import Dict, Any, List
import re

# Define all possible models
ALL_MODELS = [
    "Gemini 2.0 Flash", "Gemini 2.0 Flash Thinking", "GPT-4o"
]

def get_models_from_results(results: List[Dict[str, Any]]) -> List[str]:
    """Extract unique models from the results."""
    models = set()
    for result in results:
        model = result["model_name"]
        models.add(model)
    return sorted(list(models))

def format_problem_type(problem_type: str) -> str:
    """Convert problem type from snake_case to Title Case with spaces."""
    return problem_type.replace('_', ' ').title()

def generate_latex_table(results: List[Dict[str, Any]]) -> str:
    """Generate a LaTeX table showing success/failure for each problem."""
    # Get models dynamically from results
    models = get_models_from_results(results)
    if not models:
        return "No model results found."
    
    # Create model labels (A, B, C, etc.)
    model_labels = {model: chr(65 + i) for i, model in enumerate(models)}
    
    # Start table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|l|" + "|c" * len(models) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Type} & \\textbf{Prompt} & " + " & ".join(f"\\textbf{{{label}}}" for label in model_labels.values()) + " \\\\\n"
    latex += "\\hline\n"
    
    # Group results by type and prompt_idx
    results_by_type = {}
    for result in results:
        problem_type = result.get('type', 'Unknown')
        prompt_idx = result.get('prompt_idx', 0)
        
        if problem_type not in results_by_type:
            results_by_type[problem_type] = {}
        if prompt_idx not in results_by_type[problem_type]:
            results_by_type[problem_type][prompt_idx] = {}
            
        model_name = result.get('model_name')
        is_equivalent = result.get('is_equivalent', False)
        results_by_type[problem_type][prompt_idx][model_name] = is_equivalent
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            row = f"{format_problem_type(problem_type)} & {prompt_idx}"
            
            # Add success/failure for each model
            for model in models:
                is_correct = results_by_type[problem_type][prompt_idx].get(model, False)
                if is_correct:
                    row += " & \\cellcolor{green!25}\\textcolor{green!25}{$\\checkmark$}"  # Success
                else:
                    row += " & \\cellcolor{red!25}\\textcolor{red!25}{$\\times$}"  # Failure
            
            row += " \\\\\n\\hline\n"
            latex += row
    
    # End table
    latex += "\\end{tabular}\n"
    latex += "\\caption{Model Performance Summary}\n"
    
    # Add key with model names on separate lines
    latex += "\\begin{center}\n"
    latex += "\\begin{tabular}{ll}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Label} & \\textbf{Model} \\\\\n"
    latex += "\\hline\n"
    for model, label in model_labels.items():
        latex += f"{label} & {model} \\\\[0.5em]\n"
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{center}\n"
    
    latex += "\\end{table}\n"
    return latex

def generate_full_results_summary(results: List[Dict[str, Any]]) -> str:
    """Generate detailed summary of each problem with full model outputs."""
    latex = ""
    
    # Group results by type and prompt_idx
    results_by_type = {}
    for result in results:
        problem_type = result.get('type', 'Unknown')
        prompt_idx = result.get('prompt_idx', 0)
        
        if problem_type not in results_by_type:
            results_by_type[problem_type] = {}
        if prompt_idx not in results_by_type[problem_type]:
            results_by_type[problem_type][prompt_idx] = []
            
        results_by_type[problem_type][prompt_idx].append(result)
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        latex += f"\\section*{{{format_problem_type(problem_type)}}}\n"
        
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            latex += f"\\subsection*{{Prompt {prompt_idx}}}\n"
            
            # Add the prompt
            prompt = results_by_type[problem_type][prompt_idx][0].get('prompt', '')
            latex += "\\noindent\\textbf{{Prompt}}:\\par\n"
            latex += f"\\noindent {prompt}\\par\n\\vspace{{1em}}\n\n"
            
            # Add all model attempts
            for result in results_by_type[problem_type][prompt_idx]:
                model = result.get('model_name', 'Unknown')
                eval_result = result.get('eval_result', {})
                is_correct = eval_result.get('is_equivalent', False)
                status = "\\textcolor{green}{$\\checkmark$}" if is_correct else "\\textcolor{red}{$\\times$}"
                
                # Model header with explicit spacing
                latex += f"\\noindent \\large {status} \\textbf{{{model}}} \\normalsize\n"
                latex += "\\vspace{1em}\n\n"
                
                # Add full model response with proper formatting
                model_response = result.get('model_response', '')
                if model_response:
                    latex += "\\noindent\\textbf{{Full Model Response}}:\\par\n"
                    # Escape special LaTeX characters and preserve whitespace
                    model_response = model_response.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')
                    latex += f"\\noindent {model_response}\\par\n\\vspace{{1em}}\n\n"
                
                # Add model solution
                model_solution = eval_result.get('model', {}).get('extracted_solutions', [])
                if model_solution:
                    latex += "\\noindent\\textbf{Model Solution}:\\par\n"
                    for solution in model_solution:
                        latex += "\\[\n" + solution + "\\]\n"
                    latex += "\n"
                
                # Add expected solution
                expected_solution = eval_result.get('solution', {}).get('extracted_solutions', [])
                if expected_solution:
                    latex += "\\noindent\\textbf{Expected Solution}:\\par\n"
                    for solution in expected_solution:
                        latex += "\\[\n" + solution + "\\]\n"
                    latex += "\n"
                
                # Add evaluation results
                model_eval = eval_result.get('model', {}).get('evaluation_results', [])
                solution_eval = eval_result.get('solution', {}).get('evaluation_results', [])
                if model_eval and solution_eval:
                    latex += "\\noindent\\textbf{Evaluation Results}:\\par\n"
                    latex += f"Model: {model_eval}, Expected: {solution_eval}\n\n"
                
                # Add extra space between models
                latex += "\\vspace{1em}\n\n"
    
    return latex

def generate_detailed_summary(results: List[Dict[str, Any]]) -> str:
    """Generate detailed summary of each problem."""
    latex = ""
    
    # Group results by type and prompt_idx
    results_by_type = {}
    for result in results:
        problem_type = result.get('type', 'Unknown')
        prompt_idx = result.get('prompt_idx', 0)
        
        if problem_type not in results_by_type:
            results_by_type[problem_type] = {}
        if prompt_idx not in results_by_type[problem_type]:
            results_by_type[problem_type][prompt_idx] = []
            
        results_by_type[problem_type][prompt_idx].append(result)
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        latex += f"\\section*{{{format_problem_type(problem_type)}}}\n"
        
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            latex += f"\\subsection*{{Prompt {prompt_idx}}}\n"
            
            # Add the prompt from the first result for this type and prompt_idx
            first_result = results_by_type[problem_type][prompt_idx][0]
            prompt = first_result['prompt']  # Direct access since we know it exists
            latex += "\\noindent\\textbf{{Prompt}}:\\par\n"
            latex += f"\\noindent {prompt}\\par\n\\vspace{{1em}}\n\n"
            
            # Add all model attempts
            for result in results_by_type[problem_type][prompt_idx]:
                model = result.get('model_name', 'Unknown')
                is_correct = result.get('is_equivalent', False)
                status = "\\textcolor{green}{$\\checkmark$}" if is_correct else "\\textcolor{red}{$\\times$}"
                
                # Model header with explicit spacing
                latex += f"\\noindent \\large {status} \\textbf{{{model}}} \\normalsize\n"
                latex += "\\vspace{1em}\n\n"
                
                # Add model solution
                model_solution = result.get('model_latex_solution', '')
                if model_solution:
                    latex += "\\noindent\\textbf{Model Solution}:\\par\n"
                    for solution in model_solution:
                        latex += "\\[\n" + solution + "\\]\n"
                    latex += "\n"
                
                # Add expected solution
                expected_solution = result.get('solution_latex', '')
                if expected_solution:
                    latex += "\\noindent\\textbf{Expected Solution}:\\par\n"
                    for solution in expected_solution:
                        latex += "\\[\n" + solution + "\\]\n"
                    latex += "\n"
                
                # Add evaluation results
                model_eval = result.get('model_eval_result', '')
                solution_eval = result.get('solution_eval_result', '')
                if model_eval and solution_eval:
                    latex += "\\noindent\\textbf{Evaluation Results}:\\par\n"
                    latex += f"Model: {model_eval}, Expected: {solution_eval}\n\n"
                
                # Add extra space between models
                latex += "\\vspace{1em}\n\n"
    
    return latex

def main():
    # Read results from both JSON files
    results_dir = os.path.join(project_root, "results")
    
    # Read summary results
    summary_path = os.path.join(results_dir, "results.json")
    with open(summary_path, 'r') as f:
        summary_results = json.load(f)
    
    # Read full results
    full_path = os.path.join(results_dir, "full_results.json")
    with open(full_path, 'r') as f:
        full_results = json.load(f)
    
    # Generate LaTeX documents
    latex_dir = os.path.join(project_root, "results")
    os.makedirs(latex_dir, exist_ok=True)
    
    # Generate summary document
    summary_latex = "\\documentclass{article}\n"
    summary_latex += "\\usepackage{amsmath}\n"
    summary_latex += "\\usepackage{graphicx}\n"
    summary_latex += "\\usepackage{amssymb}\n"
    summary_latex += "\\usepackage[table]{xcolor}\n"
    summary_latex += "\\usepackage{booktabs}\n"
    summary_latex += "\\usepackage{hyperref}\n"
    summary_latex += "\\usepackage{geometry}\n"
    summary_latex += "\\geometry{margin=1in}\n"
    summary_latex += "\\begin{document}\n\n"
    
    summary_latex += generate_latex_table(summary_results)
    summary_latex += "\n\\newpage\n"
    summary_latex += generate_detailed_summary(summary_results)
    
    summary_latex += "\\end{document}\n"
    
    with open(os.path.join(latex_dir, "llm_summary.tex"), 'w') as f:
        f.write(summary_latex)
    
    # Generate full results document
    full_latex = "\\documentclass{article}\n"
    full_latex += "\\usepackage{amsmath}\n"
    full_latex += "\\usepackage{amssymb}\n"
    full_latex += "\\usepackage{graphicx}\n"
    full_latex += "\\usepackage[table]{xcolor}\n"
    full_latex += "\\usepackage{booktabs}\n"
    full_latex += "\\usepackage{hyperref}\n"
    full_latex += "\\usepackage{geometry}\n"
    full_latex += "\\geometry{margin=1in}\n"
    full_latex += "\\begin{document}\n\n"
    
    # Add table to full results using the same table generation function
    full_latex += generate_latex_table(summary_results)
    full_latex += "\n\\newpage\n"
    full_latex += generate_full_results_summary(full_results)
    
    full_latex += "\\end{document}\n"
    
    with open(os.path.join(latex_dir, "llm_full_results.tex"), 'w') as f:
        f.write(full_latex)

if __name__ == "__main__":
    main() 