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
    """Generate LaTeX tables showing success rates for each problem."""
    # Get models dynamically from results
    models = get_models_from_results(results)
    if not models:
        return "No model results found."
    
    # Create model labels (A, B, C, etc.)
    model_labels = {model: chr(65 + i) for i, model in enumerate(models)}
    
    # Group results by type and prompt_idx
    results_by_type = {}
    for result in results:
        problem_type = result.get('type', 'Unknown')
        prompt_idx = result.get('prompt_idx', 0)
        model_name = result.get('model_name')
        is_equivalent = result.get('is_equivalent', False)
        query_idx = result.get('query_idx', 0)
        
        if problem_type not in results_by_type:
            results_by_type[problem_type] = {}
        if prompt_idx not in results_by_type[problem_type]:
            results_by_type[problem_type][prompt_idx] = {}
        if model_name not in results_by_type[problem_type][prompt_idx]:
            results_by_type[problem_type][prompt_idx][model_name] = []
            
        results_by_type[problem_type][prompt_idx][model_name].append(is_equivalent)
    
    # Generate one-shot success table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|l|" + "|c" * len(models) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Type} & \\textbf{Prompt} & " + " & ".join(f"\\textbf{{{label}}}" for label in model_labels.values()) + " \\\\\n"
    latex += "\\hline\n"
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            row = f"{format_problem_type(problem_type)} & {prompt_idx}"
            
            # Add success/failure for each model (all queries must be correct)
            for model in models:
                query_results = results_by_type[problem_type][prompt_idx].get(model, [])
                all_correct = all(query_results) if query_results else False
                if all_correct:
                    row += " & \\cellcolor{successgreen!25}\\textcolor{black}{$\\checkmark$}"  # Success
                else:
                    row += " & \\cellcolor{failurered!25}\\textcolor{black}{$\\times$}"  # Failure
            
            row += " \\\\\n\\hline\n"
            latex += row
    
    # End one-shot table
    latex += "\\end{tabular}\n"
    latex += "\\caption{One-Shot Success Rate (All Queries Correct)}\n"
    
    # Add key with model names
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
    latex += "\\end{table}\n\n"
    
    # Generate percentage success table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|l|" + "|c" * len(models) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Type} & \\textbf{Prompt} & " + " & ".join(f"\\textbf{{{label}}}" for label in model_labels.values()) + " \\\\\n"
    latex += "\\hline\n"
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            row = f"{format_problem_type(problem_type)} & {prompt_idx}"
            
            # Add percentage success for each model
            for model in models:
                query_results = results_by_type[problem_type][prompt_idx].get(model, [])
                if query_results:
                    success_rate = (sum(query_results) / len(query_results)) * 100
                    # Color based on success rate
                    if success_rate >= 75:
                        color = "successgreen!25"
                    elif success_rate >= 50:
                        color = "yellow!25"
                    else:
                        color = "failurered!25"
                    row += f" & \\cellcolor{{{color}}}\\textcolor{{black}}{{{success_rate:.0f}\\%}}"
                else:
                    row += " & \\cellcolor{failurered!25}\\textcolor{black}{N/A}"
            
            row += " \\\\\n\\hline\n"
            latex += row
    
    # End percentage table
    latex += "\\end{tabular}\n"
    latex += "\\caption{Percentage of Correct Queries}\n"
    
    # Add key with model names
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
            results_by_type[problem_type][prompt_idx] = {}
            
        model_name = result.get('model_name', 'Unknown')
        if model_name not in results_by_type[problem_type][prompt_idx]:
            results_by_type[problem_type][prompt_idx][model_name] = []
        results_by_type[problem_type][prompt_idx][model_name].append(result)
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        latex += f"\\section{{{format_problem_type(problem_type)}}}\n"
        
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            latex += f"\\subsection{{Prompt {prompt_idx}}}\n"
            
            # Add the full prompt
            first_model = next(iter(results_by_type[problem_type][prompt_idx].values()))
            first_result = first_model[0]
            latex += "\\noindent\\textbf{{Full Prompt}}:\\par\n"
            latex += f"\\noindent {first_result['prompt']}\\par\n\\vspace{{1em}}\n\n"
            
            # Add all model attempts
            for model_name, model_results in sorted(results_by_type[problem_type][prompt_idx].items()):
                # Sort results by query_idx
                model_results.sort(key=lambda x: x.get('query_idx', 0))
                
                # Model header
                latex += f"\\noindent \\large \\textbf{{{model_name}}} \\normalsize\n"
                latex += "\\vspace{0.5em}\n\n"
                
                # Add each query attempt
                for result in model_results:
                    query_idx = result.get('query_idx', 0)
                    is_correct = result.get('is_equivalent', False)
                    status = "\\textcolor{successgreen}{{$\\checkmark$}}" if is_correct else "\\textcolor{failurered}{{$\\times$}}"
                    
                    # Query attempt header
                    latex += f"\\noindent \\textbf{{Query {query_idx + 1}}} {status}\\par\n"
                    latex += "\\vspace{0.5em}\n\n"
                    
                    # Add full model response
                    model_response = result.get('model_response', '')
                    if model_response:
                        latex += "\\noindent\\textbf{{Full Model Response}}:\\par\n"
                        # Escape special LaTeX characters and preserve whitespace
                        model_response = (model_response
                            .replace('&', '\\&')
                            .replace('%', '\\%')
                            .replace('#', '\\#')
                            .replace('```', ''))  # Remove markdown code block delimiters
                        latex += f"\\noindent {model_response}\\par\n\\vspace{{1em}}\n\n"
                    
                    # Add model solution
                    model_solution = result.get('model_latex_solution', [])
                    if model_solution:
                        latex += "\\noindent\\textbf{{Model Solution}}:\\par\n"
                        for solution in model_solution:
                            # Clean up the solution and ensure it's properly formatted
                            solution = solution.strip()
                            if solution:
                                # Remove any markdown code block delimiters
                                solution = solution.replace('```', '')
                                latex += "\\begin{equation*}\n"
                                latex += solution + "\n"
                                latex += "\\end{equation*}\n\n"
                    
                    # Add expected solution
                    expected_solution = result.get('solution_latex', [])
                    if expected_solution:
                        latex += "\\noindent\\textbf{{Expected Solution}}:\\par\n"
                        for solution in expected_solution:
                            # Clean up the solution and ensure it's properly formatted
                            solution = solution.strip()
                            if solution:
                                # Remove any markdown code block delimiters
                                solution = solution.replace('```', '')
                                latex += "\\begin{equation*}\n"
                                latex += solution + "\n"
                                latex += "\\end{equation*}\n\n"
                    
                    # Add evaluation results
                    model_eval = result.get('model_eval_result', [])
                    solution_eval = result.get('solution_eval_result', [])
                    if model_eval and solution_eval:
                        latex += "\\noindent\\textbf{{Evaluation Results}}:\\par\n"
                        # Escape any special characters in evaluation results
                        model_eval = str(model_eval).replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
                        solution_eval = str(solution_eval).replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
                        latex += f"Model: {model_eval}, Expected: {solution_eval}\n\n"
                    
                    # Add extra space between queries
                    latex += "\\vspace{0.5em}\n\n"
                
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
            results_by_type[problem_type][prompt_idx] = {}
            
        model_name = result.get('model_name', 'Unknown')
        if model_name not in results_by_type[problem_type][prompt_idx]:
            results_by_type[problem_type][prompt_idx][model_name] = []
        results_by_type[problem_type][prompt_idx][model_name].append(result)
    
    # Sort types alphabetically
    for problem_type in sorted(results_by_type.keys()):
        latex += f"\\section{{{format_problem_type(problem_type)}}}\n"
        
        # Sort prompts within each type
        for prompt_idx in sorted(results_by_type[problem_type].keys()):
            latex += f"\\subsection{{Prompt {prompt_idx}}}\n"
            
            # Add the full prompt
            first_model = next(iter(results_by_type[problem_type][prompt_idx].values()))
            first_result = first_model[0]
            latex += "\\noindent\\textbf{{Full Prompt}}:\\par\n"
            latex += f"\\noindent {first_result['prompt']}\\par\n\\vspace{{1em}}\n\n"
            
            # Add all model attempts
            for model_name, model_results in sorted(results_by_type[problem_type][prompt_idx].items()):
                # Sort results by query_idx
                model_results.sort(key=lambda x: x.get('query_idx', 0))
                
                # Model header
                latex += f"\\noindent \\large \\textbf{{{model_name}}} \\normalsize\n"
                latex += "\\vspace{0.5em}\n\n"
                
                # Add each query attempt
                for result in model_results:
                    query_idx = result.get('query_idx', 0)
                    is_correct = result.get('is_equivalent', False)
                    status = "\\textcolor{successgreen}{{$\\checkmark$}}" if is_correct else "\\textcolor{failurered}{{$\\times$}}"
                    
                    # Query attempt header
                    latex += f"\\noindent \\textbf{{Query {query_idx + 1}}} {status}\\par\n"
                    latex += "\\vspace{0.5em}\n\n"
                    
                    # Add model solution
                    model_solution = result.get('model_latex_solution', [])
                    if model_solution:
                        latex += "\\noindent\\textbf{{Model Solution}}:\\par\n"
                        for solution in model_solution:
                            # Clean up the solution and ensure it's properly formatted
                            solution = solution.strip()
                            if solution:
                                # Remove any markdown code block delimiters
                                solution = solution.replace('```', '')
                                latex += "\\begin{equation*}\n"
                                latex += solution + "\n"
                                latex += "\\end{equation*}\n\n"
                    
                    # Add expected solution
                    expected_solution = result.get('solution_latex', [])
                    if expected_solution:
                        latex += "\\noindent\\textbf{{Expected Solution}}:\\par\n"
                        for solution in expected_solution:
                            # Clean up the solution and ensure it's properly formatted
                            solution = solution.strip()
                            if solution:
                                # Remove any markdown code block delimiters
                                solution = solution.replace('```', '')
                                latex += "\\begin{equation*}\n"
                                latex += solution + "\n"
                                latex += "\\end{equation*}\n\n"
                    
                    # Add evaluation results
                    model_eval = result.get('model_eval_result', [])
                    solution_eval = result.get('solution_eval_result', [])
                    if model_eval and solution_eval:
                        latex += "\\noindent\\textbf{{Evaluation Results}}:\\par\n"
                        # Escape any special characters in evaluation results
                        model_eval = str(model_eval).replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
                        solution_eval = str(solution_eval).replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
                        latex += f"Model: {model_eval}, Expected: {solution_eval}\n\n"
                    
                    # Add extra space between queries
                    latex += "\\vspace{0.5em}\n\n"
                
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
    summary_latex += "\\usepackage{tocloft}\n"  # For better TOC control
    summary_latex += "\\geometry{margin=1in}\n"
    summary_latex += "\\definecolor{successgreen}{RGB}{0,128,0}\n"  # Define success green color
    summary_latex += "\\definecolor{failurered}{RGB}{255,0,0}\n"    # Define failure red color
    summary_latex += "\\hypersetup{\n"
    summary_latex += "    colorlinks=true,\n"
    summary_latex += "    linkcolor=blue,\n"
    summary_latex += "    filecolor=magenta,\n"
    summary_latex += "    urlcolor=cyan,\n"
    summary_latex += "    pdftitle={LLM Evaluation Results},\n"
    summary_latex += "    pdfpagemode=FullScreen,\n"
    summary_latex += "}\n"
    summary_latex += "\\setcounter{tocdepth}{2}\n"  # Set TOC depth to include sections and subsections
    summary_latex += "\\renewcommand{\\cftsecleader}{\\cftdotfill{\\cftdotsep}}\n"  # Add dots to TOC
    summary_latex += "\\begin{document}\n\n"
    
    # Add title and table of contents
    summary_latex += "\\title{LLM Evaluation Results}\n"
    summary_latex += "\\author{Generated Report}\n"
    summary_latex += "\\date{\\today}\n"
    summary_latex += "\\maketitle\n\n"
    summary_latex += "\\tableofcontents\n"
    summary_latex += "\\newpage\n\n"
    
    # Add section for tables
    summary_latex += "\\section{Performance Tables}\n"
    summary_latex += generate_latex_table(summary_results)
    summary_latex += "\n\\newpage\n"
    
    # Add section for detailed results
    summary_latex += "\\section{Detailed Results}\n"
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
    full_latex += "\\usepackage{tocloft}\n"  # For better TOC control
    full_latex += "\\geometry{margin=1in}\n"
    full_latex += "\\definecolor{successgreen}{RGB}{0,128,0}\n"  # Define success green color
    full_latex += "\\definecolor{failurered}{RGB}{255,0,0}\n"    # Define failure red color
    full_latex += "\\hypersetup{\n"
    full_latex += "    colorlinks=true,\n"
    full_latex += "    linkcolor=blue,\n"
    full_latex += "    filecolor=magenta,\n"
    full_latex += "    urlcolor=cyan,\n"
    full_latex += "    pdftitle={LLM Full Evaluation Results},\n"
    full_latex += "    pdfpagemode=FullScreen,\n"
    full_latex += "}\n"
    full_latex += "\\setcounter{tocdepth}{2}\n"  # Set TOC depth to include sections and subsections
    full_latex += "\\renewcommand{\\cftsecleader}{\\cftdotfill{\\cftdotsep}}\n"  # Add dots to TOC
    full_latex += "\\begin{document}\n\n"
    
    # Add title and table of contents
    full_latex += "\\title{LLM Full Evaluation Results}\n"
    full_latex += "\\author{Generated Report}\n"
    full_latex += "\\date{\\today}\n"
    full_latex += "\\maketitle\n\n"
    full_latex += "\\tableofcontents\n"
    full_latex += "\\newpage\n\n"
    
    # Add section for tables
    full_latex += "\\section{Performance Tables}\n"
    full_latex += generate_latex_table(summary_results)
    full_latex += "\n\\newpage\n"
    
    # Add section for full results
    full_latex += "\\section{Full Results}\n"
    full_latex += generate_full_results_summary(full_results)
    
    full_latex += "\\end{document}\n"
    
    with open(os.path.join(latex_dir, "llm_full_results.tex"), 'w') as f:
        f.write(full_latex)

if __name__ == "__main__":
    main() 