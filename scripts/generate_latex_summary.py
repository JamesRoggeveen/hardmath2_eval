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

def generate_latex_table(summary_stats: Dict[str, Any]) -> str:
    """Generate LaTeX tables showing success rates for each problem using summary statistics."""
    models = summary_stats["models"]
    if not models:
        return "No model results found."
    
    # Calculate the number of queries per prompt (n)
    queries_per_prompt = max(
        int(round(summary_stats[model]["total_queries"] / summary_stats[model]["total_prompts"]))
        for model in models
    )
    
    # Create model labels (A, B, C, etc.)
    model_labels = {model: chr(65 + i) for i, model in enumerate(models)}
    
    # Generate one-shot success table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|l|" + "|c" * len(models) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Type} & \\textbf{Prompt} & " + " & ".join(f"\\textbf{{{label}}}" for label in model_labels.values()) + " \\\\\n"
    latex += "\\hline\n"
    
    # Get all prompts from all models' prompt_breakdown
    prompt_entries = set()
    for model in models:
        for prompt_key, prompt_stats in summary_stats[model]["prompt_breakdown"].items():
            prompt_idx = int(prompt_key.split('_')[1])  # Extract number from "prompt_X"
            prompt_entries.add((prompt_stats["problem_type"], prompt_idx))
    
    # Convert to list and sort by problem type and prompt index
    prompt_entries = sorted(list(prompt_entries), key=lambda x: (x[0], x[1]))
    
    # Generate rows for each prompt
    for problem_type, prompt_idx in prompt_entries:
        row = f"{format_problem_type(problem_type)} & {prompt_idx}"
        
        # Add success/failure for each model
        for model in models:
            prompt_key = f"prompt_{prompt_idx}"
            prompt_stats = summary_stats[model]["prompt_breakdown"].get(prompt_key, {})
            success_rate = prompt_stats.get("success_rate", 0)
            
            if success_rate == 100:
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
    
    # Use the same sorted prompt entries
    for problem_type, prompt_idx in prompt_entries:
        row = f"{format_problem_type(problem_type)} & {prompt_idx}"
        
        # Add percentage success for each model
        for model in models:
            prompt_key = f"prompt_{prompt_idx}"
            prompt_stats = summary_stats[model]["prompt_breakdown"].get(prompt_key, {})
            success_rate = prompt_stats.get("success_rate", 0)
            
            # Color based on success rate
            if success_rate >= 75:
                color = "successgreen!25"
            elif success_rate >= 50:
                color = "yellow!25"
            else:
                color = "failurered!25"
            row += f" & \\cellcolor{{{color}}}\\textcolor{{black}}{{{success_rate:.0f}\\%}}"
        
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
    latex += "\\end{table}\n\n"

    # Get all question types including "Overall"
    question_types = ["Overall"] + sorted(summary_stats["question_types"])
    
    # Generate success rates table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|" + "|c" * len(question_types) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Model} & " + " & ".join(f"\\textbf{{{format_problem_type(qt)}}}" for qt in question_types) + " \\\\\n"
    latex += "\\hline\n"
    
    for model in models:
        row = f"{model}"
        # Add overall success rate
        overall_rate = summary_stats[model]["overall_success_rate"]
        row += f" & {overall_rate:.1f}"
        
        # Add success rates for each question type
        for q_type in question_types[1:]:  # Skip "Overall"
            q_type_stats = summary_stats[model]["question_type_breakdown"].get(q_type, {})
            success_rate = q_type_stats.get("success_rate", 0)
            row += f" & {success_rate:.1f}"
        
        row += " \\\\\n\\hline\n"
        latex += row
    
    latex += "\\end{tabular}\n"
    latex += "\\caption{Success Rates by Model and Question Type}\n"
    latex += "\\end{table}\n\n"
    
    # Generate pass@1 rates table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|" + "|c" * len(question_types) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Model} & " + " & ".join(f"\\textbf{{{format_problem_type(qt)}}}" for qt in question_types) + " \\\\\n"
    latex += "\\hline\n"
    
    for model in models:
        row = f"{model}"
        # Add overall pass@1 rate
        overall_rate = summary_stats[model]["overall_pass_at_1_rate"]
        row += f" & {overall_rate:.1f}"
        
        # Add pass@1 rates for each question type
        for q_type in question_types[1:]:  # Skip "Overall"
            q_type_stats = summary_stats[model]["question_type_breakdown"].get(q_type, {})
            pass_at_1_rate = q_type_stats.get("pass_at_1_rate", 0)
            row += f" & {pass_at_1_rate:.1f}"
        
        row += " \\\\\n\\hline\n"
        latex += row
    
    latex += "\\end{tabular}\n"
    latex += "\\caption{Pass@1 Rates by Model and Question Type}\n"
    latex += "\\end{table}\n\n"
    
    # Generate pass@n rates table
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|" + "|c" * len(question_types) + "|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Model} & " + " & ".join(f"\\textbf{{{format_problem_type(qt)}}}" for qt in question_types) + " \\\\\n"
    latex += "\\hline\n"
    
    for model in models:
        row = f"{model}"
        # Add overall pass@n rate
        overall_rate = summary_stats[model]["overall_pass_at_n_rate"]
        row += f" & {overall_rate:.1f}"
        
        # Add pass@n rates for each question type
        for q_type in question_types[1:]:  # Skip "Overall"
            q_type_stats = summary_stats[model]["question_type_breakdown"].get(q_type, {})
            pass_at_n_rate = q_type_stats.get("pass_at_n_rate", 0)
            row += f" & {pass_at_n_rate:.1f}"
        
        row += " \\\\\n\\hline\n"
        latex += row
    
    latex += "\\end{tabular}\n"
    latex += f"\\caption{{Pass@{queries_per_prompt} Rates by Model and Question Type}}\n"
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
                    model_response = result["model_response"]
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
                    model_solution = result["eval_result"].get("model", {}).get("extracted_solutions", [])
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

                    solution_latex = result["eval_result"]["solution"]["extracted_solutions"]
                    if solution_latex:
                        latex += "\\noindent\\textbf{{Solution}}:\\par\n"
                        for solution in solution_latex:
                            # Clean up the solution and ensure it's properly formatted
                            solution = solution.strip()
                            if solution:
                                # Remove any markdown code block delimiters
                                solution = solution.replace('```', '')
                                latex += "\\begin{equation*}\n"
                                latex += solution + "\n"
                                latex += "\\end{equation*}\n\n"
                    
                    # Add evaluation results
                    model_eval = result["eval_result"].get("model", {}).get("evaluation_results", [])
                    solution_eval = result["eval_result"].get("solution", {}).get("evaluation_results", [])
                    if model_eval and solution_eval:
                        latex += "\\noindent\\textbf{{Evaluation Results}}:\\par\n"
                        # Format evaluation results as a table
                        latex += "\\begin{tabular}{ll}\n"
                        latex += "\\hline\n"
                        latex += "\\textbf{Model} & \\textbf{Expected} \\\\\n"
                        latex += "\\hline\n"
                        # Format each evaluation result
                        for m_eval, s_eval in zip(model_eval, solution_eval):
                            latex += f"{m_eval:.6f} & {s_eval:.6f} \\\\\n"
                        latex += "\\hline\n"
                        latex += "\\end{tabular}\n\n"
                    
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
                    model_solution = result.get('model_solution_latex', [])
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
    
    # Check if summary.json exists, if not generate it first
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        from generate_summary import generate_summary
        generate_summary()
    
    # Read summary statistics
    with open(summary_path, 'r') as f:
        summary_stats = json.load(f)
    
    # Read results for detailed view
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
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
    summary_latex += generate_latex_table(summary_stats)
    summary_latex += "\n\\newpage\n"
    
    # Add section for detailed results
    summary_latex += "\\section{Detailed Results}\n"
    summary_latex += generate_detailed_summary(results)
    
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
    full_latex += generate_latex_table(summary_stats)
    full_latex += "\n\\newpage\n"
    
    # Add section for full results
    full_latex += "\\section{Full Results}\n"
    full_latex += generate_full_results_summary(full_results)
    
    full_latex += "\\end{document}\n"
    
    with open(os.path.join(latex_dir, "llm_full_results.tex"), 'w') as f:
        f.write(full_latex)

if __name__ == "__main__":
    main() 