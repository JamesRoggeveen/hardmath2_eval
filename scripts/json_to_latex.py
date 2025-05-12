import os
import sys
import json
import re

# This file takes the results and extracts the model responses for nonlinear PDE problems

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_latex.py <folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(parent_dir, folder_name)

    json_path = os.path.join(target_dir, "full_results.json")
    tex_path = os.path.join(target_dir, "nonlinear_pdes.tex")

    # Load JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{json_path}' not found.")
        sys.exit(1)

    # Filter for nonlinear PDEs
    nonlinear_pdes = [entry for entry in data if entry.get("type") == "nonlinear_pde"]

    # Start LaTeX doc
    latex = [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"\title{Nonlinear PDE Results}",
        r"\begin{document}",
        r"\maketitle"
    ]

    for i, entry in enumerate(nonlinear_pdes, 1):
        prompt = entry.get("prompt", "").replace("\n", "\n\n")
        model_response = entry.get("model_response", "").replace("\n", "\n\n")
        model_name = entry["model_name"]

        # Boxed expression
        boxed_exprs = re.findall(r"\$\\boxed\{(.+?)\}\$", model_response)
        boxed_latex = boxed_exprs[0] if boxed_exprs else "No boxed expression found."

        # True solution
        true_solution = entry.get("eval_result", {}).get("solution", {}).get("extracted_solutions", [])
        true_solution_latex = true_solution[0] if true_solution else "No true solution provided."

        # Evaluation
        eval_result = entry.get("eval_result", {})
        is_equivalent = eval_result.get("is_equivalent", False)
        equivalence_str = "Yes" if is_equivalent else "No"

        # Numerical values
        try:
            model_val = eval_result.get("model", {}).get("evaluation_results", [None])[0]
            true_val = eval_result.get("solution", {}).get("evaluation_results", [None])[0]
            model_val_str = f"{model_val:.6f}" if model_val is not None else "N/A"
            true_val_str = f"{true_val:.6f}" if true_val is not None else "N/A"
        except Exception:
            model_val_str = "N/A"
            true_val_str = "N/A"

        latex += [
            f"\\section*{{Problem {i}}}",
            rf"\textbf{{Model:}} {model_name}",
            "",
            r"\textbf{Prompt:}",
            r"\begin{quote}",
            prompt,
            r"\end{quote}",
            r"\textbf{Model Solution:}",
            r"\begin{quote}",
            model_response,
            r"\end{quote}",
            rf"\textbf{{Boxed Expression:}} ${boxed_latex}$",
            "",
            r"\textbf{True Solution:}",
            r"\begin{quote}",
            rf"${true_solution_latex}$",
            r"\end{quote}",
            "",
            r"\textbf{Evaluation:}",
            r"\begin{itemize}",
            rf"\item \textbf{{Is Equivalent?}} {equivalence_str}",
            rf"\item \textbf{{Model Numeric Value:}} {model_val_str}",
            rf"\item \textbf{{True Numeric Value:}} {true_val_str}",
            r"\end{itemize}",
            r"\vspace{1cm}"
        ]

    latex.append(r"\end{document}")

    # Save LaTeX file
    with open(tex_path, "w") as f:
        f.write("\n".join(latex))

    print(f"LaTeX file successfully saved to: {tex_path}")

if __name__ == "__main__":
    main()
