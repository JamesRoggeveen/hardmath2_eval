"""
Evaluator module evaluating mathematical solutions.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from . import parser
from .parser import ParsingResult

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    success: bool = False
    error_message: str = ""
    solution_result: Optional[ParsingResult] = None
    model_result: Optional[ParsingResult] = None
    is_equivalent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation result to a dictionary for JSON serialization"""
        result = {
            "success": self.success,
            "error_message": self.error_message,
            "is_equivalent": self.is_equivalent
        }
        
        # Add solution result if available
        if self.solution_result:
            result["solution"] = self.solution_result.to_dict()
            
        # Add model result if available
        if self.model_result:
            result["model"] = self.model_result.to_dict()
            
        return result

def evaluate_model(model_response: str, solution_string: str, parameter_string: str, function_str: str, parse_function: Callable, eval_function: Callable) -> EvaluationResult:
    """Evaluate an LLM's solution against a reference solution."""
    result = EvaluationResult()
    
    # Process reference solution
    solution_result = parse_function(solution_string, parameter_string, function_str)
    if not solution_result.success:
        result.error_message = f"Failed to parse reference solution: {solution_result.error_message}"
        return result

    result.solution_result = solution_result
        
    # Process model response
    model_result = parse_function(model_response, parameter_string, function_str)
    if not model_result.success:
        result.error_message = f"Failed to parse model response: {model_result.error_message}"
        return result
    
    result.model_result = model_result
    
    # Compare evaluation results
    return eval_function(result)

def evaluate_solution(query_string: str, solution_string: str, parameter_string: str) -> EvaluationResult:
    return evaluate_model(query_string, solution_string, parameter_string, "", parser.evaluate_solution, is_equivalent_numerics)

def evaluate_numeric_solution(query_string: str, solution_string: str) -> EvaluationResult:
    return evaluate_model(query_string, solution_string, "", "", parser.parse_numeric_solution, is_equivalent_numerics)

def evaluate_functional_solution(query_string: str, solution_string: str, parameter_string: str, function_string: str) -> EvaluationResult:
    return evaluate_model(query_string, solution_string, parameter_string, function_string, parser.solution_to_sympy, is_equivalent_functional_form)
    
def is_equivalent_functional_form(result: EvaluationResult) -> EvaluationResult:
    try:
        # Get sympy expressions
        model_exprs = result.model_result.sympy_expressions
        solution_exprs = result.solution_result.sympy_expressions
        local_dict = {**result.solution_result.parameter_dict, **result.solution_result.function_dict}
        
        # Validate expressions exist
        if not model_exprs or not solution_exprs:
            result.error_message = "One or both expressions failed to parse to SymPy expressions"
            result.success = False
            return result
            
        # Check expression counts match
        if len(model_exprs) != len(solution_exprs):
            result.error_message = f"Number of expressions do not match: {len(model_exprs)} vs {len(solution_exprs)}"
            result.success = True
            return result
    
        try:
            n = len(model_exprs)
            expression_difference = []
            if any('_dagger' in key for key in local_dict.keys()):
                fermionic_flag = True
            else:
                fermionic_flag = False
            results = np.zeros((n,n))
            for i,expr1 in enumerate(model_exprs):
                for j,expr2 in enumerate(solution_exprs):
                    diff = (expr1 - expr2).doit()
                    if fermionic_flag:
                        diff = parser.simplify_fermionic_expression(diff, local_dict)
                    expression_difference.append(diff)
                    results[i,j] = (diff.simplify() == 0)
            result.is_equivalent = all(any(row) for row in results)

        except Exception as e:
            result.error_message = f"Error expanding expressions: {str(e)}"
            result.success = False
            return result
        result.success = True
        return result
        
    except Exception as e:
        result.error_message = f"Unexpected error during comparison: {str(e)}"
        return result
    
def is_equivalent_numerics(result: EvaluationResult)->EvaluationResult:
    try:
        # Custom sort key that handles both complex and float values
        def sort_key(x):
            if hasattr(x, '__getitem__'):
                val = x[0]
            else:
                val = x
            # For complex numbers, sort by real part then imaginary part
            if isinstance(val, complex):
                return (val.real, val.imag)
            return (float(val), 0)  # Convert to float and use 0 for imaginary part
            
        model_solution = sorted(result.model_result.evaluation_results, key=sort_key)
        solution_solution = sorted(result.solution_result.evaluation_results, key=sort_key)
        
        # Check if shape of arrays match
        if len(model_solution) != len(solution_solution):
            result.error_message = f"Evaluation shapes don't match: {len(solution_solution)} vs {len(model_solution)}"
            result.success = True
            result.is_equivalent = False
            return result
        equivalent = True

        for model_element, solution_element in zip(model_solution, solution_solution):
            # Check if both are tuples/lists or both are scalar values
            if hasattr(model_element, '__len__') != hasattr(solution_element, '__len__'):
                # One is tuple/list and other is scalar
                result.error_message = f"Mismatched types: one is tuple/list and other is scalar"
                result.success = True
                result.is_equivalent = False
                return result
            elif hasattr(model_element, '__len__'):
                # Both are tuples/lists, check lengths match
                if len(model_element) != len(solution_element):
                    result.error_message = f"Evaluation shapes don't match: {len(solution_element)} vs {len(model_element)}"
                    result.success = True
                    result.is_equivalent = False
                    return result
            equivalent *= np.allclose(model_element, solution_element, rtol=1e-6, atol=1e-12)
        result.is_equivalent = bool(equivalent)
        result.success = True
        return result
    except Exception as e:
        result.error_message = f"Error comparing evaluation results: {str(e)}"
