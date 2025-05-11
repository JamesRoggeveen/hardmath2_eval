import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor, split_symbols_custom, _token_splittable
import regex as re
import ast
import itertools
import numpy as np
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from dataclasses import dataclass, field, asdict
from benchmark_evaluator.parser_rules import deletion_rules, replacement_rules, function_rules, nested_rules, final_rules, known_functions, intermediate_functions, subsup_rewrite_pattern, subsup_pattern, sympy_symbols, beta_function_pattern, unicode_replacement_rules

class ParseError(Exception):
    """Base class for all parsing related errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        return self.message

class SolutionExtractionError(ParseError):
    """Raised when unable to extract solution from boxed environment"""
    pass

class LatexConversionError(ParseError):
    """Raised when unable to convert LaTeX to a standard expression"""
    def __init__(self, message: str, latex: str, rule: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.latex = latex
        self.rule = rule

class SymPyConversionError(ParseError):
    """Raised when unable to convert expression to SymPy"""
    def __init__(self, message: str, expression: str, stage: str, 
                 original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.expression = expression
        self.stage = stage

class EvaluationError(ParseError):
    """Raised when unable to evaluate a SymPy expression"""
    def __init__(self, message: str, expression: Any, parameters: Optional[Dict] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message, original_error)
        self.expression = expression
        self.parameters = parameters

# ===== Data Structures =====
@dataclass
class ParsingResult:
    """Container for parsing results to avoid excessive tuple returns"""
    sympy_expressions: Optional[List[sp.Expr]] = None
    error_message: str = ""
    extracted_solutions: Optional[List[str]] = None
    intermediate_expressions: Optional[List[str]] = None
    parameter_dict: Optional[Dict[sp.Symbol, Any]] = None
    parameter_values: Optional[Dict[sp.Symbol, Any]] = None
    evaluation_results: Optional[List[Any]] = None
    function_dict: Optional[Dict[str, sp.Function]] = None

    @property
    def success(self) -> bool:
        """Check if parsing was successful"""
        return self.error_message == ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ParsingResult to a JSON-serializable dictionary."""
        # First convert to a dictionary using dataclasses.asdict
        result_dict = asdict(self)
        
        # Add success property
        result_dict['success'] = self.success
        
        # Handle sympy expressions (convert to strings)
        if result_dict['sympy_expressions']:
            result_dict['sympy_expressions'] = [str(expr) for expr in result_dict['sympy_expressions']]
        
        # Handle sympy symbols in parameter_dict
        if result_dict['parameter_dict']:
            param_dict = {}
            for key, value in result_dict['parameter_dict'].items():
                # Convert sympy symbols to strings
                if isinstance(key, sp.Symbol):
                    key = str(key)
                param_dict[key] = str(value)
            result_dict['parameter_dict'] = param_dict
            
        # Handle sympy symbols in parameter_values
        if result_dict['parameter_values']:
            param_values = {}
            for key, value in result_dict['parameter_values'].items():
                # Convert sympy symbols to strings
                if isinstance(key, sp.Symbol):
                    key = str(key)
                param_values[key] = value
            result_dict['parameter_values'] = param_values

        if result_dict['function_dict']:
            param_dict = {}
            for key, value in result_dict['function_dict'].items():
                # Convert sympy symbols to strings
                if isinstance(key, sp.Function):
                    key = str(key)
                param_dict[key] = str(value)
            result_dict['function_dict'] = param_dict
        
        # Handle complex numbers in evaluation_results
        if result_dict['evaluation_results']:
            serialized_results = []
            for value in result_dict['evaluation_results']:
                if isinstance(value, complex):
                    # Format complex numbers in a standard form
                    serialized_results.append(str(value))
                else:
                    serialized_results.append(value)
            result_dict['evaluation_results'] = serialized_results
            
        return result_dict

# ===== Main Parsing Functions =====
def extract_solution(solution_string: str) -> List[str]:
    """Extract solution from boxed environment in LaTeX string"""
    if not solution_string or not solution_string.strip():
        raise SolutionExtractionError("Empty solution string provided")
        
    try:
        # boxes = re.findall(r"(boxed|fbox)\{((?:[^\{}\}]|\{(?2)\})*)\}", solution_string)
        pattern = r"(boxed|fbox)\{((?:[^\{}\}]|\{(?2)\})*)\}"
    
        # findall returns all the capture-group contents
        all_contents = re.findall(pattern, solution_string)
        # This is a little hacky
        solution_group = all_contents[-1][1] if all_contents else None
        if not solution_group:
            raise SolutionExtractionError("No boxed solution found in response")
        
        if not solution_group.strip():
            raise SolutionExtractionError("Empty solution found in boxed environment")
        
        text_wrapper = re.match(r'^\s*\\text\{(.+)\}\s*$', solution_group, re.DOTALL)
        if text_wrapper:
            solution_group = text_wrapper.group(1).strip()
        solution_group = solution_group.strip('.,')
        solution_group = re.sub(r"\\;", r"", solution_group)
        solution_list = solution_group.split(';')
        solution_list = [s.split("\\quad")[0].strip(" \n.,") for s in solution_list]

        
        # Validate each solution part
        for i, part in enumerate(solution_list):
            if not part:
                raise SolutionExtractionError(f"Empty solution part found at index {i}")
                
        return solution_list
    
    except re.error as e:
        raise SolutionExtractionError(f"Regex error while extracting solution: {str(e)}", e)
    except SolutionExtractionError:
        raise
    except Exception as e:
        raise SolutionExtractionError(f"Unexpected error while extracting solution: {str(e)}", e)

def latex_to_expression(latex_string: str, local_dict: Dict[str, Union[sp.Symbol, sp.Function]] = {}) -> str:
    """Convert LaTeX string to an expression suitable for SymPy parsing"""
    if not latex_string or not latex_string.strip():
        raise LatexConversionError("Empty LaTeX string provided", latex_string)
        
    try:
        current_string = latex_string
        for pattern, replacement in unicode_replacement_rules.items():
            latex_string = re.sub(pattern, replacement, latex_string)
        current_known_functions = known_functions.copy()
        current_known_functions = sorted(current_known_functions, key=len, reverse=True)
        for pattern in deletion_rules:
            current_string = re.sub(pattern, "", current_string)

        formatting_strings = ["text", "mathrm", "operatorname", "displaystyle"]
        names = "|".join(map(re.escape, current_known_functions))

        for fmt in formatting_strings:
            # match \fmt{   sech   } (or any known function, with spaces)
            pattern = re.compile(
                rf'\\{fmt}\{{\s*({names})\s*\}}'
            )
            current_string = pattern.sub(r'\1', current_string)

        current_string = re.sub(r"Ai\((?P<content>[^)]+)\)", r"airyai(\g<content>)", current_string)

        beta_formatted_string = rewrite_beta_function(current_string)
        if beta_formatted_string != current_string:
            current_string = beta_formatted_string
            current_known_functions.append("betainc")
            current_known_functions.append("beta")
        current_string, new_vars = preprocess_super_and_sub(current_string)
        for var in new_vars:
            local_dict[var] = sp.Symbol(var)

        current_string, _integrals = shield_integrals(current_string,local_dict)
        if _integrals != {}:
            current_known_functions.append("integrate")
            current_known_functions.append("Integral")

        # If J is a parameter, we need to surround it with spaces to prevent it from being parsed as an imaginary number.
        if "J" in local_dict.keys():
            pattern = r'(?<![A-Za-z_])J(?![A-Za-z0-9_])'
            current_string = re.sub(pattern, r' J ', current_string)
            current_string = re.sub(r'\s+', ' ', current_string).strip()
        
        # Apply function rules
        for pattern, replacement in function_rules.items():
            current_string = re.sub(pattern, replacement, current_string)

        current_string, intermediat_funcs_list = encode_frac_powers(current_string)
        current_known_functions.extend(intermediat_funcs_list)
        # Apply nested rules iteratively until no more changes
        for _ in range(5):  # Limit iterations to prevent infinite loops
            init_string = current_string
            for pattern, replacement in nested_rules.items():
                current_string = re.sub(pattern, replacement, current_string)
            if current_string == init_string:
                break
        
        # Add additional space before known functions to prevent them from being parsed as variables (i.e. x\ln(x) -> x \ln(x))    
        current_known_functions = sorted(current_known_functions, key = len, reverse=True)
        pattern = re.compile(
            r'(?<![A-Za-z])?\\?('
            + '|'.join(map(re.escape, current_known_functions))
            + r')(?![A-Za-z])'
        )
        # 3. Replace with " space + the bare function name "
        current_string = pattern.sub(r' \1', current_string)
        if len(intermediat_funcs_list) != 0:
            pattern = re.compile(r'('+'|'.join(map(re.escape,intermediat_funcs_list))+ r')([a-zA-z\d])')
            current_string = pattern.sub(r'\1 \2', current_string)
        funcs = '|'.join(map(re.escape, current_known_functions))
        pattern = re.compile(
            rf'((?:\\)?(?:{funcs}))(?![A-Za-z_])\s*'            # group1 = \?func, no letter/_, optional space
            rf'(?:\\?\{{([^\\}}]+?)\\?\}}'                     #   OR brace‐delimited → group2
            r'|\[([^\]]+?)\]'                                  #   OR bracket‐delimited → group3
            r'|([^{}\[\]\s()+\-*/^]+))'                        #   OR bare‐arg → group4
        )

        def add_parens(s: str) -> str:
            def _repl(m: re.Match) -> str:
                fn  = m.group(1)
                # whichever group matched: braces (2), brackets (3), or bare (4)
                arg = m.group(2) or m.group(3) or m.group(4)
                return f"{fn}({arg})"
            return pattern.sub(_repl, s)
        current_string = add_parens(current_string)

        current_string = decode_frac_powers(current_string)
        for pattern, replacement in replacement_rules.items():
            current_string = re.sub(pattern, replacement, current_string)
        KNOW_USER_DEFINED_PARAMETERS = set(local_dict.keys())
        KNOWN_NAMES = sorted(
            set(current_known_functions) | set(KNOW_USER_DEFINED_PARAMETERS) | set(sympy_symbols) | set(_integrals.keys()),
            key=len, reverse=True
        )

        TOKEN_RE = re.compile(
            r'('
            + r'|'.join(map(re.escape, KNOWN_NAMES))  # group of known names
            + r'|[A-Za-z]'                            # or one single letter
            + r')'
        )
        current_string = lex_identifiers(current_string, TOKEN_RE)

        current_string = apply_final_rules(current_string, KNOWN_NAMES)

        current_string = re.sub(r'\\', ' ', current_string)
        current_string = re.sub(r'\s+', ' ', current_string)
        if _integrals != {}:
            current_string = unshield_integrals(current_string, _integrals)

        # Final validation
        if not current_string.strip():
            raise LatexConversionError("Conversion resulted in empty string", latex_string)
        return current_string, local_dict
        
    except LatexConversionError:
        raise
    except Exception as e:
        raise LatexConversionError(f"Unexpected error during LaTeX conversion: {str(e)}", latex_string, None, e)

def apply_final_rules(current_string: str, known_names: List[str]) -> str:
    # 1) get all your var-names, longest first so "x_prime" splits before "x"
    var_names = sorted(known_names, key=len, reverse=True)
    if not var_names:
        # no vars? do the naïve way
        for pat, repl in final_rules.items():
            current_string = re.sub(pat, repl, current_string)
        return current_string

    # 2) build a splitter that captures any of those var-names
    splitter = re.compile(
        "(" + "|".join(re.escape(v) for v in var_names) + ")"
    )

    # 3) split into: [ chunk, var, chunk, var, chunk, ... ]
    parts = splitter.split(current_string)

    # 4) only apply your rules on the non-var chunks
    for i, part in enumerate(parts):
        if part in known_names:
            continue  # skip your known variables entirely
        for pat, repl in final_rules.items():
            part = re.sub(pat, repl, part)
        parts[i] = part

    # 5) re-join and return
    return "".join(parts)

def encode_frac_powers(s: str):
    """
    Rewrite \func^N or \func^{\frac{M}{N}} → func_N or func_M_N,
    dropping any leading backslash.  Return (new_string, placeholders_list).
    """
    funcs_group = "|".join(intermediate_functions)
    pattern = re.compile(
        r'\\?'                          # optional leading backslash
        rf'(?:{funcs_group})'           # one of your functions
        r'\^(?:\{)?'                    # ^ or ^{
        r'(?P<pow>'                     
            r'\d+/\d+'                  #   1/2
          r'|\d+'                      #   2
          r'|\\frac\{\d+\}\{\d+\}'     #   \frac{1}{2}
        r')'
        r'(?:\})?'                      # optional closing brace
    )

    placeholders: List[str] = []
    def repl(m):
        full = m.group(0)
        # strip leading backslash and the "^..." part to find func
        func = re.match(r'\\?([a-z]+)', full).group(1)
        raw  = m.group('pow')           # e.g. "2", "1/2" or "\frac{1}{2}"
        frac = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', raw)
        if frac:
            num, den = frac.groups()
        elif '/' in raw:
            num, den = raw.split('/',1)
        else:
            num, den = raw, None

        norm = f"{num}_{den}" if den else num
        ph   = f"{func}_{norm}"
        if ph not in placeholders:
            placeholders.append(ph)
        return ph

    return pattern.sub(repl, s), placeholders

def decode_frac_powers(s: str) -> str:
    """
    Replace func_N(...) or func_M_N(...) with func(...)^N or func(...)^(M/N).
    """
    paren_re  = r'(?P<content>\((?:[^()]+|(?&content))*\))'
    funcs_group = "|".join(intermediate_functions)
    decode_pat = re.compile(
        r'\b(' + funcs_group + r')_'  # func_
        r'(\d+(?:_\d+)*)'              # norm (digits or digits_digits)
        + paren_re
    )
    def _repl(m):
        func, norm, content = m.group(1), m.group(2), m.group('content')
        if '_' in norm:
            M, N = norm.split('_', 1)
            exp = f"({M}/{N})"
        else:
            exp = norm
        return f"{func}{content}^{exp}"
    return decode_pat.sub(_repl, s)
    

def rewrite_beta_function(string: str):
    def repl(m):
        a = m.group('a').strip()
        b = m.group('b').strip()
        z = m.group('z')
        if z is not None:
            # subscripted: B_{z}(a,b) → betainc(a,b,0,z)
            return f"betainc({a},{b},0,{z.strip()})"
        else:
            # plain:       B(a,b)    → beta(a,b)
            return f"beta({a},{b})"
    return beta_function_pattern.sub(repl, string)


def _parse_braces(s: str, j: int) -> Tuple[str,int]:
    """Given s[j]=='{', return ('{...}', index_after_closing_brace)."""
    assert s[j] == '{'
    depth = 0
    start = j
    L = len(s)
    while j < L:
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[start:j+1], j+1
        j += 1
    raise ValueError("Unbalanced braces")

def normalize_limit(lim: str) -> str:
    """Turn LaTeX infinities into Sympy oo notation."""
    t = lim.strip()
    if t in (r'\infty', 'infty', r'{\infty}'):
        return 'oo'
    if t in (r'-\infty', '-infty', r'{-\infty}'):
        return '-oo'
    return t

def shield_integrals(s: str, local_dict: Dict[str, sp.Symbol]={}) -> Tuple[str, Dict[str,str]]:
    """
    Recursively find the rightmost 'int_… dvar`, replace it with
    a placeholder <INT#>…<INT#>, and collect a template for each
    placeholder.  Inner integrals get shielded first.
    """
    integrals: Dict[str,str] = {}
    counter = 0

    def _shield(expr: str) -> str:
        nonlocal counter
        idx = expr.rfind(r'\int_')
        if idx < 0:
            return expr

        L = len(expr)
        # 1) parse lower limit
        j = idx + len(r'\int_')
        if expr[j] == '{':
            raw_lo, j = _parse_braces(expr, j)
        elif expr[j] == '\\':
            m = re.match(r'\\[A-Za-z]+', expr[j:])
            raw_lo = m.group(0)
            j += len(raw_lo)
        else:
            raw_lo, j = expr[j], j+1

        # 2) expect '^'
        if j < L and expr[j] == '^':
            j += 1
        else:
            raise ValueError("Missing '^' after lower limit")

        # 3) parse upper limit
        if expr[j] == '{':
            raw_hi, j = _parse_braces(expr, j)
        elif expr[j] == '\\':
            m = re.match(r'\\[A-Za-z]+', expr[j:])
            raw_hi = m.group(0)
            j += len(raw_hi)
        else:
            raw_hi, j = expr[j], j+1

        # 4) skip whitespace
        while j < L and expr[j].isspace():
            j += 1

        # 5) parse integrand + differential
        if expr.startswith(r'\frac{d', j):
            # case A: \frac{dvar}{den}
            j_frac = j + len(r'\frac')
            numb, j2 = _parse_braces(expr, j_frac)
            m = re.match(r'\{d([A-Za-z]\w*)\}', numb)
            var = m.group(1) if m else 'x'
            if j2 < L and expr[j2] == '{':
                denb, j3 = _parse_braces(expr, j2)
                den = denb.strip('{}')
                body = f"1/({den})"
                end_diff = j3
            else:
                body, end_diff = "", j2
        else:
            # case B: scan for the first top‐level d<var>
            start_body = j
            brace_depth = paren_depth = 0
            var = None
            end_diff = j

            while j < L:
                if expr.startswith(r'\int_', j):
                    # nested integral; but since we're peeling innermost first,
                    # there shouldn't be any further \int_ here
                    pass

                c = expr[j]
                if c == '{':
                    brace_depth += 1; j += 1
                elif c == '}':
                    brace_depth -= 1; j += 1
                elif c == '(':
                    paren_depth += 1; j += 1
                elif c == ')':
                    paren_depth -= 1; j += 1
                elif c == 'd' and brace_depth == 0 and paren_depth == 0:
                    m = re.match(r'd([A-Za-z]\w*)', expr[j:])
                    if m:
                        var = m.group(1)
                        end_body = j
                        end_diff = j + m.end()
                        break
                    j += 1
                else:
                    j += 1

            if var is None:
                var = 'x'
                end_body = j
                end_diff = j

            body = expr[start_body:end_body]

        # 6) build Sympy limits
        lo = normalize_limit(raw_lo) if r'\infty' in raw_lo else latex_to_expression(raw_lo, local_dict)[0]
        hi = normalize_limit(raw_hi) if r'\infty' in raw_hi else latex_to_expression(raw_hi, local_dict)[0]

        # 7) record template + placeholder
        template = f"Integral({{EXPR}},({var},{lo},{hi}))"
        key = f"<INT{counter}>"
        integrals[key] = template
        counter += 1

        # 8) splice in placeholder
        new_expr = expr[:idx] + f"{key}{body}{key}" + expr[end_diff:]
        # 9) recurse to catch any earlier (outer) integrals
        return _shield(new_expr)

    result = _shield(s)
    return result, integrals

def unshield_integrals(s: str, integrals: Dict[str,str]) -> str:
    for key in sorted(integrals, key=len, reverse=True):
        tmpl = integrals[key]
        patt = re.compile(re.escape(key) + r'(.*?)' + re.escape(key), re.DOTALL)
        s = patt.sub(lambda m, T=tmpl: T.replace("{EXPR}", m.group(1)), s)
    return s

def lex_identifiers(s: str, TOKEN_RE: re.Pattern) -> str:
    tokens = []
    pos = 0
    L   = len(s)
    while pos < L:
        m = TOKEN_RE.match(s, pos)
        if m:
            tok = m.group(1)
            tokens.append(('ID', tok))
            pos = m.end()
        else:
            # copy any one non-ID character as a separate token
            tokens.append(('OP', s[pos]))
            pos += 1

    # now rebuild the string, inserting '*' between ID tokens when necessary
    out = []
    prev_id = False
    for typ, tok in tokens:
        # only real IDs (not placeholders) count
        if typ=='ID' and not tok.startswith('<INT') :
            if tok in ('e','i'):
                tok = tok.upper()
            if prev_id:
                out.append('*')
            out.append(tok)
            prev_id = True
        else:
            out.append(tok)
            prev_id = False
    return ''.join(out)

def rewrite_super_and_sub(m):
    base = m.group('base')
    if base.lstrip('\\') == 'int':   # skip any \int
        return m.group(0)

    mods = m.group('mods')
    raw_subs = []
    raw_sups = []
    raw_exps = []

    for gm in subsup_pattern.finditer(mods):
        # apostrophes → primes
        if gm.group(1):
            raw_sups.extend(['prime'] * len(gm.group(1)))

        # subscript branches (no recursion)
        elif gm.group(2):
            for part in re.split(r'\s*,\s*', gm.group(2)):
                raw_subs.append(part.lstrip('\\'))
        elif gm.group(3):
            raw_subs.append(gm.group(3).lstrip('\\'))
        elif gm.group(4):
            raw_subs.append(gm.group(4).lstrip('\\'))

        # allowed‐letter superscripts → recurse
        elif gm.group(5) or gm.group(6):
            tok = (gm.group(5) or gm.group(6)).lstrip('\\')
            tok = subsup_rewrite_pattern.sub(rewrite_super_and_sub, tok)
            raw_sups.append(tok)

        # generic exponent → recurse
        else:
            tok = (gm.group(7) or gm.group(8))
            tok = subsup_rewrite_pattern.sub(rewrite_super_and_sub, tok)
            raw_exps.append(tok)

    # dedupe
    subs_unique = []
    for tok in raw_subs:
        if tok not in subs_unique:
            subs_unique.append(tok)
    sups_unique = []
    for tok in raw_sups:
        if tok == 'prime' or tok not in sups_unique:
            sups_unique.append(tok)

    # reassemble
    name = base
    for tok in subs_unique:
        name += f"_{tok}"
    for tok in sups_unique:
        name += f"_{tok}"
    for tok in raw_exps:
        name += f"^{{{tok}}}"

    # --- record if we actually flattened something ---
    var_part = name.split('^', 1)[0]
    if var_part != base:
        rewrite_super_and_sub.introduced_vars.add(var_part)

    return name

rewrite_super_and_sub.introduced_vars = set()

def preprocess_super_and_sub(s: str) -> Tuple[str, Set[str]]:
    # clear the recorder
    rewrite_super_and_sub.introduced_vars.clear()

    # run your full rewrite (all recursion inside rewrite_super_and_sub
    # now also records anything it introduces)
    new_string = subsup_rewrite_pattern.sub(rewrite_super_and_sub, s)
    new_string = normalize_backslashes(new_string).strip()

    # grab whatever got recorded
    new_vars = set(rewrite_super_and_sub.introduced_vars)
    return new_string, new_vars

def string_permutations(templates: List[str],
                        index_rules: Dict[str, List[str]]
                       ) -> List[str]:
    """
    Given templates like ['a_i_j','U'] and 
    index_rules={'i':['1','2'], 'j':['1','2','3']},
    returns ['a_1_1','a_1_2',…,'a_2_3','U'] (U is untouched).
    """
    # 1) One regex per index to match "_i" only when followed by "_", "(" or end
    idx_patterns = {
        idx: re.compile(rf'_{re.escape(idx)}(?=(?:_|$|\())')
        for idx in index_rules
    }
    keys = list(index_rules.keys())

    result = []
    for base in templates:
        # 2) figure out which indices actually appear in this template
        present = [idx for idx in keys if idx_patterns[idx].search(base)]
        if not present:
            # no indexed placeholders here → leave it alone
            result.append(base)
            continue

        # 3) build only the necessary Cartesian product
        combos = itertools.product(*(index_rules[idx] for idx in present))
        for combo in combos:
            s = base
            # 4) do each replacement via a lambda to preserve the underscore
            for idx, val in zip(present, combo):
                pat = idx_patterns[idx]
                s = pat.sub(lambda m, v=val: f'_{v}', s)
            result.append(s)

    return result

def find_index_rules(string: str):
    idx_sets = re.findall(
        r'\(\s*([A-Za-z]\w*)\s*,\s*([^)]+?)\s*\)',
        string
    )
    index_rules = {}
    for idx, values in idx_sets:
        # values is a string like r"\uparrow,\downarrow"
        index_rules[idx] = [v.lstrip('\\') for v in re.split(r'\s*,\s*', values)]

    #  Remove those index-rule substrings from the main text
    string = re.sub(
        r',?\s*\(\s*[A-Za-z]\w*\s*,\s*[^(),]+(?:\s*,\s*[^(),]+)*\s*\)',
        '',
        string
    )
    return string, index_rules

def normalize_backslashes(s: str) -> str:
    # turn '\\\\dagger' or '\\\\\\dagger' -> '\\dagger'
    return re.sub(r'\\\\+', r'\\', s)

def extract_symbol_and_nc_lists(function_str: str):
    # Replace lambda with lam to avoid conflicts with python reserved word using replacement_rules
    for pattern, replacement in replacement_rules.items():
        function_str = re.sub(pattern, replacement, function_str)
    nc_raw = re.findall(r'\(\s*(.+?)\s*,\s*NC\s*\)', function_str)
    function_str = re.sub(r'\(\s*.+?\s*,\s*NC\s*\)', '', function_str)
    function_str, index_rules = find_index_rules(function_str)
    
    # 3) Split on commas to get just the raw function tokens
    raw_funcs = [tok.strip() for tok in function_str.replace('$','').split(';') if tok.strip()]
    nc_raw   = [tok.strip() for tok in nc_raw if tok.strip()]

    # 4) **Normalize backslashes**, then canonicalize
    normalized = [normalize_backslashes(f) for f in raw_funcs]
    can_funcs  = [preprocess_super_and_sub(f)[0] for f in normalized]

    normalized_nc = [normalize_backslashes(f) for f in nc_raw]
    can_nc        = [preprocess_super_and_sub(f)[0] for f in normalized_nc]
    # 5) Expand every canonical template against the index_rules
    all_funcs = string_permutations(can_funcs, index_rules)
    nc_funcs  = string_permutations(can_nc, index_rules)

    all_funcs = all_funcs + nc_funcs
    nc_funcs = set(nc_funcs)

    return all_funcs, nc_funcs

def parse_functions(function_str: str):
    all_funcs, nc_funcs = extract_symbol_and_nc_lists(function_str)
    function_dict = {}
    for name in all_funcs:
        is_nc = (name in nc_funcs)
        function_dict[name] = sp.Function(name, commutative=not is_nc)

    return function_dict

def parse_parameters(parameter_str: str):
    # parameter_str, index_rules = find_index_rules(parameter_str)
    # can_params = [preprocess_super_and_sub(tok) for tok in parameter_str.replace('$','').split(';') if tok.strip()]
    # can_params = [f.replace("\\","") for f in can_params]
    # all_params = string_permutations(can_params, index_rules)
    all_params, nc_params = extract_symbol_and_nc_lists(parameter_str)
    # for the moment remove all backslashes from parameter names
    all_params = [f.replace("\\","") for f in all_params]
    nc_params = set([f.replace("\\","") for f in nc_params])
    parameter_dict = {}
    for name in all_params:
        is_nc = (name in nc_params)
        parameter_dict[name] = sp.Symbol(name, commutative=not is_nc)
    return parameter_dict

def expression_to_sympy(expr_string: str, local_dict: Dict[str, sp.Symbol] = None) -> sp.Expr:
    """Convert expression string to SymPy expression"""
    if not expr_string or not expr_string.strip():
        raise SymPyConversionError("Empty expression provided", expr_string, "input_validation")
    
    if local_dict is None:
        local_dict = {}
        
    try:
        # Handle equals sign
        if "=" in expr_string:
            parts = expr_string.split("=")
            # if len(parts) > 2:
            #     raise SymPyConversionError("Multiple equals signs found", expr_string, "equals_handling")
            expr_string = parts[-1].strip()
            
        # Handle comma (we need to come back to this)
        # if ',' in expr_string:
        #     parts = expr_string.split(',')
        #     if len(parts) > 2:
        #         raise SymPyConversionError("Multiple commas found", expr_string, "comma_handling")
        #     expr_string = parts[0].strip()
            
        if not expr_string:
            raise SymPyConversionError("Expression is empty after splitting", expr_string, "post_split")
        
        
        expr_string = re.sub(r'\s{2,}', ' ', expr_string)
        # Parse expression with enhanced transformations
        transformations = (standard_transformations + 
                         (implicit_multiplication_application,
                          convert_xor))
        try:
            expr = sp.parsing.parse_expr(expr_string, local_dict=local_dict, transformations=transformations)
            return expr
        except Exception as e:
            raise SymPyConversionError(
                f"SymPy parsing error: {str(e)}", expr_string, "sympy_parsing", e)
            
    except SymPyConversionError:
        raise
    except Exception as e:
        raise SymPyConversionError(f"Unexpected error during conversion: {str(e)}", expr_string, "unknown", e)

def solution_to_sympy(solution_string: str, parameter_str: str = "", function_str: str = "") -> ParsingResult:
    """Process a solution string to produce SymPy expressions"""
    result = ParsingResult()
    
    try:
        # 1. Extract solution from boxed environment
        result.extracted_solutions = extract_solution(solution_string)
        
        result.parameter_dict = parse_parameters(parameter_str)
        result.function_dict = parse_functions(function_str)
        local_dict = {**result.parameter_dict, **result.function_dict}
        # 2. Convert LaTeX to standard expressions
        result.intermediate_expressions = []
        for s in result.extracted_solutions:
            latex_expr, new_vars = latex_to_expression(s, local_dict=local_dict)
            result.intermediate_expressions.append(latex_expr)
            local_dict = local_dict | new_vars
        
        # 4. Convert to SymPy expressions
        result.sympy_expressions = [expression_to_sympy(s, local_dict) 
                                   for s in result.intermediate_expressions]
        
        return result
    
    except ParseError as e:
        result.error_message = str(e)
        return result
    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}"
        return result

def evaluate_expression(expr: sp.Expr, parameter_dict: Dict[sp.Symbol, Any]) -> Any:
    """Evaluate a SymPy expression with given parameters"""
    try:
        result = expr.subs(parameter_dict).evalf()
        return result
    except Exception as e:
        raise EvaluationError(f"Failed to evaluate expression: {str(e)}", expr, parameter_dict, e)
    
def evaluate_solution(solution_str: str, parameter_str: str = "", *args, **kwargs) -> ParsingResult:
    """Full pipeline to evaluate a solution with parameters into a numeric result"""
    # Parse solution to SymPy
    result = solution_to_sympy(solution_str, parameter_str)
    if not result.success:
        return result
        
    # Set random seed for reproducible parameter values
    np.random.seed(42)
    
    try:
        # # Ensure x=2 if it's present
        # if result.parameter_dict and 'x' in result.parameter_dict:
        #     parameter_values = {symbol: np.random.uniform(1, 2) for symbol in result.parameter_dict.values()}
        #     parameter_values[result.parameter_dict['x']] = 2
        # else:
        #     parameter_values = {symbol: np.random.uniform(1, 2) for symbol in result.parameter_dict.values()}

        # Choose parameter values from a uniform distribution from 1 to 2 to avoid singularities and divide by 0 (in most cases)
        parameter_values = {symbol: np.random.uniform(1, 2) for symbol in result.parameter_dict.values()}
        # Store the actual parameter values used in the evaluation
        result.parameter_values = parameter_values
        
        # Reset random seed
        np.random.seed(None)
        
        # Evaluate each expression
        result.evaluation_results = []
        for expr in result.sympy_expressions:
            value = evaluate_expression(expr, parameter_values)
            try:
                result.evaluation_results.append(float(value))
            except TypeError:
                result.evaluation_results.append(complex(value))
        return result
        
    except ParseError as e:
        result.error_message = str(e)
        return result
    except Exception as e:
        result.error_message = f"Unexpected error during evaluation: {str(e)}"
        return result

def parse_numeric_solution(s: str, *args, **kwargs) -> ParsingResult:
    """
    Takes tokens like ["(0, 4.19)", "48", …] and returns a ParsingResult
      - .extracted_solutions: the original token list
      - .evaluation_results: list of floats or tuples of floats
      - .error_message: aggregated parse errors (empty if all succeeded)
    """
    result = ParsingResult()
    try:
        tokens = extract_solution(s)
    except Exception as e:
        result.error_message = f"Unexpected error during parsing: {str(e)}"
        return result
    for i in range(len(tokens)):
        tokens[i] = tokens[i].split("=")[-1].strip()
    result.extracted_solutions = tokens

    parsed: List[Union[float, tuple]] = []
    errors: List[str] = []

    for tok in tokens:
        s = tok.strip()
        try:
            val = ast.literal_eval(s)
        except (SyntaxError, ValueError) as e:
            errors.append(f"Could not parse {tok!r}: {e}")
            continue

        # single number?
        if isinstance(val, (int, float)):
            parsed.append(float(val))

        # tuple of numbers?
        elif isinstance(val, tuple) and all(isinstance(x, (int, float)) for x in val):
            parsed.append(tuple(float(x) for x in val))

        else:
            errors.append(f"Parsed {tok!r} → {val!r}, but it's not a number or tuple of numbers")

    result.evaluation_results = parsed
    if errors:
        # join multiple errors into one message
        result.error_message = "; ".join(errors)

    return result

# ===== Main Entry Point =====
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parse and evaluate LaTeX expressions')
    parser.add_argument('expression', type=str, help='LaTeX expression to evaluate')
    parser.add_argument('--parameters', type=str, default="x", 
                       help='Comma-separated parameters (default: "x")')
    
    args = parser.parse_args()
    
    # Run the full pipeline
    result = evaluate_solution(args.expression, args.parameters)
    
    if result.success:
        print(f"Extracted solutions: {result.extracted_solutions}")
        print(f"Intermediate expressions: {result.intermediate_expressions}")
        print(f"SymPy expressions: {result.sympy_expressions}")
        print(f"Evaluation results: {result.evaluation_results}")
    else:
        print(f"Error: {result.error_message}")
        exit(1)