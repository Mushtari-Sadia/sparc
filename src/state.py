from typing import TypedDict, List, Optional, Dict, Any
import subprocess
import tempfile
import json
import os



class NgSpiceState:

    def __init__(self, model, template=None):
        # Problem side
        self.index = -1
        self.question = ""
        self.schema = ""
        self.domain_knowledge = ""

        self.model = model
        if template:
            # Netlist sections that will be stitched into a template
            self.begin = template.begin
            self.params = template.params
            self.models = template.models
            self.connections = template.connections
            self.analysis = template.analysis
            self.print_statements = template.print_statements
            self.end = template.end
        else:
            self.begin = ""
            self.params = ""
            self.models = ""
            self.connections = ""
            self.analysis = ""
            self.print_statements = ""
            self.end = ""
        # Update spec history
        self.edit_specs: List[str] = []

        # Tool results
        self.netlist_text = ""
        self.ngspice_stdout = ""
        self.ngspice_stderr = ""
        self.parsed_results: Dict[str, Any] = {}

        # Control and routing
        self.route_stage = ""       # "plan", "analysis_type", "sweep", "run", "answer", etc
        self.analysis_kind = ""   # "dc", "ac", "tran"
        self.sweep_kind = ""      # "none", "param", "dc", "freq"
        self.needs_saturation = False
        self.needs_initial_conditions = False
        self.needs_verification = False
        self.needs_measurements = False

        self.last_error = None
        self.error_analysis = None
        self.answer = None
        self.done = False

    
    def copy_for_new_run(self):
        """Create a shallow copy for a new simulation run, sharing the model reference."""
        new_state = NgSpiceState.__new__(NgSpiceState)
        
        # Share the model reference (don't copy it)
        new_state.model = self.model
        
        # Copy primitive attributes
        new_state.index = self.index
        new_state.question = self.question
        new_state.schema = self.schema
        new_state.domain_knowledge = self.domain_knowledge
        new_state.image = self.image
        # Copy template sections
        new_state.begin = self.begin
        new_state.params = self.params
        new_state.models = self.models
        new_state.connections = self.connections
        new_state.analysis = self.analysis
        new_state.print_statements = self.print_statements
        new_state.end = self.end
        
        # Create new list for edit specs
        new_state.edit_specs = []
        
        # Reset results
        new_state.netlist_text = ""
        new_state.ngspice_stdout = ""
        new_state.ngspice_stderr = ""
        new_state.parsed_results = {}
        
        # Copy control flags
        new_state.route_stage = self.route_stage
        new_state.analysis_kind = self.analysis_kind
        new_state.sweep_kind = self.sweep_kind
        new_state.needs_saturation = self.needs_saturation
        new_state.needs_initial_conditions = self.needs_initial_conditions
        new_state.needs_verification = self.needs_verification
        new_state.needs_measurements = self.needs_measurements
        
        # Reset error/answer state
        new_state.last_error = None
        new_state.error_analysis = None
        new_state.answer = None
        new_state.done = False
        
        return new_state
    

def build_netlist_from_state(state: NgSpiceState) -> str:
    # You can add header and options here if you like
    netlist = state.begin + "\n"
    netlist += state.params + "\n"
    netlist += state.models + "\n"
    netlist += state.connections + "\n"
    netlist += state.analysis + "\n"
    netlist += state.print_statements + "\n"
    netlist += state.end + "\n"
    return netlist

def print_netlist(state):
    if state.netlist_text:
        netlist = state.netlist_text
    else:
        netlist = build_netlist_from_state(state)
    print("=== Generated Netlist ===")
    print(netlist)
    print("=========================")


import re

def _looks_like_expr_or_value(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True

    # Any of these usually means "not a node"
    if any(ch in t for ch in "(){}=,"):
        return True

    # Common analysis/value keywords
    if t.upper() in {
        "DC", "AC", "SIN", "PULSE", "EXP", "SFFM", "PWL",
        "OP", "TRAN", "AC", "DC", "PARAMS:", "TEMP", "TC", "IC"
    }:
        return True

    # Numeric literal (including 1e-3, 10k, 1u, etc.)
    # This catches "10k", "1u", "12", "-0.1", "1e5"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?[a-zA-Z]*", t):
        # If it's pure letters, not numeric, allow it (ex: node VDD)
        if re.fullmatch(r"[A-Za-z_]\w*", t):
            return False
        return True

    return False

def _extract_nodes_from_line(line: str) -> list[str]:
    s = line.strip()
    if not s or s.startswith(("*", ".")):
        return []

    parts = s.split()
    if len(parts) < 2:
        return []

    name = parts[0]
    kind = name[0].upper()

    # Expected node counts for common device cards
    # If you add more device types in your schemas, extend this mapping.
    node_counts = {
        "R": 2, "C": 2, "L": 2,
        "V": 2, "I": 2,
        "D": 2,
        "B": 2,   # behavioral source: two nodes then expression
        # "E": 4, "G": 4, "F": 2, "H": 2,   # controlled sources if you use them
        # "Q": 3, "M": 4,                   # BJT, MOSFET
        # "X": None,                        # subckt call: variable
    }

    # Subcircuit call: Xname n1 n2 ... subcktName [params]
    if kind == "X":
        nodes = []
        # collect until token looks like an expression/value, but also stop
        # before the subckt name if possible: heuristic, last "plain word" is often subckt name
        for tok in parts[1:]:
            if tok == "0":
                continue
            if _looks_like_expr_or_value(tok):
                break
            nodes.append(tok)
        return nodes

    n = node_counts.get(kind, None)
    if n is not None:
        # Just take first n node tokens after element name
        cand = parts[1:1+n]
        return [t for t in cand if t != "0" and not t.startswith(".")]

    # Fallback: collect tokens until it stops looking like nodes
    nodes = []
    for tok in parts[1:]:
        if tok == "0":
            continue
        if _looks_like_expr_or_value(tok):
            break
        nodes.append(tok)
    return nodes

def default_fixer(state: NgSpiceState) -> NgSpiceState:
    """Default fixer function to attempt to fix ngspice errors.

    Drop in replacement that avoids node inference entirely.
    It only ensures an analysis exists and adds a safe print in batch mode.
    """
    if not state.netlist_text:
        return state

    netlist = state.netlist_text
    stderr = state.ngspice_stderr.lower() if state.ngspice_stderr else ""

    # Error 1: Missing analysis / print statements in batch mode
    if "incomplete or empty netlist" in stderr or "no simulations run" in stderr:
        lines = netlist.strip().split("\n")

        has_end = any(line.strip().lower() == ".end" for line in lines)
        has_analysis = any(
            line.strip().lower().startswith((".op", ".ac", ".tran", ".dc"))
            for line in lines
        )
        has_print_or_plot = any(
            line.strip().lower().startswith((".print", ".plot", ".fourier", ".save"))
            for line in lines
        )

        fixes_to_add = []
        if not has_analysis:
            fixes_to_add.append(".op")
        if not has_print_or_plot:
            # safest possible default, never tries to infer nodes
            fixes_to_add.append(".print op all")

        if fixes_to_add:
            new_lines = []
            inserted = False
            for line in lines:
                if line.strip().lower() == ".end":
                    new_lines.extend(fixes_to_add)
                    inserted = True
                new_lines.append(line)

            # If no .end, append fixes + .end
            if not has_end:
                new_lines.extend(fixes_to_add)
                new_lines.append(".end")
            # If .end existed but wasn't matched (weird whitespace), still append
            elif not inserted:
                new_lines.extend(fixes_to_add)

            state.netlist_text = "\n".join(new_lines)
            print("=== Fixer applied: Added missing analysis/print statements ===")
            print(f"Added: {fixes_to_add}")

    # Error 2: Syntax errors in specific lines (e.g., "Not enough parameters")
    elif "not enough parameters" in stderr:
        lines = netlist.strip().split("\n")
        new_lines = []
        for line in lines:
            stripped = line.strip().lower()
            # Remove standalone analysis type words that aren't valid commands
            if stripped in ["op", "ac", "tran", "dc"]:
                continue
            # Remove malformed print-like statements (e.g., "V(NC) I(Q1)" without .print)
            if (stripped.startswith("v(") or stripped.startswith("i(")) and not stripped.startswith("."):
                continue
            new_lines.append(line)
        state.netlist_text = "\n".join(new_lines)
        print("=== Fixer applied: Removed malformed lines ===")

    return state


def run_ngspice_node(state: NgSpiceState, retry_on_error: bool = True) -> NgSpiceState:
    index = state.index
    if state.netlist_text:
        state.netlist_text = state.netlist_text.replace("</netlist_start>","")
        netlist = state.netlist_text
    else:
        netlist = build_netlist_from_state(state)
    state.netlist_text = netlist

    os.makedirs(f"files/ngspice_runs", exist_ok=True)
    with open(f"files/ngspice_runs/netlist_{index}.cir", "w") as f:
        f.write(netlist)

    with tempfile.NamedTemporaryFile(suffix=".cir", delete=False, mode="w") as f:
        f.write(netlist)
        print("=== RUNNING WITH NETLIST ===")
        print(netlist)
        path = f.name

    try:
        # Use your preferred way to run ngspice
        proc = subprocess.run(
            ["ngspice", "-b", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        state.ngspice_stdout = proc.stdout
        state.ngspice_stderr = proc.stderr
        if "error" in state.ngspice_stderr.lower():
            state.last_error = "ngspice_error"

        else:
            state.last_error = None if proc.returncode == 0 else "ngspice_error"
    except Exception as e:
        state.ngspice_stdout = ""
        state.ngspice_stderr = str(e)
        state.last_error = "ngspice_exception"

    print("=== NGSpice Execution Result ===")
    print(f"NGSpice LAST ERROR: {state.last_error}")
    print(f"NGSpice STDOUT:\n{state.ngspice_stdout}")
    print(f"NGSpice STDERR:\n{state.ngspice_stderr}")
    print("====================================")
    
    # If there's an error and retry is enabled, try fixing and rerun once
    if state.last_error and retry_on_error:
        print("=== Attempting to fix error and retry ===")
        state = default_fixer(state)
        # Rerun without retry to avoid infinite loop
        state = run_ngspice_node(state, retry_on_error=False)
    
    return state


