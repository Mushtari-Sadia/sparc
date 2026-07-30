import re
from copy import deepcopy
from collections import defaultdict
import re
from state import *

class Builder:
    def __init__(self):
        pass

    def parse_update_spec(self, spec_text: str):
        edits = []
        current_section = None

        line_re = re.compile(
            r"name:\s*(?P<name>[^,]+),\s*"
            r"type:\s*(?P<type>[^,]+),\s*"
            r"old:\s*(?P<old>[^,]*),\s*"
            r"new:\s*(?P<new>.*)"
        )

        for raw_line in spec_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()

            if lower.startswith("edit:"):
                continue

            if lower.startswith("section:"):
                current_section = line.split(":", 1)[1].strip()
                continue

            if line.startswith("-"):
                content = line.lstrip("-").strip()
                m = line_re.match(content)
                if not m:
                    # if it is badly formatted, skip instead of crashing
                    continue

                data = {
                    "section": current_section,
                    "name": m.group("name").strip(),
                    "type": m.group("type").strip(),
                    "old": m.group("old").strip(),
                    "new": m.group("new").strip(),
                }
                edits.append(data)

        return edits


    # ---------- helper to find a line by name ----------

    


    def find_line_index_by_name(self, lines, name):
        """Match by leading token name."""
        for idx, line in enumerate(lines):
            if line.lstrip().startswith(name):
                return idx
        return None


    def find_model_line_index(self, lines, model_name):
        """Find .model line whose second token is the given model name."""
        for idx, line in enumerate(lines):
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0].lower() == ".model" and tokens[1] == model_name:
                return idx
        return None


    def replace_model_param_token(self, line, old, new):
        """
        In a .model line, replace or add a parameter token inside parentheses.

        Example:
        line: .model QNPN NPN (IS=1e-15 BF=100 VAF=100 Cjc=2p Cje=5p)
        old:  VAF=100
        new:  VAF=80

        If old == "", we treat this as "add new parameter token" and append new
        inside the parentheses if it is not already present.
        """
        start = line.find("(")
        end = line.rfind(")")
        # if there is no parameter list yet
        if start == -1 or end == -1 or end <= start:
            # if we are adding a new param, create a parameter list
            if old == "" and new.lower() != "none":
                if line.strip().endswith(")"):
                    # weird malformed case, just append space
                    return line + " " + new
                else:
                    return line.rstrip() + " (" + new + ")"
            return line

        before = line[: start + 1]
        params_str = line[start + 1 : end]
        after = line[end:]

        parts = params_str.split()

        # add case: old is empty means append new token if not already present
        if old == "":
            if new.lower() != "none" and new not in parts:
                parts.append(new)
            new_params_str = " ".join(parts)
            return before + new_params_str + after

        # normal replace or remove case
        for i, p in enumerate(parts):
            if p == old:
                if new.lower() == "none":
                    parts.pop(i)
                else:
                    parts[i] = new
                break

        new_params_str = " ".join(parts)
        return before + new_params_str + after


    # NEW: helper that ensures a valid ngspice device prefix for new connection names
    def sanitize_new_connection_name(self, name: str, default_prefix: str = "V") -> str:
        """
        For newly created connection lines, make sure the first character is a legal
        ngspice device prefix (V, I, R, C, L, E, G, Q, M, D, etc).

        Examples:
        "+Vsat_A1" -> "Vsat_A1"
        "-Vsat_A1" -> "Vsat_A1"
        "Vsat_A1"  -> "Vsat_A1"
        "PSUP"     -> "VPSUP"   (fallback to voltage source)
        """
        s = (name or "").strip()
        # strip leading sign characters
        while s and s[0] in "+-":
            s = s[1:]

        if not s:
            s = "XANON"

        allowed_first = set("RCLVIEDGQMJHKXYZ")  # main ngspice device prefixes
        if s[0].upper() not in allowed_first:
            s = default_prefix + s

        return s


    def apply_ngspice_edits_struct(self, netlist_obj, edits):
        """
        Apply edit dicts (from parse_update_spec) to the structured netlist.

        Supported sections and types:
        section: connections
            type: source   -> change AC or DC keyword
            type: value    -> change numeric value
            type: node1    -> change first node
            type: node2    -> change second node
            If old == "none" for a name that does not exist yet, create a new source.

        section: analysis, params, print_statements
            type: analysis_type -> replace, remove, or add analysis or param or print lines

        section: print_statements
            type: analysis  -> change or remove analysis token after .print
            type: component -> change or remove a printed quantity

        section: models (and alias: model)
            type: param -> change a parameter inside the .model line,
                        or replace the whole .model line if new is a full line.
        """
        result = deepcopy(netlist_obj)

        # First handle creation of new connection lines when old == "none"
        conn_add_specs = defaultdict(list)
        for e in edits:
            section = (e.get("section") or "").strip()
            if section != "connections":
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            old = (e.get("old") or "").strip().lower()
            if old == "none":
                conn_add_specs[(section, name)].append(e)

        # For each such group, if there is no existing line, create a new source line
        for (section, name), group in conn_add_specs.items():
            lines = result.get(section, [])
            if self.find_line_index_by_name(lines, name) is not None:
                # name already present, do not treat as pure add
                continue

            node1 = None
            node2 = None
            source_kw = None   # DC or AC
            value = None

            for e in group:
                etype = (e.get("type") or "").strip().lower()
                new_str = (e.get("new") or "").strip()
                if etype == "node1":
                    node1 = new_str
                elif etype == "node2":
                    node2 = new_str
                elif etype == "source":
                    source_kw = new_str.upper()
                elif etype == "value":
                    value = new_str

            # defaults if some fields are missing
            if not node1:
                node1 = name + "_P"
            if not node2:
                node2 = "0"
            if not source_kw:
                source_kw = "DC"
            if not value:
                value = "0"

            # NEW: sanitize the device token that will appear at the start of the line
            device_token = self.sanitize_new_connection_name(name, default_prefix="V")

            new_line = f"{device_token} {node1} {node2} {source_kw} {value}"
            lines.append(new_line)
            result[section] = lines

        # Main pass for all edits
        for edit in edits:
            section_raw = edit.get("section")
            if not section_raw:
                continue

            # alias "model" to "models"
            if section_raw == "model":
                section = "models"
            else:
                section = section_raw

            if section not in result:
                continue

            name = (edit.get("name") or "").strip()
            etype = (edit.get("type") or "").strip()
            old = (edit.get("old") or "").strip()
            new = (edit.get("new") or "").strip()

            lines = result[section]

            # model parameter update or whole line replacement
            if section == "models" and etype == "param":
                idx = self.find_model_line_index(lines, name)
                if idx is None:
                    result[section] = lines
                    continue

                # Check if we should replace/remove the entire model line
                # This happens when:
                # 1. new is a complete .model line, OR
                # 2. new is "none" (remove the model), OR
                # 3. old is "any" (indicating whole-line replacement)
                if (new.lower().startswith(".model") or 
                    new.lower() == "none" or 
                    old.lower() == "any"):
                    # Replace or remove entire model line
                    if new.lower() == "none":
                        lines.pop(idx)
                    else:
                        lines[idx] = new
                    result[section] = lines
                    continue

                # otherwise treat new as a single parameter token
                if old.lower() == "none" and new:
                    lines[idx] = self.replace_model_param_token(lines[idx], "", new)
                elif new:
                    lines[idx] = self.replace_model_param_token(lines[idx], old, new)

                result[section] = lines
                continue

            # analysis_type edits for analysis, params and print_statements
            if section in ("analysis", "params", "print_statements") and etype == "analysis_type":
                # add new line if old == "none"
                if old.lower() == "none":
                    if new and new.lower() != "none":
                        # Adjust transient analysis to output ~25 lines
                        if section == "analysis" and new.strip().lower().startswith(".tran"):
                            new = self.adjust_transient_analysis(new)
                        lines.append(new)
                        result[section] = lines
                    continue

                # normal replace or delete for an existing line
                idx = self.find_line_index_by_name(lines, name)
                if idx is None:
                    result[section] = lines
                    continue

                if new.lower() == "none":
                    lines.pop(idx)
                else:
                    # Adjust transient analysis to output ~25 lines
                    if section == "analysis" and new.strip().lower().startswith(".tran"):
                        new = self.adjust_transient_analysis(new)
                    lines[idx] = new

                result[section] = lines
                continue

            # connections creation with old == "none" has already been handled
            if section == "connections" and old.lower() == "none":
                continue

            # everything else keeps previous behavior
            idx = self.find_line_index_by_name(lines, name)
            if idx is None:
                result[section] = lines
                continue

            line = lines[idx]

            if section == "connections":
                tokens = line.split()
                if not tokens or tokens[0] != name:
                    result[section] = lines
                    continue

                if etype == "component":
                    # replace or remove the entire line for this element
                    if new.lower() == "none":
                        lines.pop(idx)
                    else:
                        lines[idx] = new
                    result[section] = lines
                    continue

                if etype == "source":
                    # change AC or DC keyword
                    for i in range(1, len(tokens)):
                        if tokens[i].upper() in ("AC", "DC"):
                            if new.lower() == "none":
                                tokens.pop(i)
                            else:
                                tokens[i] = new
                            break
                    lines[idx] = " ".join(tokens)

                elif etype == "value":
                    # change numeric value, last token by default
                    if tokens:
                        tokens[-1] = new
                    lines[idx] = " ".join(tokens)

                elif etype == "node1":
                    # first node after name
                    if len(tokens) >= 2:
                        tokens[1] = new
                    lines[idx] = " ".join(tokens)

                elif etype == "node2":
                    # second node after name
                    if len(tokens) >= 3:
                        tokens[2] = new
                    lines[idx] = " ".join(tokens)

            elif section == "print_statements":
                tokens = line.split()
                if etype == "analysis":
                    if len(tokens) >= 2 and tokens[0].lower() == ".print":
                        if new.lower() == "none":
                            lines.pop(idx)
                            result[section] = lines
                            continue
                        if tokens[1].lower() == old.lower():
                            tokens[1] = new
                        lines[idx] = " ".join(tokens)

                elif etype == "component":
                    if idx >= len(lines):
                        result[section] = lines
                        continue
                    tokens = lines[idx].split()
                    start = 2 if len(tokens) >= 2 and tokens[0].lower() == ".print" else 0
                    for i in range(start, len(tokens)):
                        if tokens[i] == old:
                            if new.lower() == "none":
                                tokens.pop(i)
                            else:
                                tokens[i] = new
                            break
                    lines[idx] = " ".join(tokens)

            result[section] = lines

        return result



    # ---------- render back to NGSpice text ----------

    def render_ngspice_netlist(self, netlist_obj, title="* Circuit generated from schema"):
        lines = []
        if title:
            lines.append(title)

        for key in ["models", "added_source", "connections", "params", "analysis", "print_statements"]:
            for line in netlist_obj.get(key, []):
                if line.strip():
                    lines.append(line)

        lines.append(".end")
        return "\n".join(lines)
    
    def extract_spec(self, model_output: str) -> str:
        model_output = model_output[model_output.find("edit"):]
        model_output = model_output.replace("```", "")
        model_output = model_output.replace("plaintext", "")
        return model_output
    
    def adjust_transient_analysis(self, tran_line):
        """
        Adjust transient analysis parameters to output approximately 25 lines.
        
        Format: .tran <tstep> <tstop> [tstart] [tmax]
        We modify tstep to be tstop/25 to get ~25 data points.
        """
        tokens = tran_line.strip().split()
        if len(tokens) < 3 or tokens[0].lower() != ".tran":
            return tran_line
        
        try:
            # Parse the time values (handle units like m, u, n, p for milli, micro, nano, pico)
            def parse_time(val):
                """Parse time value with unit suffix (e.g., '1s', '10m', '100u')"""
                val = val.strip()
                if not val:
                    return 0.0
                
                multipliers = {
                    'p': 1e-12,
                    'n': 1e-9, 
                    'u': 1e-6,
                    'm': 1e-3,
                    's': 1.0,
                }
                
                # Check if last char is a unit
                if val[-1].lower() in multipliers:
                    return float(val[:-1]) * multipliers[val[-1].lower()]
                else:
                    return float(val)
            
            def format_time(val):
                """Format time value with appropriate unit suffix"""
                if val == 0:
                    return "0"
                
                # Choose appropriate unit
                if val >= 1:
                    return f"{val}s"
                elif val >= 1e-3:
                    return f"{val*1e3}m"
                elif val >= 1e-6:
                    return f"{val*1e6}u"
                elif val >= 1e-9:
                    return f"{val*1e9}n"
                else:
                    return f"{val*1e12}p"
            
            tstop = parse_time(tokens[2])
            
            # Calculate new tstep for ~25 points
            target_points = 25
            new_tstep = tstop / target_points
            
            # Format the new tstep
            new_tstep_str = format_time(new_tstep)
            
            # Rebuild the line with adjusted tstep
            tokens[1] = new_tstep_str
            adjusted_line = " ".join(tokens)
            
            print(f"\033[93m[Info] Adjusted transient analysis for ~25 points: {adjusted_line}\033[0m")
            return adjusted_line
            
        except (ValueError, IndexError) as e:
            print(f"\033[91m[Warning] Could not parse transient analysis line: {tran_line}\033[0m")
            return tran_line
    
    def apply_edits_to_state(self, state, edits):
        """Apply parsed edits to the state sections line by line."""
        
        # First pass: handle creation of new connection lines when old == "none"
        # Group edits by (section, name) for components that need to be created
        conn_add_specs = defaultdict(list)
        for e in edits:
            section = (e.get("section") or "").strip()
            # Handle model alias
            if section == "model":
                section = "models"
            if section != "connections":
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            old = (e.get("old") or "").strip().lower()
            if old == "none":
                conn_add_specs[(section, name)].append(e)
        
        # For each group, if line doesn't exist yet, create new connection
        for (section, name), group in conn_add_specs.items():
            section_key = "connections"
            current_text = getattr(state, section_key, "")
            lines = [line for line in current_text.split('\n') if line.strip()]
            
            # Check if line already exists
            line_exists = any(line.strip().split()[0] == name if line.strip().split() else False for line in lines)
            
            if line_exists:
                continue
                
            # Build new connection line from grouped edits
            node1 = None
            node2 = None
            source_kw = None
            value = None
            
            for e in group:
                etype = (e.get("type") or "").strip().lower()
                new_str = (e.get("new") or "").strip()
                if etype == "node1":
                    node1 = new_str
                elif etype == "node2":
                    node2 = new_str
                elif etype == "source":
                    source_kw = new_str.upper()
                elif etype == "value":
                    value = new_str
            
            # Defaults if missing
            if not node1:
                node1 = name + "_P"
            if not node2:
                node2 = "0"
            if not source_kw:
                source_kw = "DC"
            if not value:
                value = "0"
            
            # Sanitize the device name
            device_token = self.sanitize_new_connection_name(name, default_prefix="V")
            
            new_line = f"{device_token} {node1} {node2} {source_kw} {value}"
            lines.append(new_line)
            setattr(state, section_key, "\n".join(lines))
        
        # Main pass: apply all edits
        for edit in edits:
            section = edit.get("section", "")
            name = edit.get("name", "")
            edit_type = edit.get("type", "")
            old_val = edit.get("old", "")
            new_val = edit.get("new", "")
            
            # Handle model alias
            if section == "model":
                section = "models"
            
            # Map section names to state keys
            if section == "connections":
                section_key = "connections"
            elif section == "analysis":
                section_key = "analysis"
            elif section == "print_statements":
                section_key = "print_statements"
            elif section == "models":
                section_key = "models"
            elif section == "params":
                section_key = "params"
            else:
                continue
            
            # Get current text and split into lines
            current_text = getattr(state, section_key, "")
            lines = [line for line in current_text.split('\n') if line.strip()]
            
            # Special handling for analysis_type in analysis, params, print_statements
            if section in ("analysis", "params", "print_statements") and edit_type == "analysis_type":
                # Add new line if old == "none"
                if old_val.lower() == "none":
                    if new_val and new_val.lower() != "none":
                        # Adjust transient analysis to output ~25 lines
                        if section == "analysis" and new_val.strip().lower().startswith(".tran"):
                            new_val = self.adjust_transient_analysis(new_val)
                        lines.append(new_val)
                        setattr(state, section_key, "\n".join(lines))
                    continue
                
                # Find and replace/delete existing line
                line_index = None
                for idx, line in enumerate(lines):
                    if line.strip().split()[0] == name if line.strip().split() else False:
                        line_index = idx
                        break
                
                if line_index is not None:
                    if new_val.lower() == "none":
                        lines.pop(line_index)
                    else:
                        # Adjust transient analysis to output ~25 lines
                        if section == "analysis" and new_val.strip().lower().startswith(".tran"):
                            new_val = self.adjust_transient_analysis(new_val)
                        lines[line_index] = new_val
                    setattr(state, section_key, "\n".join(lines))
                continue
            
            # Skip connection creation lines (already handled above)
            if section == "connections" and old_val.lower() == "none":
                continue
            
            # Find the line that starts with the component name
            line_index = None
            for idx, line in enumerate(lines):
                tokens = line.strip().split()
                if tokens and tokens[0] == name:
                    line_index = idx
                    break
            
            if line_index is None:
                # Line not found - handle adding new line for non-connection sections
                if old_val.lower() == "none" or old_val == "":
                    if new_val and new_val.lower() != "none":
                        lines.append(new_val)
                        setattr(state, section_key, "\n".join(lines))
                continue
            
            # Found the line - now apply the edit based on type and section
            line = lines[line_index]
            tokens = line.split()
            
            if section == "connections":
                if edit_type == "component":
                    # Replace or remove entire line
                    if new_val.lower() == "none":
                        lines.pop(line_index)
                    else:
                        lines[line_index] = new_val
                        
                elif edit_type == "value":
                    # For value edits, replace the last token (the value)
                    if len(tokens) > 0:
                        tokens[-1] = new_val
                    lines[line_index] = " ".join(tokens)
                    
                elif edit_type == "source":
                    # For source type edits (AC/DC), replace the source keyword
                    for i in range(1, len(tokens)):
                        if tokens[i].upper() in ("AC", "DC"):
                            if new_val.lower() == "none":
                                tokens.pop(i)
                            else:
                                tokens[i] = new_val
                            break
                    lines[line_index] = " ".join(tokens)
                    
                elif edit_type == "node1":
                    # Replace first node (token at index 1)
                    if len(tokens) >= 2:
                        tokens[1] = new_val
                    lines[line_index] = " ".join(tokens)
                    
                elif edit_type == "node2":
                    # Replace second node (token at index 2)
                    if len(tokens) >= 3:
                        tokens[2] = new_val
                    lines[line_index] = " ".join(tokens)
                        
            elif section == "print_statements":
                if edit_type == "analysis":
                    # Change analysis type in .print statement
                    if len(tokens) >= 2 and tokens[0].lower() == ".print":
                        if new_val.lower() == "none":
                            lines.pop(line_index)
                        elif tokens[1].lower() == old_val.lower():
                            tokens[1] = new_val
                            lines[line_index] = " ".join(tokens)
                            
                elif edit_type == "component":
                    # Change or remove a specific printed quantity
                    start = 2 if len(tokens) >= 2 and tokens[0].lower() == ".print" else 0
                    for i in range(start, len(tokens)):
                        if tokens[i] == old_val:
                            if new_val.lower() == "none":
                                tokens.pop(i)
                            else:
                                tokens[i] = new_val
                            break
                    lines[line_index] = " ".join(tokens)
                        
            elif section in ("analysis", "params"):
                if edit_type == "analysis_type":
                    # Replace entire line
                    if new_val.lower() == "none":
                        lines.pop(line_index)
                    else:
                        # Adjust transient analysis to output ~25 lines
                        if section == "analysis" and new_val.strip().lower().startswith(".tran"):
                            new_val = self.adjust_transient_analysis(new_val)
                        lines[line_index] = new_val
                        
            elif section == "models":
                if edit_type == "param":
                    # Check if we should replace/remove the entire model line
                    # This happens when: 
                    # 1. new_val is a complete .model line, OR
                    # 2. new_val is "none" (remove the model), OR
                    # 3. old_val is "any" (indicating whole-line replacement)
                    if (new_val.lower().startswith(".model") or 
                        new_val.lower() == "none" or 
                        old_val.lower() == "any"):
                        # Replace or remove entire model line
                        if new_val.lower() == "none":
                            lines.pop(line_index)
                        else:
                            lines[line_index] = new_val
                    else:
                        # Replace parameter within the model line
                        lines[line_index] = self.replace_model_param_token(
                            lines[line_index], old_val, new_val
                        )
            
            # Update the state section with modified lines
            setattr(state, section_key, "\n".join(lines))
        
        return state
    
    def build(self, state, model_output):
        print_netlist(state)
        spec_text = self.extract_spec(model_output)
        edits = self.parse_update_spec(spec_text)
        state = self.apply_edits_to_state(state, edits)
        print("\033[92mEdits applied:\033[0m")
        print(spec_text)
        print("\033[94mUpdated Netlist:\033[0m")
        print_netlist(state)
        return state

