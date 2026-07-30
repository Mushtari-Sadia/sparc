import json
import sys
import os

# Add parent directory to path to import Model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import Model
from state import build_netlist_from_state, NgSpiceState
from builder import Builder
from self_consistency import self_consistency_inference


ngspice_edit_spec_language_doc = """
<NGSpice edit specification syntax>

This document tells you how to write edit specifications that change an ngspice netlist.
You should follow these rules exactly.

1. Overall format

The output always begins with this line:
edit:

Then you give one or more sections. Each section starts on its own line:
section: <section_name>

Valid section names are:
- params
- connections
- analysis
- print_statements
- models

After each section line, you list one or more edit lines, one per line:
- name: <identifier>, type: <type>, old: <old_value>, new: <new_value>

Rules for these edit lines:
- The dash must be at the very start of the line
- The fields must appear in this exact order:
  name, type, old, new
- All four fields must be present, even if "old" or "new" is empty

Removal is always written as:
new: none

Addition is always written as:
old: none


2. Allowed types

The "type" field can be one of:
- source
- node1
- node2
- value
- param
- analysis_type
- analysis
- component

Not every type is valid in every section. The allowed combinations and examples
are described below.


3. Section: connections

This section corresponds to device and source lines, for example:
V1 in 0 DC 5
R1 in out 10k
C1 out 0 1u

The "name" field is the element name that appears at the start of the line:
- name: V1 for "V1 in 0 DC 5"
- name: R1 for "R1 in out 10k"

Supported types in this section:

3.1 type: component  (replace or remove the entire line)

Use this when you want to replace the full text of a device line or remove it.

Example, remove a resistor:
Before:
  R1 in out 10k

edit:
section: connections
- name: R1, type: component, old: R1 in out 10k, new: none

Result:
  The R1 line is deleted.

Example, replace a resistor:
Before:
  R1 in out 10k

edit:
section: connections
- name: R1, type: component, old: R1 in out 10k, new: R1 in out 5k

Result:
  R1 in out 5k


3.2 type: source  (change AC or DC keyword)

Use this to change the AC or DC keyword of a source.

Before:
  V1 in 0 AC 1

edit:
section: connections
- name: V1, type: source, old: AC, new: DC

After:
  V1 in 0 DC 1


3.3 type: value  (change the numeric value at the end of the line)

Use this to change the final numeric value token.

Before:
  V1 in 0 DC 1

edit:
section: connections
- name: V1, type: value, old: 1, new: 0.1

After:
  V1 in 0 DC 0.1


3.4 type: node1  (change the first node after the name)

Use this to change the first node argument.

Before:
  R1 in out 10k

edit:
section: connections
- name: R1, type: node1, old: in, new: n_in

After:
  R1 n_in out 10k


3.5 type: node2  (change the second node after the name)

Use this to change the second node argument.

Before:
  R1 in out 10k

edit:
section: connections
- name: R1, type: node2, old: out, new: n_out

After:
  R1 in n_out 10k


3.6 Adding a new source or device line in connections

To add a new source or device, you always use old: none for that name.
You provide separate edits with type node1, node2, source, and value for the
same name. 

Example: add a new DC source VBIAS between node n1 and ground with 2 V

section: connections
- name: VBIAS, type: node1, old: none, new: n1
- name: VBIAS, type: node2, old: none, new: 0
- name: VBIAS, type: source, old: none, new: DC
- name: VBIAS, type: value, old: none, new: 2

This will create a new line, for example:
VBIAS n1 0 DC 2


4. Section: analysis

This section corresponds to analysis commands, for example:
.op
.tran 1m 1s
.ac lin 10 1k 10k

Here you always use:
type: analysis_type

4.1 Remove or replace an existing analysis line

Example, remove an AC analysis:

Before:
  .ac lin 10 1k 10k

edit:
section: analysis
- name: .ac, type: analysis_type, old: .ac lin 10 1k 10k, new: none

Result:
  The .ac line is deleted.

Example, change the AC sweep:

section: analysis
- name: .ac, type: analysis_type, old: .ac lin 10 1k 10k, new: .ac dec 20 100 10k


4.2 Add a new analysis line

Use old: none and place the full analysis line in new.

Example, add an operating point analysis:

section: analysis
- name: .op, type: analysis_type, old: none, new: .op
- name: .ic, type: analysis_type, old: none, new: .ic V(No)=0 V(Nn)=0
- name: .tran, type: analysis_type, old: none, new: .tran 50m 1s


5. Section: params

This section corresponds to parameter definition lines, for example:
.param Rload=1k
.param Vin=0.1

You also use:
type: analysis_type

5.1 Remove a parameter line

section: params
- name: .param, type: analysis_type, old: .param Rload=1k, new: none

5.2 Add a parameter line

section: params
- name: .param, type: analysis_type, old: none, new: .param Rload=2k

5.3 Replace a parameter line

section: params
- name: .param, type: analysis_type, old: .param Vin=0.1, new: .param Vin=0.2

All parameter definition lines must be edited through the params section.


6. Section: print_statements

This section corresponds to .print and .save lines, for example:
.print ac V(out) V(in)
.print tran V(out)
.save V(out) I(V1)

Three kinds of edits are allowed here:
- type: analysis_type   for full line add or remove
- type: analysis        to change the analysis keyword after .print
- type: component       to change or remove one printed quantity

6.1 Whole line edits with type: analysis_type

Use this when you want to add, replace, or delete entire .print or .save lines.

Example, remove a .save line:

section: print_statements
- name: .save, type: analysis_type, old: .save V(out) I(V1), new: none

Example, add a .print line:

section: print_statements
- name: .print, type: analysis_type, old: none, new: .print tran V(out)


6.2 Change the analysis keyword inside .print with type: analysis

Use this when you want to change the token after .print, for example
from "ac" to "tran".

Before:
  .print ac V(out) V(in)

edit:
section: print_statements
- name: .print, type: analysis, old: ac, new: tran

After:
  .print tran V(out) V(in)

If you set new: none for type: analysis, the entire .print line is removed.


6.3 Change or remove a single printed quantity with type: component

Use this when you want to adjust one term, for example V(N1) or I(V1).

Before:
  .print tran V(N1) V(N2) I(V1)

Remove V(N1):

section: print_statements
- name: .print, type: component, old: V(N1), new: none

After:
  .print tran V(N2) I(V1)

Replace V(N2) with V(out):

section: print_statements
- name: .print, type: component, old: V(N2), new: V(out)

After:
  .print tran V(out) I(V1)


7. Section: models

This section corresponds to .model lines, for example:
.model QNPN NPN (IS=1e-15 BF=100 VAF=100)
.model DIO D (IS=1e-14 N=1)

Here you always use:
type: param

There are two ways to edit models:

7.1 Edit a single parameter token inside parentheses

In this case, "new" is just one token such as:
VAF=80
BV=5
IS=1e-15

Examples:

Change an existing parameter:

Before:
  .model QNPN NPN (IS=1e-15 BF=100 VAF=100)

edit:
section: models
- name: QNPN, type: param, old: VAF=100, new: VAF=80

After:
  .model QNPN NPN (IS=1e-15 BF=100 VAF=80)

Add a new parameter:

Before:
  .model DIO D (IS=1e-14)

edit:
section: models
- name: DIO, type: param, old: none, new: BV=5

After:
  .model DIO D (IS=1e-14 BV=5)

Remove a parameter:

Before:
  .model DIO D (IS=1e-14 BV=5)

edit:
section: models
- name: DIO, type: param, old: BV=5, new: none

After:
  .model DIO D (IS=1e-14)


7.2 Replace or remove an entire .model line

If you want to replace the whole line, put the full .model line in new.

Example, replace a model completely:

section: models
- name: QNPN, type: param, old: any, new: .model QNPN NPN (IS=1e-15 BF=200)

Example, remove a model completely:

section: models
- name: QNPN, type: param, old: any, new: none


8. Addition and removal summary

Addition:
- Use old: none and set new to the desired content.
  Examples:
  - new analysis line in the analysis section
  - new .param line in the params section
  - new .print or .save in the print_statements section
  - new source line in the connections section
  - new parameter token inside parentheses in the models section

Removal:
- Use new: none
  Examples:
  - remove a line in analysis, params, or print_statements
  - remove a device line in connections when type: component
  - remove a parameter token inside a .model line in models
  - remove a whole .model line in models

All parameter definition lines belong in the params section.

</NGSpice edit specification syntax>"""

def get_update_agent_prompt(if_error):
  if if_error:
    error_statement = "3. If you are given ngspice simulation log containing errors, along with an explanation of the errors, you must update the circuit specifically to FIX the ERRORS only. You must NOT make any other changes to the circuit or analysis. Your output must be an ngspice edit specification that ONLY fixes the errors mentioned in the log."
  else:
    error_statement = ""

  update_agent_prompt = f"""## Output format

      You must always produce your answer in two parts, in this order:

      1. A "Reasoning" section where you explain, step by step, how you decided which elements or analysis statements need to be updated, and whether each candidate value or analysis type should change or not. This must reference the names and values in the question and in the netlist.
      2. The NGSpice edit specification, starting with the word "edit:" and following the exact syntax given below.
      {error_statement}
      Your final output must therefore look like:

      Reasoning:
      - What edits are needed and why?

      edit:
      section: ...
      - name: ..., type: ..., old: ..., new: ...

      If no updates are needed, your "Reasoning" section should explain why the existing netlist already matches the question, and your edit specification should be exactly:

      edit:

      ## Output syntax
      Use the following NGSpice edit specification syntax to output the required changes. An "edit:" with no following text means no updates are needed.
      {ngspice_edit_spec_language_doc}

      ## Examples (An "edit:" with no following text means no updates are needed.)

      Example 1:

      # Question:
      What is the current through the circuit when the resistor R1=5k?

      # Domain knowledge:
      In a circuit, the current is calculated by Ohm's Law: I = V / R, where I is the current, V is the voltage across the component, and R is the resistance. To find the current through a resistor, you need to know the voltage across it and its resistance value.

      # Current netlist:
      * Circuit generated from schema
      V1 N1 0 DC 5
      R1 N1 N2 5k
      C1 N2 0 10u
      .ac lin 1 1 1
      .print ac V(N2)
      .end

      # Output:

      Reasoning:
      - The question states R1 = 5kΩ.
      - The netlist already has R1 with value 5k.
      - The analysis type and source already match a DC supply and small signal AC analysis is present but not in conflict with the question.
      - Therefore, no updates to the netlist are needed.

      edit:


      Example 2:

      # Question:
      What is the current through the circuit when the resistor R1=5k?

      # Domain knowledge:
      In a circuit, the current is calculated by Ohm's Law: I = V / R, where I is the current, V is the voltage across the component, and R is the resistance value.

      # Current netlist:
      * Circuit generated from schema
      V1 N1 0 DC 5
      R1 N1 N2 1
      C1 N2 0 10u
      .ac lin 1 1 1
      .print ac V(N2)
      .end

      # Output:

      Reasoning:
      - The question states R1 = 5kΩ.
      - In the netlist, R1 has value 1, which does not match 5k.
      - Only R1 is explicitly constrained by the question, and no analysis type change is requested.
      - Therefore, only the value of R1 needs to be updated.

      edit:
      section: connections
      - name: R1, type: value, old: 1, new: 5k

      Example 3:

      # Question:
      Find the output at ω = 200 rad/s.

      # Domain knowledge:
      Frequency in rad/s must be converted to Hz using f = ω / (2π).
      The .ac statement should sweep only at that frequency.
      
      # Current netlist:
      * RC low pass filter
      Vin in 0 AC 1
      R1 in out 1k
      C1 out 0 1u
      .ac lin 1 1 1
      .print ac V(out)
      .end

      # Output:

      Reasoning:
      - The question specifies an angular frequency ω = 200 rad/s.
      - The netlist uses .ac lin 1 1 1, which sweeps at 1 Hz, not 200 rad/s.
      - Frequency in Hz should be f = ω / (2π), so we introduce a parameter f = 200/(2*3.14159265).
      - The .ac statement should then sweep only at this f.
      - No component values are constrained by the question, so they remain unchanged.

      edit:
      section: analysis
      - name: .ac, type: analysis_type, old: .ac lin 1 1 1, new: .ac lin 1 {{f}} {{f}}
      section: params
      - name: .param, type: analysis_type, old: none, new: .param f = 200/(2*3.14159265)
  """
  return update_agent_prompt

def update_agent(state, prompt,self_consistency_number=1):
      edit_spec_text = "\n".join(str(edit) for edit in state.edit_specs)
      if state.netlist_text:
          netlist = state.netlist_text
      else:
          netlist = build_netlist_from_state(state)
      
      if state.last_error is not None:
          if_error = True
          full_prompt = f"""{prompt}

# Schema:
{state.schema}

# Question:
{state.question}

# Domain knowledge:
{state.domain_knowledge}

# Edits by previous agents:
{edit_spec_text}

# Current netlist:
{netlist}

# Last error log:
{state.ngspice_stdout}
{state.ngspice_stderr}

# Error analysis:
{state.error_analysis}

# Output:
"""

      else:
          if_error = False
          full_prompt = f"""{prompt}

# Schema:
{state.schema}

# Question:
{state.question}

# Domain knowledge:
{state.domain_knowledge}

# Edits by previous agents:
{edit_spec_text}

# Current netlist:
{netlist}

# Output:
"""
      system_prompt = "You are an expert in electrical engineering and ngspice. You will be given an existing netlist of a circuit, a natural language question about the circuit, and some domain knowledge of the circuit. Your job is to update the given netlist according to rules and syntax defined below.\n" + get_update_agent_prompt(if_error)

      # Define inference function for self-consistency
      def single_inference():
          spec_text =  state.model.predict(system_prompt=system_prompt, user_prompt=full_prompt)
          edit_index = spec_text.find("edit:")
          if edit_index != -1:
              only_spec = spec_text[edit_index:].strip()
          else:
              only_spec = "edit:"
          return only_spec
      
      # Use self-consistency with 5 samples
      spec_text, vote_count, all_outputs = self_consistency_inference(
          inference_function=single_inference,
          num_samples=self_consistency_number,
          similarity_threshold=0.9
      )
      
      print(f"=== Self-Consistency Results ===")
      print(f"Vote count: {vote_count}/5")
      print(f"All outputs similarity checked")

        
      print("=== Agent Output (Majority Vote) ===")
      print(spec_text)

      state.edit_specs.append(spec_text)

      # Apply the edit specification using the Builder
      builder = Builder()
      state = builder.build(state, spec_text)
      return state

      # # LLM-based netlist generation (commented out in favor of Builder approach)
      # system_prompt, user_prompt = generate_netlist_prompt(spec_text, netlist)
      # netlist_updated = state.model.predict(system_prompt=system_prompt, user_prompt=user_prompt)
      # 
      # # parse updated netlist using <netlist_start> and <netlist_end> tags
      # start_tag = "<netlist_start>"
      # end_tag = "<netlist_end>"
      # 
      # start_index = netlist_updated.find(start_tag)
      # end_index = netlist_updated.find(end_tag)
      # 
      # # Handle different cases: both tags, only end tag, or no tags
      # if start_index != -1 and end_index != -1:
      #     # Both tags present
      #     netlist_updated_text = netlist_updated[start_index + len(start_tag):end_index].strip()
      # elif start_index != -1:
      #     # Only start tag present
      #     netlist_updated_text = netlist_updated[start_index + len(start_tag):].strip()
      # elif end_index != -1:
      #     # Only end tag present
      #     netlist_updated_text = netlist_updated[:end_index].strip()
      # else:
      #     # No tags present, use the entire output
      #     netlist_updated_text = netlist_updated.strip()
      # 
      # state.netlist_text = netlist_updated_text
      # return state

# def generate_netlist_prompt(spec_text, netlist):
#     system_prompt = f"""Given the original netlist and the NGSpice edit specification, generate the updated netlist by applying the specified edits to the original netlist. Output only the updated netlist in <netlist_start> and <netlist_end> tags without any additional explanations or text."""
#     user_prompt = f"""
# Original netlist:
# {netlist}
# NGSpice edit specification:
# {spec_text}
# Updated netlist:
# <netlist_start>
# """
#     return system_prompt, user_prompt

