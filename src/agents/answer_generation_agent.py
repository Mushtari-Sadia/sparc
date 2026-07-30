import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from self_consistency import self_consistency_inference


def chain_of_thought_reasoning(question, states):
    model = states[0].model
    schema = states[0].schema
    system_prompt = "Given the diagram, the schema of the diagram and the question, do step by step chain-of-thought reasoning of the problem, and provide the solution."
    user_prompt = f"""
Schema:
{schema}
    Question:
{question}
Provide the chain of thought reasoning to solve the problem in a step by step manner:
# Reasoning:
Then provide the final answer:
"""
    cot_answer = model.predict_vl(system_prompt=system_prompt, user_prompt=user_prompt, images=states[0].image)
    return cot_answer


def _get_answer_question_from_ngspice_output(question, states):

    runs_info = ""
    schema = states[0].schema
    dk = states[0].domain_knowledge
    cot_answer = states[0].cot_answer
    q_lower = question.lower()
    derived_query = ("____" in question) or any(k in q_lower for k in [
        "find", "determine", "calculate", "equivalent", "norton", "thevenin", "transfer function", "gain", "input impedance", "output impedance"
    ])


    for index, state in enumerate(states):
        problem = state.question
        ngspice_program = state.netlist_text
        ngspice_output = state.ngspice_stdout
        runs_info += f"Question {index}: {problem}\n"
        runs_info += f"Ngspice program {index}:\n{ngspice_program}\n"
        runs_info += f"Ngspice program output {index}:\n{ngspice_output}\n\n" 



    system_prompt = f"""
    You are an expert in circuit analysis and in reading ngspice simulation outputs.

    You will receive:
    1. A schema of a circuit diagram, and the diagram.
    2. One or more questions about the circuit, an associated nspice program to simulate the circuit according to the question, and the text output produced by running that ngspice program, including any printed voltages, currents, and other quantities.
    3. The final question to answer from the ngspice output(s)
    4. The chain-of-thought reasoning answer to the question based on the circuit diagram, schema, and question.
    5. Additional domain knowledge that may include hints about the question, typical simplifications, or known formulas.

    You have to do step by step reasoning using the following steps:
    - First, interpret the ngspice output(s), and use the output to provide the correct answer.
    - Even if the ngspice output(s) contain warnings, still use any information provided in the output to find the correct solution.
    - Identify which quantities from the output(s) are relevant to the question.
    - Decide and show what calculations are needed to answer the question.
    - Use the available tools for every numeric calculation.
    - If the question contain multiple choice options with equations, verify each equation by calling the appropriate tools, and choose the correct option.
    - Explain the sequence of tool calls you make.
    - Produce a clear final answer to the original problem.

    You must not do numeric calculations directly. Whenever you need to compute a quantity, you must call one or more of the tools below. Your job is to choose:
    - which tool to call,
    - which arguments to pass (taken from ngspice output or the question),
    - and in what order to call tools.

    Available tools (conceptual description):

    1. ohms_law_voltage(R, I)
        - Returns V for a branch using V = I * R.

    2. ohms_law_current(V, R)
        - Returns I for a branch using I = V / R.

    3. ohms_law_resistance(V, I)
        - Returns R for a branch using R = V / I.

    4. voltage_add(values)
        - Returns the sum of several voltage values, used for series voltage relations or KVL.

    5. voltage_difference(v_high, v_low)
        - Returns the voltage drop between two nodes, often used in KVL and to interpret node voltages.

    6. current_add(values)
        - Returns the algebraic sum of currents at a node, used for KCL checks or to solve for an unknown current.

    7. impedance_from_voltage_current(V, I)
        - Returns complex impedance Z seen at a port using Z = V / I.

    8. impedance_series(impedances)
        - Returns the total impedance of elements in series by summing complex impedances.

    9. impedance_parallel(impedances)
        - Returns the equivalent impedance of elements in parallel using the usual parallel rule.

    10. complex_add(values)
        - Adds complex quantities such as phasor voltages or currents.

    11. complex_multiply(a, b)
        - Multiplies two complex quantities.

    12. complex_divide(a, b)
        - Divides two complex quantities.

    13. magnitude(x)
        - Returns the magnitude of a real or complex quantity, for example the magnitude of voltage or impedance.

    14. phase_deg(x)
        - Returns the phase in degrees of a complex quantity.

    15. complex_from_polar(magnitude_value, phase_deg_value)
        - Builds a complex phasor from magnitude and phase in degrees.

    16. equation_verification(lhs, rhs, tolerance)
        - Verifies if lhs and rhs are equal within a specified tolerance. Use this to verify equations in multiple choice options to choose the correct one. 

    You may conceptually use these tools to implement KCL and KVL:
    - For KCL, use current_add to form sums of currents entering and leaving a node.
    - For KVL, use voltage_add or voltage_difference to express sums of voltage rises and drops around a loop.

    Guidelines:
    1. Use the ngspice output(s) as the main source of truth. Use domain knowledge or the netlist only to interpret what each printed quantity means.

    2. When the question asks for any equivalent resistance or equivalent impedance (for example R_eq, Z_in, Z_eq, input impedance, Thevenin resistance, Thevenin or Norton impedance):
        - You must always base it on a voltage and a current taken from the ngspice output at the correct terminals.
        - First identify the correct voltage across the port or pair of nodes from the ngspice output.
        - Then identify the corresponding current through the port or source from the ngspice output.
        - Then compute the resistance or impedance only by calling ohms_law_resistance(V, I), impedance_from_voltage_current(V, I), or complex_divide(V, I).
        - Do NOT derive equivalent resistance or impedance only from component values or by manual algebra. Always use the ngspice voltage and current at the correct points and divide using the tools.

    3. Maintain the polarity in the question. If the question asks for voltage difference between node A and B (V_AB), you must find V(A) - V(B) from the ngspice output, even if the output is negative. You must not swap the order of subtraction.

    4. If the input source has AC amplitude 1, you can compute input impedance as Z = V(node) / I(source). You must do this by calling impedance_from_voltage_current with the node voltage and source current taken from the ngspice output, and then magnitude and phase if needed.

    5. Do not invent data that is not present in the ngspice output or implied by basic circuit laws. If the needed quantity is missing, say so and explain briefly.

    6. If the final question asks for an unknown (blank "____", or words like find, determine, calculate, equivalent, Norton, Thevenin):
        - Do NOT accept a value just because it appears as a DC/AC value in the ngspice program (this can be a guessed or hard-coded number).
        - You must derive the requested quantity from measured voltages and currents in the ngspice output (for example, short-circuit current for Norton, open-circuit voltage for Thevenin, or Z = V/I for impedance).
        - If the ngspice output does not include the needed voltage or current to compute it, you must say the measurement is missing.

    7. If the question asks for power factor, cos(phi), phase, leading or lagging:
        - You must use quantities that encode phase, such as complex V and I, complex power, or explicit phase output.
        - If the ngspice output only contains node voltages without source currents or phase information, you must say the required measurement is missing.

    8. If the question asks whether a quantity increases or decreases due to a modification (e.g., "R2 is open-circuited", "switch changes", "component removed"):
        - You must compare the quantity across two cases: baseline and modified.
        - If the runs_info contains only one case, you must state that the baseline or modified case is missing, so the direction cannot be verified from ngspice output alone.
        - Do NOT answer direction-of-change purely from intuition when comparison data is missing.


    
    9. Keep the explanation compact. A short explanation followed by the final numeric answer and option letter when applicable is preferred.

    10. For every numeric step, think about which tool is appropriate and call that tool instead of doing arithmetic yourself.

    11. When the question has multiple choice options with equations, you must verify each equation by calling the appropriate tools, and choose the correct option.
    12. When the question has multiple choice options with non-numeric values or variable names, you must compute the numeric value of each variable and verify which option is closest to the computed value using tools.

    Output format:
    - Show the sequence of tool calls.
    - Provide a short explanation if helpful.
    - Then clearly state the final answer.
    - If the problem asks for an option letter, answer with the option letter.
    """

    user_prompt = f"""
    Derived quantity guard: {derived_query}

    1. A schema of a circuit diagram:
    Schema of the circuit (If there is a labels section, specifically look at it to check if any labels are important to interpret the question and answer it):
    {schema} 

    2. One or more questions about the circuit, an associated nspice program to simulate the circuit according to the question, and the text output produced by running that ngspice program, including any printed voltages, currents, and other quantities:
    {runs_info}

    3. The final question to answer from the ngspice output(s):
    Final question to answer:
    {question}

    4. The chain-of-thought reasoning answer to the question based on the circuit diagram, schema, and question.
    Chain-of-thought reasoning answer:
    {cot_answer}
    
    5. Additional domain knowledge that may include hints about the question, typical simplifications, or known formulas.
    Domain knowledge:
    {dk}

    Using the provided Ngspice output(s) above (if there are more than one, analyze all) and by calling the available tools for any numeric calculations, answer the question now in the following format:
    ### Interpretation of ngspice output(s)
    - For each ngspice output, identify which printed quantities are relevant to the question, and provide detailed reasoning of how to interpret the output to find the solution.
    ### Tool calls
    - For every numeric calculation needed to answer the question, show which tool you call, with what arguments, and explain why.
    ### Sequence of tool calls
    - show the sequence of tool calls and explain reasoning.
    ### Final analysis
    - question - {question}
    - final answer and explanation based on tool outputs. You MUST explain your answer.
    <start_final_answer> output the final answer in one word here. If the question asks for an option letter, output only the option letter here. No explanations or any other text permitted. <end_final_answer>
    """
    return system_prompt, user_prompt

def check_mcq_sanity(question, answer):
    mcq_indicators = mcq_indicators = ["which of the following", "the correct", "choose", "multiple choice", "choices"]
    answer_options = ["A", "B", "C", "D", "E"]
    if any(indicator in question.lower() for indicator in mcq_indicators):
        if not any(option in answer for option in answer_options) or len(answer) > 3:
            return False
        
    return True

    
def extract_answer(ngspice_QA_output: str) -> str:
    # extract answer between <start_final_answer> and <end_final_answer> tags
    start_tag = "<start_final_answer>"
    end_tag = "<end_final_answer>"
    start_index = ngspice_QA_output.find(start_tag)
    end_index = ngspice_QA_output.find(end_tag)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return ngspice_QA_output[start_index + len(start_tag):end_index].strip()
    else:
        # return last 50 characters as fallback
        return ngspice_QA_output[-50:].strip()
    


def answer_generation_agent(question, states, num_samples=1):
    model = states[0].model
    
    # print("\033[92m[System Prompt]\033[0m")
    # print(system_prompt)
    # print("\033[92m[User Prompt]\033[0m")
    # print(user_prompt)
    
    # Define inference function for self-consistency
    def single_inference():
        states[0].cot_answer = chain_of_thought_reasoning(question, states)
        system_prompt, user_prompt = _get_answer_question_from_ngspice_output(question,states)
        if model.name == "qwen" or model.name == "llama" or model.name == "glm":
            response = model.predict_tool(system_prompt=system_prompt, user_prompt=user_prompt, images=None)
        # without image ablation
        # system_prompt, user_prompt = _get_answer_question_from_ngspice_output_without_tools(question,states)
        # response = model.predict(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=1024)
        elif model.name == "qwen32b":
            system_prompt, user_prompt = _get_answer_question_from_ngspice_output_without_tools(question,states)
            response = model.predict(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=1024)
        else:
            response = model.predict_tool(system_prompt=system_prompt, user_prompt=user_prompt, images=states[0].image)
        print("\033[92m[Agent Response]\033[0m")
        print(response)
        extracted_answer = extract_answer(response)
        return extracted_answer
    
    answer, vote_count, all_outputs = self_consistency_inference(
        inference_function=single_inference,
        num_samples=num_samples,
        similarity_threshold=0.9
    )
    
    states[0].answer = answer  # Store the majority voted answer
    
    # print(f"=== Self-Consistency Results ===")
    # print(f"Vote count: {vote_count}/1")
    # print("=== Agent Output (Majority Vote) ===")
    # print(states[0].answer)
    
    # extracted_answer = extract_answer(states[0].answer)

    if not check_mcq_sanity(question, answer):
        return "MCQ_SANITY_CHECK_FAILED"
    # print("\033[92m[Extracted Answer]\033[0m")
    # print(answer)

    return answer

# alternative to tool calls

def _get_answer_question_from_ngspice_output_without_tools(question, states):

    runs_info = ""
    schema = states[0].schema
    dk = states[0].domain_knowledge
    cot_answer = states[0].cot_answer
    q_lower = question.lower()
    derived_query = ("____" in question) or any(k in q_lower for k in [
        "find", "determine", "calculate", "equivalent", "norton", "thevenin", "transfer function", "gain", "input impedance", "output impedance"
    ])

    for index, state in enumerate(states):
        problem = state.question
        ngspice_program = state.netlist_text
        ngspice_output = state.ngspice_stdout
        runs_info += f"Question {index}: {problem}\n"
        runs_info += f"Ngspice program {index}:\n{ngspice_program}\n"
        runs_info += f"Ngspice program output {index}:\n{ngspice_output}\n\n" 



    system_prompt = f"""
    You are an expert in circuit analysis and in reading ngspice simulation outputs.

    You will receive:
    1. A schema of a circuit diagram, and the diagram.
    2. One or more questions about the circuit, an associated nspice program to simulate the circuit according to the question, and the text output produced by running that ngspice program, including any printed voltages, currents, and other quantities.
    3. The final question to answer from the ngspice output(s)
    4. The chain-of-thought reasoning answer to the question based on the circuit diagram, schema, and question.
    5. Additional domain knowledge that may include hints about the question, typical simplifications, or known formulas.

    You have to do step by step reasoning using the following steps:
    - First, interpret the ngspice output(s), and use the output to provide the correct answer.
    - Even if the ngspice output(s) contain warnings, still use any information provided in the output to find the correct solution.
    - Identify which quantities from the output(s) are relevant to the question.
    - Decide and show what calculations are needed to answer the question.
    - If the question contain multiple choice options with equations, verify each equation.

    Guidelines:
    1. Use the ngspice output(s) as the main source of truth. Use domain knowledge or the netlist only to interpret what each printed quantity means.

    2. When the question asks for any equivalent resistance or equivalent impedance (for example R_eq, Z_in, Z_eq, input impedance, Thevenin resistance, Thevenin or Norton impedance):
        - You must always base it on a voltage and a current taken from the ngspice output at the correct terminals.
        - First identify the correct voltage across the port or pair of nodes from the ngspice output.
        - Then identify the corresponding current through the port or source from the ngspice output.
        - Then compute the resistance or impedance only by using ohms_law_resistance(V, I), impedance_from_voltage_current(V, I), or complex_divide(V, I).
        - Do NOT derive equivalent resistance or impedance only from component values or by manual algebra. Always use the ngspice voltage and current at the correct points and divide.

    3. Maintain the polarity in the question. If the question asks for voltage difference between node A and B (V_AB), you must find V(A) - V(B) from the ngspice output, even if the output is negative. You must not swap the order of subtraction.

    4. If the input source has AC amplitude 1, you can compute input impedance as Z = V(node) / I(source). You must do this by calling impedance_from_voltage_current with the node voltage and source current taken from the ngspice output, and then magnitude and phase if needed.

    5. Do not invent data that is not present in the ngspice output or implied by basic circuit laws. If the needed quantity is missing, say so and explain briefly.
    
    6. If the final question asks for an unknown (blank "____", or words like find, determine, calculate, equivalent, Norton, Thevenin):
        - Do NOT accept a value just because it appears as a DC/AC value in the ngspice program (this can be a guessed or hard-coded number).
        - You must derive the requested quantity from measured voltages and currents in the ngspice output (for example, short-circuit current for Norton, open-circuit voltage for Thevenin, or Z = V/I for impedance).
        - If the ngspice output does not include the needed voltage or current to compute it, you must say the measurement is missing.
    
    7. If the question asks for power factor, cos(phi), phase, leading or lagging:
        - You must use quantities that encode phase, such as complex V and I, complex power, or explicit phase output.
        - If the ngspice output only contains node voltages without source currents or phase information, you must say the required measurement is missing.
    
    8. If the question asks whether a quantity increases or decreases due to a modification (e.g., "R2 is open-circuited", "switch changes", "component removed"):
        - You must compare the quantity across two cases: baseline and modified.
        - If the runs_info contains only one case, you must state that the baseline or modified case is missing, so the direction cannot be verified from ngspice output alone.
        - Do NOT answer direction-of-change purely from intuition when comparison data is missing.

    
    9. Keep the explanation compact. A short explanation followed by the final numeric answer and option letter when applicable is preferred.

    Output format:
    - Provide a short explanation of the reasoning.
    - Then clearly state the final answer.
    - If the problem asks for an option letter, answer with the option letter.
    """

    user_prompt = f"""
    Derived quantity guard: {derived_query}

    1. A schema of a circuit diagram:
    Schema of the circuit (If there is a labels section, specifically look at it to check if any labels are important to interpret the question and answer it):
    {schema} 

    2. One or more questions about the circuit, an associated nspice program to simulate the circuit according to the question, and the text output produced by running that ngspice program, including any printed voltages, currents, and other quantities:
    {runs_info}

    3. The final question to answer from the ngspice output(s):
    Final question to answer:
    {question}

    4. The chain-of-thought reasoning answer to the question based on the circuit diagram, schema, and question.
    Chain-of-thought reasoning answer:
    {cot_answer}
    
    5. Additional domain knowledge that may include hints about the question, typical simplifications, or known formulas.
    Domain knowledge:
    {dk}

    Using the provided Ngspice output(s) above (if there are more than one, analyze all) answer the question now in the following format:
    ### Interpretation of ngspice output(s)
    - For each ngspice output, identify which printed quantities are relevant to the question, and provide detailed reasoning of how to interpret the output to find the solution.
    
    ### Final analysis
    - question - {question}
    - final answer and explanation. You MUST explain your answer.
    <start_final_answer> output the final answer in one word here. If the question asks for an option letter, output only the option letter here. No explanations or any other text permitted. <end_final_answer>
    """
    return system_prompt, user_prompt