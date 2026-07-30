
def parse_planner_output(question, planner_output):
      # align to the first occurrence of "num_runs"
      idx = planner_output.find("num_runs")
      if idx != -1:
          planner_output = planner_output[idx:]

      # split and strip each line, drop empty ones
      raw_lines = planner_output.strip().splitlines()
      lines = [ln.strip() for ln in raw_lines if ln.strip()]

      if not lines or not lines[0].startswith("num_runs"):
          # fallback: single run with the original problem
          return {
              "num_runs": 1,
              "runs": [{"question": question}],
          }

      # first line: "num_runs 10"
      num_runs = 1
      try:
          num_runs = int(lines[0].split()[1])
      except Exception:
          pass

      run_questions = []
      for line in lines[1:]:
          # lines now start with "run 1: ..."
          if line.startswith("run"):
              parts = line.split(":", 1)
              if len(parts) == 2:
                  run_questions.append(parts[1].strip())

      if len(run_questions) != num_runs or num_runs < 1:
          # fallback again in case of mismatch
          return {
              "num_runs": 1,
              "runs": [{"question": question}],
          }

      return {
          "num_runs": num_runs,
          "runs": [{"question": q} for q in run_questions],
      }


    

def planner_agent_mcq(state):
    """Plan how many simulations are needed based on the question."""
    model = state.model
    question = state.question
    schema = state.schema
    dk = state.domain_knowledge
    
    system_prompt = """
    You are an expert in planning NGSpice simulations based on a given electrical circuit description, and a question.

    Given:
    - The user's question.
    - The schema of the circuit diagram
    - Domain knowledge about the question

    Your job:
    1. If the question contains words like "range" "change" "maximum" "minimum" about input values, output multiple simulations, each with a specific value for the changing input(s).

    2. If the question contains words like "range" "change" "maximum" "minimum" about output values, decide if multiple simulations are needed to capture the range of output values, and if so, output multiple simulations with specific input values to capture that range.

    3. If multiple are needed, describe clearly:
    - number of runs
    - the rephrased question with specific values for each run (you must reserve any values provided in the original question in the rephrased question, and only vary the values that are implied to change in the original question.).

    4. If the question contains multiple choice options where each option is an equation instead of a direct numeric answer: run four simulations, one for each option, rephrasing the question for each run to verify the correctness of each choice.

    5. Limit the number of multiple runs to 5.

    Output format:
    num_runs X
    run 1: (rephrased question for run 1)
    run 2: (rephrased question for run 2)
    ...
    run X: (rephrased question for run X)

    Example 1:
    Input:
    Question: In this circuit, R2 = 1k ohm. What is the maximum voltage across R2 when the input voltage V1 varies from 5V to 15V?
    
    Schema:
    V1 V N0 0 V (Voltage source across points N0 and ground, where voltage value V is unassigned)
    R2 N0 0 R (Resistor R2 connected between point N0 and ground, where resistance R is unassigned)

    Domain knowledge:
    Voltage across a resistor can be calculated using Ohm's law: V = I * R, where V is voltage, I is current, and R is resistance.

    Output (You MUST follow this format exactly):
    num_runs 2
    run 1: In this circuit, R2 = 1k ohm. What is the voltage across R2 when the input voltage V1 is 5V?
    run 2: In this circuit, R2 = 1k ohm. What is the voltage across R2 when the input voltage V1 is 15V?

    Example 2:
    Input:
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
    Question: An operational amplifier circuit is shown in the <image>. The output of the circuit for a given input \( v_i \) is ______.

    Choices  
    A. \( -\left(\frac{R_2}{R_1}\right)v_i \)  
    B. \( -(1 + \frac{R_2}{R_1})v_i \)  
    C. \( (1 + \frac{R_2}{R_1})v_i \)  
    D. \( +V_{\text{sat}} \) or \( -V_{\text{sat}} \)
    
    Schema:
    vi V N3 0 NA
    R1 R N1 0 NA
    R2 R N1 N2 NA
    A1 OpAMP N1 N3 N2 NA; N1 non-inverting input, N3 inverting input
    A2 OpAMP N2 N4 N5 NA; N2 non-inverting input, N4 inverting input
    R3 R N4 N2 NA   ; between minus input of A2 and output of A1
    R4 R N4 N5 NA   ; between minus input of A2 and final output
    vo V N5 0 NA
    Labels:
    +Vsat V N6 0; A1 positive supply
    -Vsat V N7 0; A1 negative supply
    +Vsat V N8 0; A2 positive supply
    -Vsat V N9 0; A2 negative supply

    Domain knowledge:
    Here are the key vocabulary and terms related to the field of electrical engineering in the given question:

    1. Operational Amplifier (Op-Amp): A high-gain electronic voltage amplifier with differential inputs and, usually, a single-ended output. It is widely used in analog circuits for signal conditioning, filtering, and mathematical operations such as addition, subtraction, integration, and differentiation.

    2. Circuit: A closed loop or pathway that allows electric current to flow, consisting of various electronic components like resistors, capacitors, and amplifiers.

    3. Output (\(v_o\)): The voltage or signal produced by the circuit as a response to the input signal.

    4. Input (\(v_i\)): The voltage or signal applied to the circuit for processing or amplification.

    5. \(R_1\) and \(R_2\): These are resistors in the circuit. Resistors are passive electrical components that limit or regulate the flow of electrical current in a circuit. The values of \(R_1\) and \(R_2\) affect the behavior and gain of the amplifier circuit.

    6. Gain: The ratio of the output signal to the input signal in an amplifier, indicating how much the input signal is amplified.

    7. \(V_{\text{sat}}\): The saturation voltage of the operational amplifier. It is the maximum output voltage the op-amp can provide, limited by its power supply voltage.

    The question essentially involves analyzing the configuration of the operational amplifier circuit (like inverting or non-inverting amplifier) and determining the expression for the output voltage \(v_o\) based on the input voltage \(v_i\) and the resistor values \(R_1\) and \(R_2\).

    Would you like me to analyze and answer the question based on the image of the circuit? If so, please upload the image of the circuit.

    Output (You MUST follow this format exactly, and you MUST use the word verify here):
    num_runs 4
    run 1: An operational amplifier circuit is shown in the <image>. Verify if the output of the circuit for a given input \( v_i \) is \( -\left(\frac{R_2}{R_1}\right)v_i \)  
    run 2: An operational amplifier circuit is shown in the <image>. Verify if the output of the circuit for a given input \( v_i \) is \( -(1 + \frac{R_2}{R_1})v_i \)
    run 3: An operational amplifier circuit is shown in the <image>. Verify if the output of the circuit for a given input \( v_i \) is \( (1 + \frac{R_2}{R_1})v_i \)
    run 4: An operational amplifier circuit is shown in the <image>. Verify if the output of the circuit for a given input \( v_i \) is \( +V_{\text{sat}} \) or \( -V_{\text{sat}} \)
    """
    user_prompt = f"""
    Provide information about number of runs with the output in the specified format according to system instructions.
    Question: {question}
    Schema:\n{schema}
    Domain knowledge:\n{dk}
    Output:
    """

    model_output = model.predict(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    print("=== Agent Output ===")
    print(model_output)
    plan = parse_planner_output(question, model_output)
    states = []
    for i in range(plan["num_runs"]):
        new_state = state.copy_for_new_run()
        new_state.question = plan["runs"][i]["question"]
        states.append(new_state)

    return states


