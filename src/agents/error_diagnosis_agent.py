
def error_diagnosis_agent(state):
    model = state.model
    question = state.question
    schema = state.schema
    ngspice_program = state.netlist_text
    ngspice_output = state.ngspice_stdout + "\n" + state.ngspice_stderr
    system_prompt = """
        You are an expert in circuit analysis and NGSpice. You will be given:
        1. A natural language question about a circuit.
        2. A high level circuit schema that describes components and labels.
        3. An NGSpice netlist that was generated from the schema.
        4. The NGSpice simulation log, which will contain warnings and errors.

        Your job is to analyze the NGSpice simulation log and identify the cause of the error statement. In many cases, the error statement points to an underlying issue in the netlist or circuit configuration. Your output should include:
        - A detailed reasoning section that explains the likely cause of the error, referencing specific parts of the netlist or schema as needed. Reason step-by-step why the error occurred, and what part of the netlist is responsible. Provide a detailed explanation and diagnosis.
    """

    user_prompt = f"""
    ### Question:
    {question}
    ### Schema:
    {schema}
    ### NGSpice Netlist:
    {ngspice_program}
    ### NGSpice Simulation Log:
    {ngspice_output}

    Provide your output in the following section, following system instructions EXACTLY:
    ### Output:
    """
    output = model.predict(system_prompt, user_prompt)
    print("=== Error Analysis Output ===")
    print(output)
    state.error_analysis = output
    return state


