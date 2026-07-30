import subprocess
import tempfile
from typing import Optional, Tuple
from evaluation import is_correct
from model import *
import pandas as pd
import argparse
import os
import time


def run_ngspice_netlist(netlist_text: str, timeout_s: int = 30) -> Tuple[Optional[str], str, str, int]:
    """
    Run ngspice (-b batch mode) on a netlist string.

    Returns:
        (error, stdout, stderr, returncode)

        error is:
          - None if the run looks successful
          - "ngspice_error" if ngspice reports an error or exits nonzero
          - "ngspice_exception" if Python raises (timeout, missing binary, etc.)
    """
    netlist = (netlist_text or "").replace("</netlist_start>", "")

    try:
        with tempfile.NamedTemporaryFile(suffix=".cir", delete=False, mode="w") as f:
            f.write(netlist)
            path = f.name

        proc = subprocess.run(
            ["ngspice", "-b", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode

        error = None
        if returncode != 0 or "error" in stderr.lower():
            error = "ngspice_error"

        return error, stdout, stderr, returncode

    except Exception as e:
        return "ngspice_exception", "", str(e), -1


def baseline_ngspice(model, question, schema):
    system_prompt = "Given the circuit schema, generate an ngspice program from the schema to solve the question."
    user_prompt = f"""
Schema:
{schema}
Question:
{question}

Provide the ngspice program only without any explanations between the tags <ngspice_program> and </ngspice_program>.
"""
    
    output = model.predict(system_prompt, user_prompt, max_tokens=512)
    # time.sleep(0.5)  # to avoid rate limiting
    # parse the output to extract the ngspice program between the tags
    start_tag = "<ngspice_program>"
    end_tag = "</ngspice_program>"
    start_index = output.find(start_tag)
    end_index = output.find(end_tag)
    if start_index != -1 and end_index != -1:
        ngspice_program = output[start_index + len(start_tag):end_index].strip()
    else:
        ngspice_program = output.strip()

    # run ngspice program
    error, ngspice_stdout, ngspice_stderr, returncode = run_ngspice_netlist(ngspice_program)

    if error:
        return False
    elif "error" in ngspice_stdout.lower():
        return False
    else:
        return True
#     else:
#         prompt = f"""The following is the output of the ngspice program:
# {ngspice_stdout}
# Question:
# {question}
# Provide the final answer to the question based on the output above, in one word, no explanations.
# """
#         answer = model.predict(system_prompt="Given the ngspice output and the question, provide the final answer to the question.", user_prompt=prompt, reasoning=False, max_tokens=64)
#         is_correct_flag = is_correct(answer.strip(), gt_ans)
#         return True, is_correct_flag
    

if __name__ == "__main__":
    # define model name as system argument
    with open("../private/azureapikey.txt", "r") as f:
        azure_api_key = f.read().strip()
    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
    
    with open("../private/openrouterapikey.txt", "r") as f:
        openrouter_api_key = f.read().strip()
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai")
    args = parser.parse_args()

    # Load dataset
    executed = []
    # accurate = []
    total = []
    df = pd.read_csv('../datasets/cktbench/cktbench.csv')
    # load lists if existing
    try:
        with open(f'baselines/{args.model}_baseline_ngspice_executed.txt', 'r') as f:
            executed = [int(line.strip()) for line in f.readlines()]
    except:
        executed = []
    # try:
    #     with open(f'baselines/{args.model}_baseline_ngspice_accurate.txt', 'r') as f:
    #         accurate = [int(line.strip()) for line in f.readlines()]
    # except:
    #     accurate = []
    try:
        with open(f'baselines/{args.model}_baseline_ngspice_total.txt', 'r') as f:
            total = [int(line.strip()) for line in f.readlines()]
    except:
        total = []

    model = Model(args.model) # Initialize your model here
    # define the start index
    start_index = max(total) + 1 if total else 0


    for index in range(start_index, len(df)):
        print(f"Processing index: {index}")
        total.append(index)
        question = df.iloc[index]['problem']
        schema = df.iloc[index]['schema']
        gt_ans = df.iloc[index]['solution']

        try:
            executed_flag = baseline_ngspice(model, question, schema)
            time.sleep(0.5)  # to avoid rate limiting
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            executed_flag= False

        
        if executed_flag:
            executed.append(index)
        # if is_correct_flag:
        #     accurate.append(index)
        # write to a file every iteration with percentage of executed and accurate
        with open(f'baselines/{args.model}_baseline_ngspice_results.txt', 'w') as f:
            f.write(f"Executed: {len(executed)}/{len(total)} ({len(executed)/len(total)*100:.2f}%)\n")
            # f.write(f"Accurate: {len(accurate)}/{len(total)} ({len(accurate)/len(total)*100:.2f}%)\n")
            # also print the lists
            f.write(f"Executed indices: {executed}\n")
            # f.write(f"Accurate indices: {accurate}\n")
            f.write(f"Total indices: {total}\n")
        # save lists
        with open(f'baselines/{args.model}_baseline_ngspice_executed.txt', 'w') as f:
            for idx in executed:
                f.write(f"{idx}\n")
        # with open(f'baselines/{args.model}_baseline_ngspice_accurate.txt', 'w') as f:
        #     for idx in accurate:
        #         f.write(f"{idx}\n")
        with open(f'baselines/{args.model}_baseline_ngspice_total.txt', 'w') as f:
            for idx in total:
                f.write(f"{idx}\n")

