import sys
import os
import signal
import contextlib
import traceback
import argparse

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import *
from model import *
from webagent import *
from evaluation import is_correct
from router import *

RESULT_DIR_BASE = "ablation_results/amsnet_ablation_results_without_M"
FILES_DIR_BASE = "files/amsnet_ablation_files_without_M"
TIMEOUT_SECONDS = 10 * 60

RESULT_DIR = None


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class RouterWithoutM(Router):
    """Router that skips the output/measurement specification (M) agent."""

    def _run_static_sequence(self, sim_state):
        sim_state = update_agent(sim_state, circuit_specification_agent_prompt, self_consistency_number=1)
        sim_state = update_agent(sim_state, analysis_specification_agent_prompt)
        sim_state = run_ngspice_node(sim_state)
        return sim_state

    def _execute_recovery_sequence(self, sim_state, recovery_sequence):
        agent_map = {
            "circuit_specification_agent": circuit_specification_agent_prompt,
            "analysis_specification_agent": analysis_specification_agent_prompt,
            "transient_analysis_specification_agent": transient_analysis_specification_agent_prompt,
        }
        for agent_name in recovery_sequence:
            if agent_name in agent_map:
                sim_state = update_agent(sim_state, agent_map[agent_name])
                sim_state = run_ngspice_node(sim_state)
                if not sim_state.last_error:
                    return sim_state
                else:
                    sim_state = error_diagnosis_agent(sim_state)
        return sim_state


def process_routing(router):
    if router.route() == -1:
        return True, "N/A"
    answer = router.solve()
    # MCQ retry disabled for amsnet (max_retries=0)
    retry_count = 0
    max_retries = 0
    while answer == "MCQ_SANITY_CHECK_FAILED" and retry_count < max_retries:
        if router.route(mcq_enabled=True) == -1:
            return True, "N/A"
        answer = router.solve()
        retry_count += 1
    if retry_count >= max_retries and answer == "MCQ_SANITY_CHECK_FAILED":
        return True, "N/A"
    return False, answer


def process_problem(index, df, agent, model, accurate_list, accurate_list_old, total_list):
    image = df.iloc[index]['image_path']
    pil_image = Image.open(image)

    if index in accurate_list_old:
        answer = accurate_list_old[index]
        accurate_list[index] = answer
        total_list[index] = answer
        return

    f2 = open(f"{RESULT_DIR}/{index}.txt", "w")
    sys.stdout = f2

    try:
        print("\033[91mProblem:\033[0m")
        print(df.iloc[index]['problem'])
        print("\033[91mSchema:\033[0m")
        print(df.iloc[index]['schema'])

        domain_knowledge = agent.answer_with_cache(df.iloc[index]['problem'])
        print("\033[91mDomain knowledge:\033[0m")
        print(domain_knowledge)

        config = {
            'index': index,
            'question': df.iloc[index]['problem'],
            'schema': df.iloc[index]['schema'],
            'netlist': df.iloc[index]['schema'],
            'domain_knowledge': domain_knowledge,
            'image': pil_image,
            'max_retries': 2
        }

        skip = False
        answer = "N/A"

        try:
            with time_limit(TIMEOUT_SECONDS):
                router = RouterWithoutM(config, model)
                skip, answer = process_routing(router)
        except TimeoutException as e:
            skip = True
            answer = "N/A"
            print(f"Timeout: {e}")
            traceback.print_exc()
        except Exception:
            skip = True
            answer = "N/A"
            traceback.print_exc()

        if skip:
            with open('files/skips.txt', 'a') as f:
                f.write(f"{index}\n")
            return

        print("\033[91mSolution:\033[0m")
        print(df.iloc[index]['solution'])

        if is_correct(answer, df.iloc[index]['solution']):
            accurate_list[index] = answer
        total_list[index] = answer

    finally:
        sys.stdout = sys.__stdout__
        f2.close()


def save_summary(result_dir, accurate_list, total_list):
    with open(f"{result_dir}/summary.txt", "w") as f:
        total = len(total_list)
        f.write(f"Accurate: {len(accurate_list)}\n")
        f.write(f"Total: {total}\n")
        f.write(f"Accuracy: {len(accurate_list)/total*100:.2f}%\n\n")
        f.write(f"Accurate indices: {list(accurate_list.keys())}\n")
        f.write(f"Total indices: {list(total_list.keys())}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AMSNet ablation: without output/measurement specification agent (M).')
    parser.add_argument('--model', type=str, default='openai',
        choices=['openai', 'claude', 'gemini', 'qwen', 'qwen32b', 'llava', 'internvl', 'nemotron', 'glm', 'llama'])
    parser.add_argument('--indices', type=str, default=None)
    args = parser.parse_args()

    RESULT_DIR = f"{RESULT_DIR_BASE}_{args.model}"
    FILES_DIR = f"{FILES_DIR_BASE}_{args.model}"

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)

    print(f"AMSNet Ablation: WITHOUT Output/Measurement Specification Agent (M) | Model: {args.model}")
    print(f"Results: {RESULT_DIR}")

    with open('../private/webapikey.txt', 'r') as f:
        web_api_key = f.read().strip()
    with open('../private/azureapikey.txt', 'r') as f:
        azure_api_key = f.read().strip()
    with open('../private/openrouterapikey.txt', 'r') as f:
        openrouter_api_key = f.read().strip()

    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
    os.environ["WEBAGENT_API_KEY"] = web_api_key
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

    df = pd.read_csv('../datasets/netq/netq.csv')
    print(f"DataFrame loaded of length: {len(df)}")

    lists = load_lists(FILES_DIR)
    accurate_list = lists.get('accurate_list', {})
    total_list = lists.get('total_list', {})

    if 'accurate_list_old' in lists:
        accurate_list_old = lists['accurate_list_old']
    elif 'accurate_list' in lists and lists['accurate_list']:
        accurate_list_old = lists['accurate_list']
    else:
        accurate_list_old = {}

    agent = WebAgent(api_key=os.getenv("WEBAGENT_API_KEY"))
    model = Model(args.model)

    specific_indices = None
    if args.indices:
        specific_indices = [int(i.strip()) for i in args.indices.split(',')]
        print(f"Will process specific indices: {specific_indices}")

    start_index = max(total_list.keys()) + 1 if total_list else 0
    if not specific_indices:
        print(f"Resuming from index: {start_index}")

    for index in range(start_index, len(df)):
        if specific_indices and index not in specific_indices:
            continue
        print(f"\n\n=== Processing index {index} ===")

        process_problem(index, df, agent, model, accurate_list, accurate_list_old, total_list)

        try:
            save_summary(RESULT_DIR, accurate_list, total_list)
        except Exception:
            pass

        save_lists(FILES_DIR, {'accurate_list': accurate_list, 'total_list': total_list})

    total = len(total_list)
    print(f"\nFinal Accuracy: {len(accurate_list)/total*100:.2f}%" if total else "\nFinal Accuracy: N/A (no items processed)")
