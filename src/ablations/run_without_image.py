import sys
import os
import signal
import contextlib
import traceback
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import *
from model import *
from webagent import *
from evaluation import is_correct
from router import *

IMAGE_PATH = "../datasets/cktbench/images"
RESULT_DIR_BASE = "ablation_results/ablation_results_without_image"
FILES_DIR_BASE = "files/ablation_files_without_image"
TIMEOUT_SECONDS = 8 * 60

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


def process_routing(router):
    if router.route() == -1:
        return True, "N/A"
    answer = router.solve()
    retry_count = 0
    max_retries = 3
    while answer == "MCQ_SANITY_CHECK_FAILED" and retry_count < max_retries:
        if router.route(mcq_enabled=True) == -1:
            return True, "N/A"
        answer = router.solve()
        retry_count += 1
    if retry_count >= max_retries and answer == "MCQ_SANITY_CHECK_FAILED":
        return True, "N/A"
    return False, answer


def process_problem(index, df, agent, model, accurate_list, accurate_list_old, total_list):
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

        # Image disabled: pass None so chain_of_thought and answer agents receive no image
        config = {
            'index': index,
            'question': df.iloc[index]['problem'],
            'schema': df.iloc[index]['schema'],
            'netlist': df.iloc[index]['netlist'],
            'domain_knowledge': domain_knowledge,
            'image': None,
            'max_retries': 3
        }

        skip = False
        answer = "N/A"

        try:
            with time_limit(TIMEOUT_SECONDS):
                router = Router(config, model)
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
    parser = argparse.ArgumentParser(description='Ablation: without image in the final prompt.')
    parser.add_argument('--model', type=str, default='openai',
        choices=['openai', 'claude', 'gemini', 'qwen', 'qwen32b', 'llava', 'internvl', 'nemotron', 'glm', 'llama', 'gpt4o'])
    parser.add_argument('--indices', type=str, default=None)
    args = parser.parse_args()

    RESULT_DIR = f"{RESULT_DIR_BASE}_{args.model}"
    FILES_DIR = f"{FILES_DIR_BASE}_{args.model}"

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)

    print(f"Ablation: WITHOUT Image | Model: {args.model}")
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

    df = pd.read_csv('../datasets/cktbench/cktbench.csv')
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

    skip_list = set()
    skip_file = 'files/skip.txt'
    if os.path.exists(skip_file):
        with open(skip_file, 'r') as f:
            skip_list = set(eval(f.read().strip()))
        print(f"Loaded {len(skip_list)} indices to skip")

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
        if index in skip_list:
            continue
        print(f"\n\n=== Processing index {index} ===")

        process_problem(index, df, agent, model, accurate_list, accurate_list_old, total_list)

        try:
            save_summary(RESULT_DIR, accurate_list, total_list)
        except Exception:
            pass

        save_lists(FILES_DIR, {'accurate_list': accurate_list, 'total_list': total_list})

    total = len(total_list)
    print(f"\nFinal Accuracy: {len(accurate_list)/total*100:.2f}%")
