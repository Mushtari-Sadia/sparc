import sys
import os
import signal
import contextlib
import json
import traceback
import argparse

import pandas as pd
from PIL import Image

from utils import *
from model import *
from webagent import *
from evaluation import is_correct
from router import *

# Configuration

RESULT_DIR_BASE = "results/main_agent_amsnet"
FILES_DIR_BASE = "files/main_agent_amsnet"
TIMEOUT_SECONDS = 10 * 60  # 10 minutes


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Context manager to limit execution time using signals."""
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
    """
    Process routing and solving with automatic MCQ retry on sanity check failure.
    
    Args:
        router: Router instance configured with problem parameters
    
    Returns:
        tuple: (skip: bool, answer: str)
            - skip: True if routing failed and should skip this problem
            - answer: The solution answer or "N/A" if skipped
    """
    # First routing attempt
    if router.route() == -1:
        return True, "N/A"
    
    answer = router.solve()
    
    # Retry with MCQ enabled if sanity check failed (max 3 retries)
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
    """Process a single problem from the dataset."""

    # Load image
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
        # Print problem details
        print("\033[91mProblem:\033[0m")
        print(df.iloc[index]['problem'])
        print("\033[91mSchema:\033[0m")
        print(df.iloc[index]['schema'])
        
        # Get domain knowledge
        domain_knowledge = agent.answer_with_cache(df.iloc[index]['problem'])
        print("\033[91mDomain knowledge:\033[0m")
        print(domain_knowledge)
        
        # Configure and run router
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
                router = Router(config, model)
                skip, answer = process_routing(router)
        except TimeoutException as e:
            skip = True
            answer = "N/A"
            print(f"Timeout: {e}")
            traceback.print_exc()
        except Exception as e:
            skip = True
            answer = "N/A"
            traceback.print_exc()
        
        if skip:
            with open('files/skips.txt', 'a') as f:
                f.write(f"{index}\n")
            return
        
        # save_answer(FILES_DIR, index, answer)
        
        # Print solution
        print("\033[91mSolution:\033[0m")
        print(df.iloc[index]['solution'])
        
        # Check accuracy
        if is_correct(answer, df.iloc[index]['solution']):
            accurate_list[index] = answer

        total_list[index] = answer
        
    finally:
        sys.stdout = sys.__stdout__
        f2.close()


def save_summary(result_dir, accurate_list, total_list):
    """Save summary statistics to file."""
    with open(f"{result_dir}/summary.txt", "w") as f:
        total = len(total_list)
        
        f.write(f"Accurate: {len(accurate_list)}\n")
        f.write(f"Total: {total}\n")
        f.write(f"Accuracy: {len(accurate_list)/total*100:.2f}%\n\n")
        
        # Indices
        f.write(f"Accurate indices: {list(accurate_list.keys())}\n")
        f.write(f"Total indices: {list(total_list.keys())}\n")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run diagram agent with specified model.')
    parser.add_argument(
        '--model',
        type=str,
        default='openai',
        choices=['openai', 'claude', 'gemini', 'qwen', 'qwen32b', 'llava', 'internvl', 'nemotron', 'glm', 'llama'],
        help='Model to use for inference (default: openai)'
    )
    parser.add_argument(
        '--indices',
        type=str,
        default=None,
        help='Comma-separated list of specific indices to process (e.g., "0,5,10,15")'
    )
    args = parser.parse_args()
    
    # Create model-specific directories
    RESULT_DIR = f"{RESULT_DIR_BASE}_{args.model}"
    FILES_DIR = f"{FILES_DIR_BASE}_{args.model}"
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FILES_DIR, exist_ok=True)
    
    print(f"Starting the diagram agent with model: {args.model}")
    print(f"Results directory: {RESULT_DIR}")
    print(f"Files directory: {FILES_DIR}")
    
    # Load API keys
    with open('../private/webapikey.txt', 'r') as f:
        web_api_key = f.read().strip()
    with open('../private/azureapikey.txt', 'r') as f:
        azure_api_key = f.read().strip()
    with open('../private/openrouterapikey.txt', 'r') as f:
        openrouter_api_key = f.read().strip()
    
    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
    os.environ["WEBAGENT_API_KEY"] = web_api_key
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
    
    # Load dataset
    df = pd.read_csv('../datasets/netq/netq.csv')
    print(f"DataFrame loaded of length: {len(df)}")
    
    # Load existing lists
    lists = load_lists(FILES_DIR)
    accurate_list = lists.get('accurate_list', {})
    total_list = lists.get('total_list', {})
    
    # Load accurate_list_old: first check if it exists, otherwise use accurate_list, otherwise empty
    if 'accurate_list_old' in lists:
        accurate_list_old = lists['accurate_list_old']
    elif 'accurate_list' in lists and lists['accurate_list']:
        accurate_list_old = lists['accurate_list']
    else:
        accurate_list_old = {}
    
    # Load skip list
    skip_list = set()
    # skip_file = 'files/skip.txt'
    # if os.path.exists(skip_file):
    #     with open(skip_file, 'r') as f:
    #         skip_list = set(eval(f.read().strip()))
    #     print(f"Loaded {len(skip_list)} indices to skip")
    
    # print("Lists loaded.")
    
    # Initialize agents and models
    api_key = os.getenv("WEBAGENT_API_KEY")
    agent = WebAgent(api_key=api_key)
    print("WebAgent initialized.")
    
    model = Model(args.model)
    print(f"Model '{args.model}' initialized.")
    
    # Parse specific indices if provided
    specific_indices = None
    if args.indices:
        specific_indices = [int(i.strip()) for i in args.indices.split(',')]
        print(f"Will process specific indices: {specific_indices}")
    
    # find the largest index from total_list to resume from there
    # start_index = 0
    start_index = max(total_list.keys()) + 1 if total_list else 0
    if not specific_indices:
        print(f"Resuming from index: {start_index}")
    
    # Process all problems
    for index in range(start_index,len(df)):
        if specific_indices and index not in specific_indices:
            continue
        if index in skip_list:
            print(f"Skipping index {index} (in skip list)")
            continue
        print(f"\n\n=== Processing index {index} ===")
        
        process_problem(index, df, agent, model, accurate_list, accurate_list_old, total_list)
        
        # Save summary after each problem
        try:
            save_summary(RESULT_DIR, accurate_list, total_list)
        except Exception as e:
            pass
        
        # Save all lists
        save_lists(FILES_DIR, {
            'accurate_list': accurate_list,
            'total_list': total_list,
        })
    
    # Print final statistics
    total = len(total_list)
    print(f"\nFinal Accuracy: {len(accurate_list)/total*100:.2f}%")