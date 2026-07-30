#!/usr/bin/env python3
"""
Script to run all baselines on indices from total_list.json
Outputs results to summary_baselines.txt
"""

import sys
import os
import argparse
import json
from pathlib import Path
import threading

import pandas as pd
from PIL import Image

from utils import load_lists, save_lists
from model import Model
# from baselines import baseline_1, baseline_2, baseline_3, baseline_4_cot, baseline_5_cot, baseline_6_cot
from baselines import baseline_4_cot, baseline_5_cot, baseline_6_cot, baseline_7_sympy
from evaluation import is_correct


# Configuration
IMAGE_PATH = "annotations/images_saved5"


def load_total_list_indices(files_dir):
    """Load indices from total_list.json"""
    total_list_path = f"{files_dir}/total_list.json"
    if not os.path.exists(total_list_path):
        return []
    
    with open(total_list_path, 'r') as f:
        total_list = json.load(f)
    
    # Convert string keys to integers
    indices = [int(k) for k in total_list.keys()]
    return sorted(indices)


def run_baselines_for_index(index, df, model, baseline_lists, is_local_model):
    """Run all 4 baselines for a given index"""
    
    # Get image path (will be loaded separately in each thread for thread-safety)
    image_path = f"{IMAGE_PATH}/{index}.png"
    if not os.path.exists(image_path):
        return None
    
    # Get problem data
    schema = df.iloc[index]['schema']
    problem = df.iloc[index]['problem']
    solution = df.iloc[index]['solution']
    
    # Run all baselines
    results = {}
    
    # try:
    #     results['baseline_1'] = baseline_1(model, schema, problem)
    # except Exception as e:
    #     results['baseline_1'] = "ERROR"
    
    # try:
    #     results['baseline_2'] = baseline_2(model, schema, problem, pil_image)
    # except Exception as e:
    #     results['baseline_2'] = "ERROR"
    
    # try:
    #     results['baseline_3'] = baseline_3(model, problem, pil_image)
    # except Exception as e:
    #     results['baseline_3'] = "ERROR"
    
    # Helper function to run a baseline in a thread
    def run_baseline(baseline_num, baseline_func, *args):
        baseline_name = f'baseline_{baseline_num}'
        list_name = f'baseline_{baseline_num}_list'
        
        # Check if already exists and not ERROR
        if index in baseline_lists[list_name] and baseline_lists[list_name][index] != "ERROR":
            results[baseline_name] = baseline_lists[list_name][index]
        else:
            try:
                results[baseline_name] = baseline_func(*args)
            except Exception as e:
                print(f"ERROR in {baseline_name} for index {index}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results[baseline_name] = "ERROR"
    
    if is_local_model:
        # Run sequentially for local models (thread-safety for PyTorch models)
        print(f"Running baselines sequentially for local model at index {index}")
        run_baseline(4, baseline_4_cot, model, problem, schema)
        run_baseline(5, baseline_5_cot, model, problem, schema, Image.open(image_path))
        run_baseline(6, baseline_6_cot, model, problem, Image.open(image_path))
        run_baseline(7, baseline_7_sympy, model, problem, schema, Image.open(image_path))
    else:
        # Run baselines 4, 5, 6, and 7 in parallel for API-based models
        # Load images separately in each thread for thread-safety (PIL Images are not thread-safe)
        threads = []
        threads.append(threading.Thread(target=run_baseline, args=(4, baseline_4_cot, model, problem, schema)))
        threads.append(threading.Thread(target=run_baseline, args=(5, baseline_5_cot, model, problem, schema, Image.open(image_path))))
        threads.append(threading.Thread(target=run_baseline, args=(6, baseline_6_cot, model, problem, Image.open(image_path))))
        threads.append(threading.Thread(target=run_baseline, args=(7, baseline_7_sympy, model, problem, schema, Image.open(image_path))))

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    
    # Check correctness
    for baseline_name, answer in list(results.items()):
        if answer != "ERROR":
            results[f'{baseline_name}_correct'] = is_correct(answer, solution)
        else:
            results[f'{baseline_name}_correct'] = False
    
    return results


def save_summary(result_dir, baseline_results, total_indices):
    """Save summary statistics to file."""
    summary_path = f"{result_dir}/summary_baselines.txt"
    
    with open(summary_path, "w") as f:
        total = len(total_indices)
        
        f.write("=" * 80 + "\n")
        f.write("BASELINE EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total indices evaluated: {total}\n")
        f.write(f"Indices: {total_indices}\n\n")
        
        # Calculate accuracies for each baseline
        # baseline_names = ['baseline_1', 'baseline_2', 'baseline_3', 'baseline_4']
        # baseline_names = ['baseline_1', 'baseline_2', 'baseline_3', 'baseline_4', 'baseline_5', 'baseline_6']
        baseline_names = ['baseline_4', 'baseline_5', 'baseline_6', 'baseline_7']

        for baseline_name in baseline_names:
            correct_count = sum(
                1 for idx in total_indices
                if idx in baseline_results
                and baseline_results[idx].get(f'{baseline_name}_correct', False)
            )
            accuracy = (correct_count / total * 100) if total > 0 else 0

            f.write(f"{baseline_name.upper()}:\n")
            f.write(f"  Correct: {correct_count}/{total}\n")
            f.write(f"  Accuracy: {accuracy:.2f}%\n")

            # List correct indices
            correct_indices = [
                idx for idx in total_indices
                if idx in baseline_results
                and baseline_results[idx].get(f'{baseline_name}_correct', False)
            ]
            f.write(f"  Correct indices: {correct_indices}\n\n")

        # Comparison table
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Index':<8} {'B4':<6} {'B5':<6} {'B6':<6} {'B7':<6}\n")
        f.write("-" * 40 + "\n")

        for idx in total_indices:
            if idx in baseline_results:
                b4 = "✓" if baseline_results[idx].get('baseline_4_correct', False) else "✗"
                b5 = "✓" if baseline_results[idx].get('baseline_5_correct', False) else "✗"
                b6 = "✓" if baseline_results[idx].get('baseline_6_correct', False) else "✗"
                b7 = "✓" if baseline_results[idx].get('baseline_7_correct', False) else "✗"
                f.write(f"{idx:<8} {b4:<6} {b5:<6} {b6:<6} {b7:<6}\n")

        f.write("\n" + "=" * 80 + "\n")

        # Summary statistics
        f.write("\nSUMMARY:\n")
        for baseline_name in baseline_names:
            correct_count = sum(
                1 for idx in total_indices
                if idx in baseline_results
                and baseline_results[idx].get(f'{baseline_name}_correct', False)
            )
            accuracy = (correct_count / total * 100) if total > 0 else 0
            f.write(f"  {baseline_name}: {accuracy:.2f}%\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run all baselines on indices from total_list.json'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai',
        choices=['openai', 'qwen', 'qwen32b', 'claude', 'gemini', 'llava', 'internvl', 'nemotron', 'glm'],
        help='Model to use for inference (default: openai)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='files/21_dec',
        help='Base directory containing model-specific folders (default: files/7_dec)'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default='results_21_dec',
        help='Base directory for results (default: results_7_dec)'
    )
    args = parser.parse_args()
    
    # Create model-specific directories
    FILES_DIR = f"{args.base_dir}_{args.model}"
    RESULT_DIR = f"{args.result_dir}_{args.model}"
    
    # Check if directories exist
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Load API keys
    try:
        with open('private/azureapikey.txt', 'r') as f:
            azure_api_key = f.read().strip()
        with open('private/openrouterapikey.txt', 'r') as f:
            openrouter_api_key = f.read().strip()
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
    except FileNotFoundError:
        pass
    
    # Load total_list indices
    total_indices = load_total_list_indices(FILES_DIR)
    
    if not total_indices:
        print("No indices found in total_list.json. Exiting.")
        sys.exit(1)
    
    # Load skip list
    skip_list = set()
    skip_file = 'files/skip.txt'
    if os.path.exists(skip_file):
        with open(skip_file, 'r') as f:
            skip_list = set(eval(f.read().strip()))
        print(f"Loaded {len(skip_list)} indices to skip")
    
    # Filter out skipped indices
    total_indices = [idx for idx in total_indices if idx not in skip_list]
    print(f"Processing {len(total_indices)} indices after filtering skips")
    
    # Load dataset
    df = pd.read_csv('annotations/df_1359.csv')
    
    # Initialize model
    model = Model(args.model)
    
    # Detect if model is local or API-based
    # Local models (PyTorch) are not thread-safe for concurrent inference
    is_local_model = args.model in ['llava','qwen']
    if is_local_model:
        print(f"Note: {args.model} is a local model. Baselines will run sequentially for thread-safety.")
    else:
        print(f"Note: {args.model} is an API-based model. Baselines will run in parallel.")
    
    # Load existing baseline lists if they exist
    lists = load_lists(FILES_DIR)
    baseline_lists = {
        # 'baseline_1_list': lists.get('baseline_1_list', {}),
        # 'baseline_1_accurate_list': lists.get('baseline_1_accurate_list', {}),
        # 'baseline_2_list': lists.get('baseline_2_list', {}),
        # 'baseline_2_accurate_list': lists.get('baseline_2_accurate_list', {}),
        # 'baseline_3_list': lists.get('baseline_3_list', {}),
        # 'baseline_3_accurate_list': lists.get('baseline_3_accurate_list', {}),
        'baseline_4_list': lists.get('baseline_4_list', {}),
        'baseline_4_accurate_list': lists.get('baseline_4_accurate_list', {}),
        'baseline_5_list': lists.get('baseline_5_list', {}),
        'baseline_5_accurate_list': lists.get('baseline_5_accurate_list', {}),
        'baseline_6_list': lists.get('baseline_6_list', {}),
        'baseline_6_accurate_list': lists.get('baseline_6_accurate_list', {}),
        'baseline_7_list': lists.get('baseline_7_list', {}),
        'baseline_7_accurate_list': lists.get('baseline_7_accurate_list', {})
    }
    
    # Store all baseline results
    baseline_results = {}
    
    # Process each index
    for i, index in enumerate(total_indices, 1):
        results = run_baselines_for_index(index, df, model, baseline_lists, is_local_model)
        
        if results:
            baseline_results[index] = results
            
            # Update baseline lists
            for baseline_num in range(4, 8):
                baseline_name = f'baseline_{baseline_num}'
                answer = results.get(baseline_name)
                is_correct_flag = results.get(f'{baseline_name}_correct', False)
                
                # Update answer list
                baseline_lists[f'{baseline_name}_list'][index] = answer
                
                # Update accurate list (always dict format now)
                if is_correct_flag:
                    baseline_lists[f'{baseline_name}_accurate_list'][index] = answer
            
            # Save lists periodically
            save_lists(FILES_DIR, baseline_lists)
        
        # Save summary after each problem
        save_summary(RESULT_DIR, baseline_results, total_indices[:i])
    
    # Final save
    save_lists(FILES_DIR, baseline_lists)
    save_summary(RESULT_DIR, baseline_results, total_indices)


if __name__ == "__main__":
    main()
