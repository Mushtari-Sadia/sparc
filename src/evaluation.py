# Import additional libraries for data analysis
import pandas as pd
from collections import Counter
import json
from utils import load_config, load_select_dataset
from IPython.display import display

from sentence_transformers import SentenceTransformer
import numpy as np

# load once, reuse
_sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def cosine_sim(u: str, v: str) -> float:
    a, b = _sbert.encode([u, v], normalize_embeddings=True)
    return float(np.dot(a, b))  # cosine since vectors are L2-normalized

def is_semantically_correct(gt: str, mo: str, threshold: float = 0.85) -> bool:
    return cosine_sim(gt, mo) >= threshold

def is_correct(gt: str, mo: str) -> bool:
    if str(gt).strip().upper() == str(mo).strip().upper():
        return True
    try:
        return abs(abs(float(gt)) - abs(float(mo))) < 0.5
    except (ValueError, TypeError):
        pass
    return is_semantically_correct(gt, mo)


import re

def extract_final_answer(text):
    """
    Extracts the substring inside [FINAL_ANSWER_START type="..."] and [FINAL_ANSWER_END].

    Args:
        text (str): Input string containing the pattern.

    Returns:
        str or None: Extracted answer or None if pattern not found.
    """
    pattern = r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>'
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def evaluate(model_name=None, method="baseline",output_path=None):
    """
    Evaluate model results.
    
    Args:
        filename (str): Name of the results file
    """
    # Load configuration
    config = load_config()
    if model_name is None:
        model_name = config['model']
    if output_path is None:
        output_path = f"{config['results_dir']}/{config['dataset_name']}_{model_name}_{method}.jsonl"

    # dset = load_select_dataset()


    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                results.append(json.loads(line))
                

    
        
    accuracy = 0
    total = 0
    
    for i, item in enumerate(results):
        
        idx = item['index']
        # question = dset[idx]['problem']
        # image = dset[idx]['image']
        gt = str(item['solution']).strip()
        mo_full = str(item['model_output']).strip()
        
        mo = extract_final_answer(mo_full)

        # if "thinking" in item:
        #     thinking = item['thinking']
        
        
        is_correct_bool = is_correct(gt, mo)
        accuracy+=1 if is_correct_bool else 0
        total+=1
        status = "✓ CORRECT" if is_correct_bool else "✗ INCORRECT"
        print(f"GT: {gt} | MO: {mo}" + f" => {status}")


        # print(f"\nDataset Index: {idx}:")
        # print(f"Question: {question}")

        # display(image)
        # print(f"{status} - GT: '{gt}' | MO: '{mo}'")
        # print(f"Model Output with reasoning: {mo_full}")
        # if "thinking" in item:
        #     print(f"Model Thinking:\n{thinking}")

    if total > 0:
        correct = accuracy
        accuracy = accuracy / total
        
    
    
    print("ACCURACY RESULTS")
    print("=" * 50)
    print(f"Total valid comparisons: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 50)

