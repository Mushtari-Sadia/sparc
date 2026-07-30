"""
Self-consistency module for majority voting with cosine similarity.
"""
from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two text strings using TF-IDF vectors.
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        Cosine similarity score between 0 and 1
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        # Fallback to exact match if vectorization fails
        return 1.0 if text1.strip() == text2.strip() else 0.0


def find_similar_groups(outputs: List[str], similarity_threshold: float = 0.85) -> List[List[int]]:
    """
    Group outputs by similarity using cosine similarity.
    
    Args:
        outputs: List of output strings
        similarity_threshold: Minimum similarity to consider outputs as similar (default 0.85)
    
    Returns:
        List of groups, where each group is a list of indices that are similar to each other
    """
    n = len(outputs)
    groups = []
    assigned = [False] * n
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start a new group with this output
        current_group = [i]
        assigned[i] = True
        
        # Find all similar outputs
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            
            similarity = calculate_cosine_similarity(outputs[i], outputs[j])
            if similarity >= similarity_threshold:
                current_group.append(j)
                assigned[j] = True
        
        groups.append(current_group)
    
    return groups


def majority_vote_with_similarity(outputs: List[str], similarity_threshold: float = 0.85) -> Tuple[str, int]:
    """
    Perform majority voting on outputs using cosine similarity to group similar responses.
    
    Args:
        outputs: List of output strings from multiple inferences
        similarity_threshold: Minimum similarity to consider outputs as similar (default 0.85)
    
    Returns:
        Tuple of (majority_output, vote_count)
    """
    if not outputs:
        return "", 0
    
    if len(outputs) == 1:
        return outputs[0], 1
    
    # Group similar outputs
    groups = find_similar_groups(outputs, similarity_threshold)
    
    # Find the largest group (majority)
    largest_group = max(groups, key=len)
    majority_count = len(largest_group)
    
    # Return the first output from the largest group as the representative
    majority_output = outputs[largest_group[0]]
    
    return majority_output, majority_count


def self_consistency_inference(
    inference_function,
    num_samples: int = 1,
    similarity_threshold: float = 0.85,
    **kwargs
) -> Tuple[str, int, List[str]]:
    """
    Perform self-consistency inference by running multiple inferences and taking majority vote.
    
    Args:
        inference_function: Function that performs a single inference (should return a string)
        num_samples: Number of times to run inference (default 5)
        similarity_threshold: Minimum similarity to consider outputs as similar (default 0.85)
        **kwargs: Additional keyword arguments to pass to inference_function
    
    Returns:
        Tuple of (majority_output, vote_count, all_outputs)
    """
    outputs = []
    
    print(f"Running self-consistency inference with {num_samples} samples...")
    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...")
        output = inference_function(**kwargs)
        outputs.append(output)
    
    majority_output, vote_count = majority_vote_with_similarity(outputs, similarity_threshold)
    
    print(f"Self-consistency voting complete: {vote_count}/{num_samples} samples agreed")
    
    return majority_output, vote_count, outputs
