import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity.item()