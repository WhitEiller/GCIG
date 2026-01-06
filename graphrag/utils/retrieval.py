import numpy as np


def get_cos_sim_matrix(x: np.array, y: np.array) -> np.array:
    # Cosine_similarity(x, y) = (xy) / (||x|| ||y||)
    x_norms = np.linalg.norm(x, axis=1, keepdims=True)
    y_norms = np.linalg.norm(y, axis=1, keepdims=True)
    cos_sim_matrix =  (x @ y.T) / (x_norms @ y_norms.T)
    return cos_sim_matrix

def retrieve(query: np.array, target: np.array, **kwds) -> tuple[list[list[int]], list[list[float]]]:
    cos_sim_matrix = get_cos_sim_matrix(query, target)
    
    indices = []
    similarities = []
    top_k = kwds.get("top_k", None)
    threshold = kwds.get("threshold", None)
    
    for cos_sim in cos_sim_matrix:
        if top_k is not None:
            top_indices = np.argsort(cos_sim)[-top_k:][::-1]
            top_similarities = cos_sim[top_indices]
        else:
            top_indices = np.arange(len(cos_sim))
            top_similarities = cos_sim
        
        if threshold is not None:
            mask = top_similarities >= threshold
            top_indices = top_indices[mask]
            top_similarities = top_similarities[mask]
        
        indices.append(top_indices.tolist())
        similarities.append(top_similarities.tolist())
    return indices, similarities