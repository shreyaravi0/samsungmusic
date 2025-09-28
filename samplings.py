# samplings.py
import numpy as np

def top_p_sampling(probs, top_p=0.9, return_probs=False):
    """
    Nucleus (top-p) sampling: keep smallest set of tokens
    whose cumulative probability >= top_p.
    """
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    cutoff_idx = np.searchsorted(cumulative_probs, top_p)
    top_indices = sorted_indices[:cutoff_idx + 1]
    top_probs = sorted_probs[:cutoff_idx + 1]

    top_probs = top_probs / np.sum(top_probs)  # renormalize

    if return_probs:
        return dict(zip(top_indices, top_probs))
    else:
        return np.random.choice(top_indices, p=top_probs)


def temperature_sampling(probs, temperature=1.0):
    """
    Apply temperature scaling to probabilities before sampling.
    """
    if isinstance(probs, dict):
        indices = list(probs.keys())
        values = np.array(list(probs.values()))
    else:
        indices = np.arange(len(probs))
        values = np.array(probs)

    if temperature <= 0:  # greedy
        return indices[np.argmax(values)]

    scaled = np.log(values + 1e-9) / temperature
    exp_scaled = np.exp(scaled)
    new_probs = exp_scaled / np.sum(exp_scaled)

    return np.random.choice(indices, p=new_probs)
