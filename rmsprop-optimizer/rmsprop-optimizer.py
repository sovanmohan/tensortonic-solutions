import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w_arr = np.asarray(w, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    # Step 1: Update running squared gradient average
    new_s = beta * s_arr + (1.0 - beta) * (g_arr ** 2)
    # Step 2: Update parameters
    new_w = w_arr - (lr / (np.sqrt(new_s) + eps)) * g_arr
    return new_w.tolist(), new_s.tolist()