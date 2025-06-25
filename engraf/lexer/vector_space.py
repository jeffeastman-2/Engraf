import numpy as np
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---
VECTOR_SPACE = {
    'cube': np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    'box': np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    'sphere': np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    'arch': np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    'object': np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    'red': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    'green': np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    'blue': np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    'large': np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]),
    'small': np.array([0.0, 0.0, 0.0, -0.5, -0.5, -0.5]),
    'tall': np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.0]),
    'wide': np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0]),
    'deep': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5]),
    'the': np.zeros(6)
}
