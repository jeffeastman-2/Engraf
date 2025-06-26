import numpy as np
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---

VECTOR_DIMENSIONS = [
    "verb", "tobe", "prep", "det", "def", "adv", "adj", "noun", 
    "number", "vector",
    "locX", "locY", "locZ",
    "scaleX", "scaleY", "scaleZ",
    "rotX", "rotY", "rotZ",
    "red", "green", "blue"
]

VECTOR_LENGTH = len(VECTOR_DIMENSIONS)

class VectorSpace:
    def __init__(self, array=None):
        if array is None:
            self.vector = np.zeros(VECTOR_LENGTH)
        else:
            if len(array) != VECTOR_LENGTH:
                raise ValueError(f"Expected vector of length {VECTOR_LENGTH}, got {len(array)}")
            self.vector = np.array(array, dtype=float)

    def to_array(self):
        return self.vector

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.vector[key]
        idx = VECTOR_DIMENSIONS.index(key)
        return self.vector[idx]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.vector[key] = value
        else:
            idx = VECTOR_DIMENSIONS.index(key)
            self.vector[idx] = value

    def __repr__(self):
        return f"VectorSpace({{ {', '.join(f'{k}={self[k]:.2f}' for k in VECTOR_DIMENSIONS)} }})"

    def __iadd__(self, other):
        if isinstance(other, VectorSpace):
            self.vector += other.vector
        else:
            raise TypeError("Can only add another VectorSpace instance")
        return self

    def __add__(self, other):
        if isinstance(other, VectorSpace):
            return VectorSpace(self.vector + other.vector)
        else:
            raise TypeError("Can only add another VectorSpace instance")
            
    def isa(self, category: str) -> bool:
        """Returns True if the category is 'active' in this vector."""
        try:
            idx = VECTOR_DIMENSIONS.index(category.lower())
            return self.vector[idx] > 0.5  # threshold hardcoded here
        except ValueError:
            return False

    @property
    def shape(self):
        return self.vector.shape



def vector_from_features(pos, loc=None, scale=None, rot=None, color=None):
    vs = VectorSpace()
    for tag in pos.split():
        if tag in VECTOR_DIMENSIONS:
            vs[tag] = 1.0
        else:
            raise ValueError(f"Unknown POS tag '{tag}' in vector_from_features")
    if loc: vs["locX"], vs["locY"], vs["locZ"] = loc
    if scale: vs["scaleX"], vs["scaleY"], vs["scaleZ"] = scale
    if rot: vs["rotX"], vs["rotY"], vs["rotZ"] = rot
    if color: vs["red"], vs["green"], vs["blue"] = color
    return vs
