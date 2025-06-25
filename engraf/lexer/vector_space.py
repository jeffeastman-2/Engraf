import numpy as np
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---

VECTOR_DIMENSIONS = [
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

    @property
    def shape(self):
        return self.vector.shape



def vector_from_features(loc=None, scale=None, rot=None, color=None):
    vs = VectorSpace()
    if loc: vs["locX"], vs["locY"], vs["locZ"] = loc
    if scale: vs["scaleX"], vs["scaleY"], vs["scaleZ"] = scale
    if rot: vs["rotX"], vs["rotY"], vs["rotZ"] = rot
    if color: vs["red"], vs["green"], vs["blue"] = color
    return vs

SEMANTIC_VECTOR_SPACE = {
    'cube': vector_from_features(loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'box': vector_from_features(loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'sphere': vector_from_features(loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'arch': vector_from_features(loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'object': vector_from_features(loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'red': vector_from_features(color=[1.0, 0.0, 0.0]),
    'green': vector_from_features(color=[0.0, 1.0, 0.0]),
    'blue': vector_from_features(color=[0.0, 0.0, 1.0]),
    'yellow': vector_from_features(color=[1.0, 1.0, 0.0]),
    'purple': vector_from_features(color=[0.5, 0.0, 0.5]),
    'orange': vector_from_features(color=[1.0, 0.5, 0.0]),
    'black': vector_from_features(color=[0.0, 0.0, 0.0]),
    'white': vector_from_features(color=[1.0, 1.0, 1.0]),
    'gray': vector_from_features(color=[0.5, 0.5, 0.5]),
    'large': vector_from_features(scale=[2.0, 2.0, 2.0]),
    'small': vector_from_features(scale=[-0.5, -0.5, -0.5]),
    'tall': vector_from_features(scale=[0.0, 1.5, 0.0]),
    'wide': vector_from_features(scale=[1.5, 0.0, 0.0]),
    'deep': vector_from_features(scale=[0.0, 0.0, 1.5]),
    'the': vector_from_features()
}
