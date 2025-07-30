# vector_space.py
import numpy as np
import math
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---

VECTOR_DIMENSIONS = [
    "verb", "tobe", "action", "prep", "det", "def", "adv", "adj", "noun", "pronoun",
    "number", "vector", "singular", "plural", "conj", "disj", "tobe", "unit",
    "comp",      # comparative forms (rougher, taller, etc.)
    "locX", "locY", "locZ",
    "scaleX", "scaleY", "scaleZ",
    "rotX", "rotY", "rotZ",
    "red", "green", "blue",
    "texture", "transparency",
    "quoted", 
    # High-level verb intent vectors
    "create",    # draw, create, place, make
    "transform", # move, rotate, scale
    "style",     # color, texture
    "organize",  # group, ungroup, align, position
    "edit",      # delete, undo, redo, copy, paste
    "select",    # select
    # Semantic preposition dimensions
    "spatial_vertical",      # above/over (+), below/under (-), on (contact)
    "spatial_proximity",     # near (+), at (specific), in (containment)
    "directional_target",    # to (+), from (-)
    "directional_agency",    # by (+), with (accompaniment)
    "relational_possession", # of (belongs to, part of)
    "relational_comparison"  # than (comparison baseline)
]

VECTOR_LENGTH = len(VECTOR_DIMENSIONS)

class VectorSpace:
    def __init__(self, array=None, word=None, data=None):
        if array is None:
            self.vector = np.zeros(VECTOR_LENGTH)
        else:
            if len(array) != VECTOR_LENGTH:
                raise ValueError(f"Expected vector of length {VECTOR_LENGTH}, got {len(array)}")
            self.vector = np.array(array, dtype=float)
        self.word = word  # NEW
        self.data = data or {}

    # Add optional __str__ override for easier debugging
    def __repr__(self):
        vec_str = ', '.join(f'{k}={self[k]:.2f}' for k in VECTOR_DIMENSIONS)
        return f"VectorSpace(word={self.word!r}, {{ {vec_str} }})"

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

    def __mul__(self, scalar):
        return VectorSpace(self.vector * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def isa(self, category: str) -> bool:
        """Returns True if the category is 'active' in this vector."""
        try:
            idx = VECTOR_DIMENSIONS.index(category.lower())
            return self.vector[idx] > 0.0  # threshold hardcoded here
        except ValueError:
            return False

    def cosine_similarity(self, other):
        # Calculate dot product
        dot = sum(self[k] * other[k] for k in self.data if k in other.data)

        # Calculate norms
        norm_self = math.sqrt(sum(v * v for v in self.data.values()))
        norm_other = math.sqrt(sum(v * v for v in other.data.values()))

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return dot / (norm_self * norm_other)

    @property
    def shape(self):
        return self.vector.shape

    def scalar_projection(self, dim="adv"):
        i = VECTOR_DIMENSIONS.index(dim)
        return self.vector[i]

    def copy(self):
        new_vs = VectorSpace()
        new_vs.word = self.word
        new_vs.vector = self.vector.copy()
        return new_vs


def vector_from_features(pos, adverb=None, loc=None, scale=None, rot=None, color=None, word=None, number=None, texture=None, transparency=None, **semantic_dims):
    vs = VectorSpace(word)
    for tag in pos.split():
        if tag in VECTOR_DIMENSIONS:
            vs[tag] = 1.0
        else:
            raise ValueError(f"Unknown POS tag '{tag}' in vector_from_features")
    if loc: vs["locX"], vs["locY"], vs["locZ"] = loc
    if scale: vs["scaleX"], vs["scaleY"], vs["scaleZ"] = scale
    if rot: vs["rotX"], vs["rotY"], vs["rotZ"] = rot
    if color: vs["red"], vs["green"], vs["blue"] = color
    if adverb is not None: vs["adv"] = adverb
    if texture is not None: vs["texture"] = texture
    if transparency is not None: vs["transparency"] = transparency
    if number is not None: vs["number"] = number
    
    # Handle semantic dimensions for prepositions
    for dim_name, dim_value in semantic_dims.items():
        if dim_name in VECTOR_DIMENSIONS:
            vs[dim_name] = dim_value
        else:
            raise ValueError(f"Unknown semantic dimension '{dim_name}' in vector_from_features")
    
    return vs

