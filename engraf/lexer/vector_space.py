# vector_space.py
import numpy as np
import math
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---

VECTOR_DIMENSIONS = [
    "verb", "tobe", "prep", "det", "def", "adv", "adj", "noun", "pronoun",
    "number", "vector", "singular", "plural", "conj", "tobe",
    "locX", "locY", "locZ",
    "scaleX", "scaleY", "scaleZ",
    "rotX", "rotY", "rotZ",
    "red", "green", "blue",
    "texture", "transparency",
    "quoted"
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
            return self.vector[idx] > 0.5  # threshold hardcoded here
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


def vector_from_features(pos, adverb=None, loc=None, scale=None, rot=None, color=None, word=None, number=None, texture=None, transparency=None):
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
    return vs

def is_determiner(tok): return tok is not None and tok.isa("det")
def is_pronoun(tok): return tok is not None and tok.isa("pronoun")
def is_adjective(tok): return tok is not None and tok.isa("adj")
def is_adverb(tok): return tok is not None and tok.isa("adv")
def is_noun(tok): return tok is not None and tok.isa("noun")
def is_verb(tok): return tok is not None and tok.isa("verb")
def is_tobe(tok): return tok is not None and tok.isa("tobe")
def is_preposition(tok): return tok is not None and tok.isa("prep")
def is_number(tok): return tok is not None and tok["number"] > 0.0
def is_none(tok): return tok is None
def is_vector(tok): return tok is not None and tok.isa("vector")
def is_quoted(tok): return tok is not None and tok.isa("quoted")
def is_adjective_or_adverb(tok): return tok is not None and (tok.isa("adj") or tok.isa("adv"))
def any_of(*predicates):
    def combined_predicate(tok):
        return any(pred(tok) for pred in predicates)
    return combined_predicate
def is_conjunction(tok): return tok is not None and tok.isa("conj")

