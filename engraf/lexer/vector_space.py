# vector_space.py
import numpy as np
import math
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
# --- Updated semantic vector space (6D: RGB + X/Y/Z size) ---



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
        
        # Static fields to replace dynamic attributes used in Layer 2/3
        self._grounded_phrase = None  # Set by Layer 2 grounding
        self._original_np = None      # Set when creating NP tokens
        self._original_pp = None      # Set when creating PP tokens
        self._original_sp = None    # Set when creating Sentence tokens
        self.scene_object = None      # Expected by Layer 3 spatial validation
        self._attachment_info = {}    # Set during PP attachment processing
        self._reference_object = None # Expected by Layer 3 spatial grounding results
        self._preposition = None      # Expected by Layer 3 spatial grounding results

    # Add optional __str__ override for easier debugging
    def __repr__(self):
        # Only show non-zero dimensions for cleaner output
        non_zero_dims = self.non_zero_dims()
        return f"VS(word={self.word!r}, {{ {non_zero_dims} }})"

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

    def __contains__(self, key):
        """Support for 'key in vector' syntax."""
        try:
            if isinstance(key, int):
                return 0 <= key < len(self.vector)
            return key in VECTOR_DIMENSIONS
        except:
            return False

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
    
    def non_zero_dims(self) -> str:
        non_zero_dims = [f'{k}={self[k]:.2f}' for k in VECTOR_DIMENSIONS if self[k] != 0.0]
        vec_str = ', '.join(non_zero_dims)
        return vec_str

    def isa(self, category: str) -> bool:
        """Returns True if the category is 'active' in this vector."""
        try:
            # Try exact match first
            idx = VECTOR_DIMENSIONS.index(category)
            return self.vector[idx] > 0.0  # threshold hardcoded here
        except ValueError:
            try:
                # Try lowercase match for backwards compatibility
                idx = VECTOR_DIMENSIONS.index(category.lower())
                return self.vector[idx] > 0.0
            except ValueError:
                return False

    def cosine_similarity(self, other):
        """Calculate cosine similarity between this vector and another vector."""
        # Use numpy arrays for proper cosine similarity calculation
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector) 
        norm_other = np.linalg.norm(other.vector)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
            
        return dot / (norm_self * norm_other)
    
    def semantic_similarity(self, other):
        """Calculate semantic similarity focusing only on pure semantic attributes.
        
        This excludes POS dimensions (noun, adj, verb, etc.) and focuses only on 
        semantic content dimensions like colors, spatial properties, and intent vectors.
        
        This is asymmetric - it only tests features that are present in this vector (the query).
        If the query is just "sphere", it matches any sphere regardless of additional attributes.
        If the query is "blue sphere", it only matches blue spheres.
        """
        from engraf.An_N_Space_Model.vector_dimensions import SEMANTIC_DIMENSIONS
        
        # Only consider semantic dimensions that are specified (non-zero) in the query vector
        active_dims = []
        query_values = []
        target_values = []
        
        for dim in SEMANTIC_DIMENSIONS:
            if dim in VECTOR_DIMENSIONS and self[dim] != 0.0:
                active_dims.append(dim)
                query_values.append(self[dim])
                target_values.append(other[dim])
        
        # If no semantic features are specified in query, return high similarity (matches anything)
        if not active_dims:
            return 1.0
        
        # Calculate cosine similarity only on the semantic dimensions specified in the query
        query_array = np.array(query_values)
        target_array = np.array(target_values)
        
        dot = np.dot(query_array, target_array)
        norm_query = np.linalg.norm(query_array)
        norm_target = np.linalg.norm(target_array)
        
        if norm_query == 0:
            return 1.0  # No semantic constraints
        if norm_target == 0:
            return 0.0  # Target has no matching features
            
        return dot / (norm_query * norm_target)

    def semantic_vector(self):
        """Return a new VectorSpace containing only semantic dimensions (masked).
        
        This is similar to Q/K transformation in transformer attention - 
        creates a masked version suitable for semantic comparison.
        """
        from engraf.An_N_Space_Model.vector_dimensions import get_semantic_mask
        
        semantic_mask = get_semantic_mask()
        masked_array = self.vector * np.array(semantic_mask, dtype=float)
        
        result = VectorSpace(masked_array, word=self.word, data=self.data.copy() if self.data else None)
        return result
    
    def pos_vector(self):
        """Return a new VectorSpace containing only POS dimensions (masked).
        
        Useful for grammatical analysis separate from semantic content.
        """
        from engraf.An_N_Space_Model.vector_dimensions import get_pos_mask
        
        pos_mask = get_pos_mask()
        masked_array = self.vector * np.array(pos_mask, dtype=float)
        
        result = VectorSpace(masked_array, word=self.word, data=self.data.copy() if self.data else None)
        return result

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

