from engraf.lexer.vector_space import VectorSpace
from engraf.utils.noun_inflector import singularize_noun
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE


def add_to_vocabulary(word, vector_space):
    """Add or update a word in the runtime vocabulary."""
    word = word.lower()
    SEMANTIC_VECTOR_SPACE[word] = vector_space


def get_from_vocabulary(word: str) -> VectorSpace:
    """Safely retrieves a vector from the vocabulary."""
    return SEMANTIC_VECTOR_SPACE.get(word.lower())


def has_word(word: str) -> bool:
    return word.lower() in SEMANTIC_VECTOR_SPACE


def vector_from_word(word: str) -> VectorSpace:
    base_vector = SEMANTIC_VECTOR_SPACE.get(word.lower())
    if base_vector:
        copy = base_vector.copy()
        copy.word = word
        return copy

    # Try singularizing in case it's a plural noun
    singular, is_plural = singularize_noun(word)
    if singular != word:
        base_vector = SEMANTIC_VECTOR_SPACE.get(singular.lower())
        if base_vector:
            v = base_vector.copy()
            v.word = word  # Preserve the original word
            if is_plural:
                v["plural"] = 1.0
                v["singular"] = 0.0
            return v

    # Try base form in case it's a comparative adjective
    from engraf.utils.adjective_inflector import find_root_adjective
    base_adj, form_type, found_adjective = find_root_adjective(word)
    if found_adjective and base_adj != word:
        base_vector = SEMANTIC_VECTOR_SPACE.get(base_adj.lower())
        if base_vector:
            v = base_vector.copy()
            v.word = word
            # Mark as comparative or superlative and boost intensity
            if form_type == 'comp':
                v["comp"] = 1.0  # Mark as comparative form
                multiplier = 1.2  # moderate boost for comparatives
            elif form_type == 'super':
                v["super"] = 1.0  # Mark as superlative form
                multiplier = 1.5  # stronger boost for superlatives
            
            if v["adj"] > 0:
                # Scale semantic features differently for comparative vs superlative
                semantic_features = ["scaleX", "scaleY", "scaleZ", "red", "green", "blue", "texture", "transparency"]
                for key in semantic_features:
                    if v[key] != 0:
                        v[key] = v[key] * multiplier
            return v

    # Still unknown? Raise
    raise ValueError(f"Unknown token: {word}")
