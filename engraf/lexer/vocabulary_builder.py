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
    base_adj, form_type = base_adjective_from_comparative(word)
    if form_type != 'base' and base_adj != word:
        base_vector = SEMANTIC_VECTOR_SPACE.get(base_adj.lower())
        if base_vector:
            v = base_vector.copy()
            v.word = word
            # Mark as comparative or superlative and boost intensity
            if form_type == 'comparative':
                v["comp"] = 1.0  # Mark as comparative form
                multiplier = 1.2  # moderate boost for comparatives
            elif form_type == 'superlative':
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


def base_adjective_from_comparative(word: str) -> tuple[str, str]:
    """Convert comparative adjective to base form. Returns (base_form, form_type).
    
    form_type can be:
    - 'base': not a comparative/superlative form
    - 'comparative': -er form (bigger, taller, etc.)
    - 'superlative': -est form (biggest, tallest, etc.)
    """
    word = word.lower()
    
    # Handle irregular comparative/superlative forms first
    irregular_forms = {
        "better": ("good", "comparative"),
        "best": ("good", "superlative"),
        "worse": ("bad", "comparative"),
        "worst": ("bad", "superlative"),
        "more": ("much", "comparative"),
        "most": ("much", "superlative"),
        "further": ("far", "comparative"),
        "furthest": ("far", "superlative"),
        "farther": ("far", "comparative"),
        "farthest": ("far", "superlative")
    }
    
    if word in irregular_forms:
        # For irregular forms, return as base form since we don't have the base adjective in vocabulary
        return word, 'base'
    
    # Handle -er endings (rougher -> rough, taller -> tall)
    if word.endswith('er'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('gger'):  # bigger -> big
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('tter'):  # fatter -> fat
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('nner'):  # thinner -> thin
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('dder'):  # redder -> red
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('er'):
            base = word[:-2]
            return base, 'comparative'
    
    # Handle -est endings (roughest -> rough, tallest -> tall)
    if word.endswith('est'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('ggest'):  # biggest -> big
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('ttest'):  # fattest -> fat
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('nnest'):  # thinnest -> thin
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('ddest'):  # reddest -> red
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('est'):
            base = word[:-3]
            return base, 'superlative'
    
    # Not a comparative/superlative
    return word, 'base'
