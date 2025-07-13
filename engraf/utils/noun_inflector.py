import re

# Example plural-to-singular dictionary for irregulars
IRREGULAR_PLURALS = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "geese": "goose",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
}

def singularize_noun(word):
    """Convert plural noun to its singular form."""
    word_lower = word.lower()
    if word_lower in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word_lower], True
    elif re.match(r".+ves$", word_lower):
        return re.sub(r"ves$", "f", word_lower), True
    elif re.match(r".+ies$", word_lower):
        return re.sub(r"ies$", "y", word_lower), True
    elif re.search(r"([b-df-hj-np-tv-z]es)$", word_lower) and not word_lower.endswith("ses") and not word_lower.endswith("ies"):
        # Handles 'cubes' -> 'cube', 'boxes' -> 'box', 'buses' -> 'bus', etc.
        return word_lower[:-1], True
    elif re.match(r".+es$", word_lower) and not word_lower.endswith("ies"):
        # Handles regular plurals like 'cubes' -> 'cube'
        return re.sub(r"es$", "", word_lower), True
    elif re.match(r".+s$", word_lower) and not word_lower.endswith("ss"):
        return re.sub(r"s$", "", word_lower), True
    else:
        return word, False

def is_plural(word):
    singular, flag = singularize_noun(word)
    return flag

# Integration in Token creation:
from engraf.lexer.vector_space import vector_from_features, VectorSpace

def analyze_word(word):
    from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
    features = []
    if word.lower() in SEMANTIC_VECTOR_SPACE:
        # If word is in semantic space, return it directly
        return SEMANTIC_VECTOR_SPACE[word.lower()]
    elif is_plural(word):
        features = "noun plural"
    else:
        features = "noun singular"
    return vector_from_features(features)

# Example use:
# vector = analyze_word("boxes")
