from engraf.lexer.vector_space import vector_from_features
from engraf.lexer.vector_space import VectorSpace

SEMANTIC_VECTOR_SPACE = {
    # Nouns
    'cube': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'box': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'sphere': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'arch': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'object': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'square': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'rectangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'triangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'circle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'cylinder': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'cone': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'tetrahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'hexahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'octahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'dodecahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'icosahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'pyramid': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]),
    'prism': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0]), 
    # Pronouns
    'it': vector_from_features("pronoun singular"),
    'they': vector_from_features("pronoun plural"),
    'them': vector_from_features("pronoun plural"),
     # Adjectives
    'red': vector_from_features("adj", color=[1.0, 0.0, 0.0]),
    'green': vector_from_features("adj", color=[0.0, 1.0, 0.0]),
    'blue': vector_from_features("adj", color=[0.0, 0.0, 1.0]),
    'yellow': vector_from_features("adj", color=[1.0, 1.0, 0.0]),
    'purple': vector_from_features("adj", color=[0.5, 0.0, 0.5]),
    'orange': vector_from_features("adj", color=[1.0, 0.5, 0.0]),
    'black': vector_from_features("adj", color=[0.0, 0.0, 0.0]),
    'white': vector_from_features("adj", color=[1.0, 1.0, 1.0]),
    'gray': vector_from_features("adj", color=[0.5, 0.5, 0.5]),
    'large': vector_from_features("adj", scale=[2.0, 2.0, 2.0]),
    'small': vector_from_features("adj", scale=[-0.5, -0.5, -0.5]),
    'tall': vector_from_features("adj", scale=[0.0, 1.5, 0.0]),
    'wide': vector_from_features("adj", scale=[1.5, 0.0, 0.0]),
    'deep': vector_from_features("adj", scale=[0.0, 0.0, 1.5]),
    'rough': vector_from_features("adj", texture=2.0),
    'smooth': vector_from_features("adj", texture=0.5),
    'shiny': vector_from_features("adj", texture=0.0),
    'clear': vector_from_features("adj", transparency=2.0),
    'transparent': vector_from_features("adj", transparency=2.0),
    'opaque': vector_from_features("adj", transparency=0.0),
    # Adverbs
    'very': vector_from_features("adv", adverb=1.5),
    'more': vector_from_features("adv", adverb=1.5),
    'much': vector_from_features("adv", adverb=1.5),
    'extremely': vector_from_features("adv", adverb=2.0),
    'slightly': vector_from_features("adv", adverb=0.75),
    # Determiners
    'the': vector_from_features("det def", number=1.0),
    'one': vector_from_features("det def", number=1.0),
    'two': vector_from_features("det def", number=2.0),
    'three': vector_from_features("det def", number=3.0),
    'four': vector_from_features("det def", number=4.0),
    'five': vector_from_features("det def", number=5.0),
    'six': vector_from_features("det def", number=6.0),
    'seven': vector_from_features("det def", number=7.0),
    'eight': vector_from_features("det def", number=8.0),
    'nine': vector_from_features("det def", number=9.0),
    'ten': vector_from_features("det def", number=10.0),
    'a': vector_from_features("det", number=1.0),
    'an': vector_from_features("det", number=1.0),

    # Verbs
    # create
    'create': vector_from_features("verb action create"),
    'draw': vector_from_features("verb action create"),
    'make': vector_from_features("verb action create"),
    'place': vector_from_features("verb action create"),

    # edit
    'copy': vector_from_features("verb action edit"),
    'delete': vector_from_features("verb action edit"),
    'paste': vector_from_features("verb action edit"),

    # organize
    'align': vector_from_features("verb action organize"),
    'group': vector_from_features("verb action organize"),
    'position': vector_from_features("verb action organize"),
    'ungroup': vector_from_features("verb action organize"),

    # select
    'select': vector_from_features("verb action select"),

    # style
    'color': vector_from_features("verb action style"),
    'texture': vector_from_features("verb action style"),

    # transform
    'move': vector_from_features("verb action transform"),
    'rotate': vector_from_features("verb action transform"),
    'scale': vector_from_features("verb action transform"),

    # generic (no third term)
    'redo': vector_from_features("verb action"),
    'undo': vector_from_features("verb action"),

    # Prepositions
    'on': vector_from_features("prep"),
    'over': vector_from_features("prep"),
    'under': vector_from_features("prep"),
    'above': vector_from_features("prep"),
    'below': vector_from_features("prep"),
    'with': vector_from_features("prep"),
    'to': vector_from_features("prep"),
    'from': vector_from_features("prep"),
    'in': vector_from_features("prep"),
    'at': vector_from_features("prep"),
    'by': vector_from_features("prep"),
    'near': vector_from_features("prep"),
    # Conjunctions
    'and': vector_from_features("conj"),
    'or': vector_from_features("conj"),
    # To be verbs
    'is': vector_from_features("tobe"),
    'are': vector_from_features("tobe"),
    'was': vector_from_features("tobe"),
    'were': vector_from_features("tobe"),
    'be': vector_from_features("tobe"),
    'been': vector_from_features("tobe")
}

def add_to_vocabulary(word, vector_space):
    """Add or update a word in the runtime vocabulary."""
    word = word.lower()
    SEMANTIC_VECTOR_SPACE[word] = vector_space

def get_from_vocabulary(word: str) -> VectorSpace:
    """Safely retrieves a vector from the vocabulary."""
    return SEMANTIC_VECTOR_SPACE.get(word.lower())

def has_word(word: str) -> bool:
    return word.lower() in SEMANTIC_VECTOR_SPACE


from engraf.utils.noun_inflector import singularize_noun

def vector_from_word(word: str) -> VectorSpace:
    base_vector = SEMANTIC_VECTOR_SPACE.get(word)
    if base_vector:
        copy = base_vector.copy()
        copy.word = word
        return copy

    # Try singularizing in case it's a plural noun
    singular, is_plural = singularize_noun(word)
    if singular != word:
        base_vector = SEMANTIC_VECTOR_SPACE.get(singular)
        if base_vector:
            v = base_vector.copy()
            if is_plural:
                v["plural"] = 1.0
                v["singular"] = 0.0
            return v

    # Still unknown? Raise
    raise ValueError(f"Unknown token: {word}")
