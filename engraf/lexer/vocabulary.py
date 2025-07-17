from engraf.lexer.vector_space import vector_from_features
from engraf.lexer.vector_space import VectorSpace

SEMANTIC_VECTOR_SPACE = {
    # Nouns
    'cube': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'box': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'sphere': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'arch': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'table': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'object': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'square': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'rectangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'triangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'circle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'cylinder': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'cone': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'tetrahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'hexahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'octahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'dodecahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'icosahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'pyramid': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'prism': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'table': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    
    # Units
    'degree': vector_from_features("noun unit", number=1.0),  # angular unit
    'unit': vector_from_features("noun unit", number=1.0),   # generic unit
    'pixel': vector_from_features("noun unit", number=1.0),  # screen unit
    'meter': vector_from_features("noun unit", number=1.0),  # distance unit
    'inch': vector_from_features("noun unit", number=1.0),   # distance unit
    'foot': vector_from_features("noun unit", number=1.0),   # distance unit
    'yard': vector_from_features("noun unit", number=1.0),   # distance unit
 
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
    'brown': vector_from_features("adj", color=[0.6, 0.3, 0.1]),
    'large': vector_from_features("adj", scale=[2.0, 2.0, 2.0]),
    'big': vector_from_features("adj", scale=[2.0, 2.0, 2.0]),
    'huge': vector_from_features("adj", scale=[3.0, 3.0, 3.0]),
    'small': vector_from_features("adj", scale=[-0.5, -0.5, -0.5]),
    'tiny': vector_from_features("adj", scale=[-0.7, -0.7, -0.7]),
    'tall': vector_from_features("adj", scale=[0.0, 1.5, 0.0]),
    'short': vector_from_features("adj", scale=[0.0, -0.5, 0.0]),
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
    'bright': vector_from_features("adv", adverb=1.5),
    'much': vector_from_features("adv", adverb=1.5),
    'a little bit': vector_from_features("adv", adverb=1.15),
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
    'build': vector_from_features("verb action create"),
    'place': vector_from_features("verb action create"),

    # edit
    'copy': vector_from_features("verb action edit"),
    'delete': vector_from_features("verb action edit"),
    'remove': vector_from_features("verb action edit"),
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
    'xrotate': vector_from_features("verb action transform"),  # rotate around x-axis
    'yrotate': vector_from_features("verb action transform"),  # rotate around y-axis
    'zrotate': vector_from_features("verb action transform"),  # rotate around z-axis
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
    'of': vector_from_features("prep"),
    'than': vector_from_features("prep"),
    
    # Additional verbs for time travel
    'go': vector_from_features("verb action"),
    'back': vector_from_features("adv"),
    'forward': vector_from_features("adv"),
    # Conjunctions
    'and': vector_from_features("conj"),
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
            if is_plural:
                v["plural"] = 1.0
                v["singular"] = 0.0
            return v

    # Try base form in case it's a comparative adjective
    base_adj, is_comparative = base_adjective_from_comparative(word)
    if is_comparative and base_adj != word:
        base_vector = SEMANTIC_VECTOR_SPACE.get(base_adj.lower())
        if base_vector:
            v = base_vector.copy()
            v.word = word
            # Mark as comparative and boost intensity
            v["comp"] = 1.0  # Mark as comparative form
            if v["adj"] > 0:
                # Scale semantic features by 1.2 for comparative effect
                semantic_features = ["scaleX", "scaleY", "scaleZ", "red", "green", "blue", "texture", "transparency"]
                for key in semantic_features:
                    if v[key] != 0:
                        v[key] = v[key] * 1.2
            return v

    # Still unknown? Raise
    raise ValueError(f"Unknown token: {word}")

from engraf.utils.noun_inflector import singularize_noun

def base_adjective_from_comparative(word: str) -> tuple[str, bool]:
    """Convert comparative adjective to base form. Returns (base_form, is_comparative)."""
    word = word.lower()
    
    # Handle -er endings (rougher -> rough, taller -> tall)
    if word.endswith('er'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('gger'):  # bigger -> big
            base = word[:-3]
            return base, True
        elif word.endswith('tter'):  # fatter -> fat
            base = word[:-3]
            return base, True
        elif word.endswith('nner'):  # thinner -> thin
            base = word[:-3]
            return base, True
        elif word.endswith('dder'):  # redder -> red
            base = word[:-3]
            return base, True
        elif word.endswith('er'):
            base = word[:-2]
            return base, True
    
    # Handle -est endings (roughest -> rough, tallest -> tall)
    if word.endswith('est'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('ggest'):  # biggest -> big
            base = word[:-4]
            return base, True
        elif word.endswith('ttest'):  # fattest -> fat
            base = word[:-4]
            return base, True
        elif word.endswith('nnest'):  # thinnest -> thin
            base = word[:-4]
            return base, True
        elif word.endswith('ddest'):  # reddest -> red
            base = word[:-4]
            return base, True
        elif word.endswith('est'):
            base = word[:-3]
            return base, True
            base = word[:-3]
            return base, True
    
    # Not a comparative/superlative
    return word, False
