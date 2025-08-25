VECTOR_DIMENSIONS = [
    # Basic grammatical categories
    "verb",      # action words (draw, move, create)
    "tobe",      # copula verbs (is, are, was, were, be, been)
    "action",    # action semantics marker
    "prep",      # prepositions (to, from, above, below)
    "det",       # determiners (a, an, the, one, two)
    "def",       # definite articles and specific determiners
    "adv",       # adverbs and adverbial modifiers (very, extremely)
    "adj",       # adjectives (red, large, smooth)
    "noun",      # nouns (cube, sphere, object)
    "proper_noun", # proper nouns (user-assigned names like 'sun', 'fred')
    "pronoun",   # pronouns (it, they, them)
    "assembly",  # assembly/compound objects (house, car, table_setting)
    "unknown",   # unrecognized tokens that don't match vocabulary
    "NP",        # noun phrase tokens (Layer 2 composite tokens)
    "PP",        # prepositional phrase tokens (Layer 3 composite tokens)
    "VP",        # verb phrase tokens (Layer 4 composite tokens)
    
    # Grammatical features
    "number",    # numeric quantity information
    "vector",    # coordinate/vector literal marker
    "singular",  # singular number agreement
    "plural",    # plural number agreement
    "conj",      # conjunctions (and)
    "disj",      # disjunctions (or)
    "neg",       # negation (not, no)
    "modal",     # modal verbs (can, could, may, might, must, shall, should, will, would)
    "question",  # question markers (who, what, where, when, why,
    "tobe",      # to-be verb marker (duplicate - consider removing)
    "unit",      # measurement units (degree, meter, pixel)
    
    # Verb inflection forms
    "verb_past",        # past tense verb form (called, named, created)
    "verb_past_part",   # past participle form (called, named, created)
    "verb_present_part", # present participle/gerund form (calling, naming, creating)
    
    # Comparative/superlative forms
    "comp",      # comparative forms (bigger, redder, taller)
    "super",     # superlative forms (biggest, reddest, tallest)
    
    # Spatial coordinates
    "locX", "locY", "locZ",           # 3D position coordinates
    "scaleX", "scaleY", "scaleZ",     # 3D scaling factors
    "rotX", "rotY", "rotZ",           # 3D rotation angles
    
    # Visual properties
    "red", "green", "blue",           # RGB color values
    "texture",     # surface texture properties
    "transparency", # opacity/transparency level
    "quoted",      # quoted/literal text marker
    
    # High-level verb intent vectors
    "create",    # creation verbs (draw, create, place, make)
    "transform", # modification verbs (move, rotate, scale, color)
    "move",      # movement verbs (move, shift, translate)
    "rotate",    # rotation verbs (rotate, spin)
    "scale",     # scaling verbs (scale, resize)
    "style",     # styling verbs (color, texture)
    "organize",  # organization verbs (group, ungroup, align, position)
    "edit",      # editing verbs (delete, undo, redo, copy, paste)
    "select",    # selection verbs (select, choose)
    "naming",    # naming verbs (call, name)
    
    # Semantic preposition dimensions
    "spatial_location",      # spatial relationships: prepositions affecting object positioning
    "spatial_proximity",     # proximity relationships: near (+), at (specific), in (containment)
    "directional_target",    # directional movement: to (+), from (-)
    "directional_agency",    # agency/means: by (+), with (accompaniment)
    "relational_possession", # possession/part-of: of (belongs to, part of)
    "relational_comparison"  # comparison baseline: than (comparison reference)
]

# Semantic dimension categories for masking during similarity comparisons
# POS dimensions should not be used for semantic similarity
POS_DIMENSIONS = {
    "verb", "tobe", "action", "prep", "det", "def", "adv", "adj", "noun", 
    "proper_noun", "pronoun", "assembly", "unknown", "NP", "SO", "PP", "VP",
    "number", "vector", "singular", "plural", "conj", "disj", "neg", "modal", 
    "question", "unit", "verb_past", "verb_past_part", "verb_present_part",
    "comp", "super", "quoted"
}

# Semantic content dimensions that should be used for similarity
SEMANTIC_DIMENSIONS = {
    # Spatial coordinates
    "locX", "locY", "locZ", "scaleX", "scaleY", "scaleZ", "rotX", "rotY", "rotZ",
    # Visual properties  
    "red", "green", "blue", "texture", "transparency",
    # High-level verb intent vectors
    "create", "transform", "move", "rotate", "scale", "style", "organize", 
    "edit", "select", "naming",
    # Semantic preposition dimensions
    "spatial_location", "spatial_proximity", "directional_target", 
    "directional_agency", "relational_possession", "relational_comparison"
}

def get_semantic_mask():
    """Return a boolean mask for semantic dimensions only."""
    return [dim in SEMANTIC_DIMENSIONS for dim in VECTOR_DIMENSIONS]

def get_pos_mask():
    """Return a boolean mask for POS dimensions only."""
    return [dim in POS_DIMENSIONS for dim in VECTOR_DIMENSIONS]