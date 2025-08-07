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
    "pronoun",   # pronouns (it, they, them)
    
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
    
    # Semantic preposition dimensions
    "spatial_vertical",      # vertical relationships: above/over (+), below/under (-), on (contact)
    "spatial_proximity",     # proximity relationships: near (+), at (specific), in (containment)
    "directional_target",    # directional movement: to (+), from (-)
    "directional_agency",    # agency/means: by (+), with (accompaniment)
    "relational_possession", # possession/part-of: of (belongs to, part of)
    "relational_comparison"  # comparison baseline: than (comparison reference)
]