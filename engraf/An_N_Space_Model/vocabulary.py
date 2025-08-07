from engraf.lexer.vector_space import vector_from_features
from engraf.lexer.vector_space import VectorSpace

SEMANTIC_VECTOR_SPACE = {
    # Nouns
    'cube': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'box': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'sphere': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'ellipsoid': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
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
    'the': vector_from_features("det def singular", number=1.0),
    'one': vector_from_features("det def singular", number=1.0),
    'two': vector_from_features("det def plural", number=2.0),
    'three': vector_from_features("det def plural", number=3.0),
    'four': vector_from_features("det def plural", number=4.0),
    'five': vector_from_features("det def plural", number=5.0),
    'six': vector_from_features("det def plural", number=6.0),
    'seven': vector_from_features("det def plural", number=7.0),
    'eight': vector_from_features("det def plural", number=8.0),
    'nine': vector_from_features("det def plural", number=9.0),
    'ten': vector_from_features("det def plural", number=10.0),
    'a': vector_from_features("det singular", number=1.0),
    'an': vector_from_features("det singular", number=1.0),

    # Verbs
    # create
    'create': vector_from_features("verb action create"),
    'draw': vector_from_features("verb action create"),
    'make': vector_from_features("verb action create scale style"),
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

    # modify
    'move': vector_from_features("verb action move"),
    'rotate': vector_from_features("verb action rotate"),
    'xrotate': vector_from_features("verb action rotate", rotX=1.0),  # rotate around x-axis
    'yrotate': vector_from_features("verb action rotate", rotY=1.0),  # rotate around y-axis
    'zrotate': vector_from_features("verb action rotate", rotZ=1.0),  # rotate around z-axis
    'scale': vector_from_features("verb action scale"),

    # generic (no third term)
    'redo': vector_from_features("verb action"),
    'undo': vector_from_features("verb action"),

    # Prepositions with semantic dimensions
    # Spatial vertical relationships
    'on': vector_from_features("prep", spatial_vertical=0.5),      # contact-high
    'over': vector_from_features("prep", spatial_vertical=1.0),    # higher
    'above': vector_from_features("prep", spatial_vertical=1.0),   # higher
    'under': vector_from_features("prep", spatial_vertical=-1.0),  # lower
    'below': vector_from_features("prep", spatial_vertical=-1.0),  # lower
    
    # Spatial proximity relationships
    'in': vector_from_features("prep", spatial_proximity=0.3),     # containment
    'at': vector_from_features("prep", spatial_proximity=0.5),     # specific location
    'near': vector_from_features("prep", spatial_proximity=1.0),   # close
    
    # Directional/movement relationships
    'to': vector_from_features("prep", directional_target=1.0),    # toward destination
    'from': vector_from_features("prep", directional_target=-1.0), # away from source
    
    # Agency/instrumentality relationships
    'by': vector_from_features("prep", directional_agency=1.0),    # agent/means
    'with': vector_from_features("prep", directional_agency=0.7),  # accompaniment/instrument
    
    # Relational dimensions
    'of': vector_from_features("prep", relational_possession=1.0), # belongs to, part of
    'than': vector_from_features("prep", relational_comparison=1.0), # comparison baseline
    
    # Additional verbs for time travel
    'go': vector_from_features("verb action"),
    'back': vector_from_features("adv"),
    'forward': vector_from_features("adv"),
    'time': vector_from_features("noun"),
    
    # Conjunctions
    'and': vector_from_features("conj"),
    
    # Disjunctions
    'or': vector_from_features("disj"),
    
    # Negation
    'not': vector_from_features("neg"),
    'no': vector_from_features("neg"),
    
    # Modal verbs
    'can': vector_from_features("verb modal"),
    'could': vector_from_features("verb modal"),
    'may': vector_from_features("verb modal"),
    'might': vector_from_features("verb modal"),
    'must': vector_from_features("verb modal"),
    'shall': vector_from_features("verb modal"),
    'should': vector_from_features("verb modal"),
    'will': vector_from_features("verb modal"),
    'would': vector_from_features("verb modal"),
    
    # Question markers
    'who': vector_from_features("question"),
    'what': vector_from_features("question"),
    'where': vector_from_features("question"),
    'when': vector_from_features("question"),
    'why': vector_from_features("question"),
    'how': vector_from_features("question"),
    'which': vector_from_features("question"),
    
    # To be verbs
    'is': vector_from_features("tobe"),
    'are': vector_from_features("tobe"),
    'was': vector_from_features("tobe"),
    'were': vector_from_features("tobe"),
    'be': vector_from_features("tobe"),
    'been': vector_from_features("tobe")
}
