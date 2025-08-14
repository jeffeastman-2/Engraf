import re
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE

# Common verb inflection patterns
VERB_INFLECTION_PATTERNS = [
    # Past participle patterns (for -ed ending)
    (r"(.+)ed$", "verb_past_part", lambda m: [m.group(1), m.group(1) + "e"]),  # called -> call, named -> name
    
    # Present participle patterns (for -ing ending) 
    (r"(.+)ing$", "verb_present_part", lambda m: [m.group(1), m.group(1) + "e"]),  # calling -> call, naming -> name
]

# Irregular verb forms that don't follow standard patterns
# Note: Only include forms that are NOT already in the main vocabulary
IRREGULAR_VERB_FORMS = {
    # Common irregular past participles (only if not in main vocab)
    "done": ("do", "verb_past_part"), 
    "gone": ("go", "verb_past_part"),
    "seen": ("see", "verb_past_part"),
    "taken": ("take", "verb_past_part"),
    "given": ("give", "verb_past_part"),
    "made": ("make", "verb_past_part"),
    "said": ("say", "verb_past_part"),
    "told": ("tell", "verb_past_part"),
    "found": ("find", "verb_past_part"),
    
    # Common irregular past tense forms (only if not in main vocab)
    "had": ("have", "verb_past"),
    "did": ("do", "verb_past"),
    "went": ("go", "verb_past"),
    "saw": ("see", "verb_past"),
    "took": ("take", "verb_past"),
    "gave": ("give", "verb_past"),
    "said": ("say", "verb_past"),
    "told": ("tell", "verb_past"),
    "found": ("find", "verb_past"),
}

def find_root_verb(inflected_word):
    """
    Try to find the root form of an inflected verb.
    Returns (root_verb, inflection_type, found) tuple where:
    - root_verb is the base form 
    - inflection_type is the verb inflection dimension to set
    - found is True if a root was found in vocabulary
    """
    word_lower = inflected_word.lower()
    
    # Check if it's already a known word
    if word_lower in SEMANTIC_VECTOR_SPACE:
        return word_lower, None, True
    
    # Check irregular forms first
    if word_lower in IRREGULAR_VERB_FORMS:
        root, inflection_type = IRREGULAR_VERB_FORMS[word_lower]
        if root in SEMANTIC_VECTOR_SPACE:
            # Check if root has verb or tobe dimension (both are verb-like)
            root_vector = SEMANTIC_VECTOR_SPACE[root]
            if root_vector["verb"] > 0 or root_vector["tobe"] > 0:
                return root, inflection_type, True
    
    # Try regular inflection patterns
    for pattern, inflection_type, root_candidates_func in VERB_INFLECTION_PATTERNS:
        match = re.match(pattern, word_lower)
        if match:
            # Try each possible root form
            candidates = root_candidates_func(match)
            for candidate in candidates:
                if candidate in SEMANTIC_VECTOR_SPACE:
                    # Check if candidate has verb or tobe dimension
                    candidate_vector = SEMANTIC_VECTOR_SPACE[candidate]
                    if candidate_vector["verb"] > 0 or candidate_vector["tobe"] > 0:
                        return candidate, inflection_type, True
    
    return word_lower, None, False

def is_verb_inflection(word):
    """
    Check if a word appears to be a verb inflection that could have a root form.
    """
    _, _, found = find_root_verb(word)
    return found and word.lower() not in SEMANTIC_VECTOR_SPACE
