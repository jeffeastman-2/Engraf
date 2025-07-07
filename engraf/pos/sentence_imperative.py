import numpy as np
from engraf.lexer.vector_space import VectorSpace

class SentenceImperative():
    """Represents an imperative sentence in a sentence structure.
    This is a non-terminal part of speech (POS) type.
    """
    
    def __init__(self, verb, verb_phrase: VerbPhrase):
        self.verb = verb  # Assuming verb is a string or similar identifier
        self.verb_phrase = verb_phrase if verb_phrase is not None else []  # Default to empty list if None

    def __repr__(self):
        return f"SentenceImperative(vector={self.vector}, verb_phrase={self.verb_phrase})"