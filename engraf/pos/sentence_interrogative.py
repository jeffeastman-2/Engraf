import numpy as np
x
class SentenceInterrogative():
    """Represents an interrogative sentence in a sentence structure.
    This is a non-terminal part of speech (POS) type.
    """
    
    def __init__(self, verb_phrase, noun_phrase=None):
        self.verb_phrase = verb_phrase
        self.noun_phrase = noun_phrase


    def __repr__(self):
        return f"SentenceImperative"