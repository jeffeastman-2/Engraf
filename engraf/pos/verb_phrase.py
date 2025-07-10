import numpy as np
from typing import Optional
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase

class VerbPhrase():
    def __init__(self):
        super().__init__()
        self.verb = None
        self.noun_phrase = None

    def __repr__(self):
        return f"VerbPhrase(verb={self.verb}, noun_phrase={self.noun_phrase})"

    def to_vector(self) -> VectorSpace:
        # Combine verb meaning with its object’s vector (if present)
        v = self.verb.to_vector().copy()
        if self.noun_phrase:
            v += self.noun_phrase.to_vector()
        return v

    def apply_verb(self, tok):
        self.verb = tok.word
        self.vector = tok

    def apply_np(self, np):
        self.noun_phrase = np
