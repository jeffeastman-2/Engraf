import numpy as np
from typing import Optional
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase

class VerbPhrase():
    def __init__(self):
        super().__init__()
        self.verb = None
        self.object_np = None

    def to_vector(self) -> VectorSpace:
        # Combine verb meaning with its object’s vector (if present)
        v = self.verb.to_vector().copy()
        if self.object_np:
            v += self.object_np.to_vector()
        return v

    def verb_action(self, tok):
        this.verb = tok.word
        this.vector = tok

