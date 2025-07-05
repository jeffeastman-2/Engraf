import numpy as np
from typing import Optional
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.pos import POSNonTerminal
from engraf.pos.noun_phrase import NounPhrase

class VerbPhrase(POSNonTerminal):
    def __init__(self, verb: POSTerminal, object_np: Optional[NounPhrase] = None):
        super().__init__()
        self.verb = verb
        self.object_np = object_np

    def to_vector(self) -> VectorSpace:
        # Combine verb meaning with its object’s vector (if present)
        v = self.verb.to_vector().copy()
        if self.object_np:
            v += self.object_np.to_vector()
        return v
