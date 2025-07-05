import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.pos import POSTerminal

class Adjective(POSTerminal):
    """
    Represents an adjective in the sentence.
    This is a non-terminal part of speech (POS) type.
    """
    def __init__(self, word, vector: VectorSpace):
        super().__init__(word, vector)

    def __repr__(self):
        return f"Adjective({self.word}, vector={self.vector})"

    def __str__(self):
        return f"Adjective: {self.vector.word or 'unknown'}"