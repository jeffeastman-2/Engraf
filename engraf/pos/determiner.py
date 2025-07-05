import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.pos import POSTerminal

class Determiner(POSTerminal):
    """
    Represents a determiner in the sentence.
    """
    def __init__(self, word, vector: VectorSpace):
        super().__init__(word, vector)
        self.type = "determiner"

    def __repr__(self):
        return f"Determiner({self.word}, vector={self.vector})"

    def __str__(self):
        return f"Determiner: {self.vector.word or 'unknown'}"