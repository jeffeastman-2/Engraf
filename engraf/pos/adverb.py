import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.pos import POSTerminal

class Adverb(POSTerminal):
    """
    Adverb class for handling adverbs in the Engraf language.
    Inherits from POSNonterminal.
    """

    def __init__(self, word, vector: VectorSpace):
        super().__init__(word, vector)
     
    def __repr__(self):
        return f"Adverb(word={self.word}, vector={self.vector})"