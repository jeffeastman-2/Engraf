import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.pos import POSTerminal

class Noun(POSTerminal):
    """Represents a noun in the sentence.
    This is a non-terminal part of speech (POS) type.
    """   
     def __init__(self, word, vector: VectorSpace):
        super().__init__(word, vector)

    def __repr__(self):
        return f"Noun(word={self.word}, vector={self.vector})"  



