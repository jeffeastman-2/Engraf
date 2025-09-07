from engraf.lexer.vector_space import VectorSpace   
 

class PrepositionalPhrase():
    """Represents a prepositional phrase in a sentence.
    This is a non-terminal part of speech (POS) type.
    """
    def __init__(self):
        super().__init__()
        self.preposition = None
        self.noun_phrase = None 
        self.vector = VectorSpace()
        
    def __repr__(self):
        return f"PrepositionalPhrase(preposition={self.preposition!r}, noun_phrase={self.noun_phrase!r})"

    def apply_preposition(self, tok):
        self.preposition = tok.word
        self.vector = tok

    def apply_vector(self, tok):
        self.vector += tok

    def apply_np(self, np):
        self.noun_phrase = np

    def printString(self):
        if self.vector:
            return f"{self.preposition} + {self.vector.non_zero_dims()}"
        return f"{self.preposition} + {self.noun_phrase.printString() if self.noun_phrase else ''}"