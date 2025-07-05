from engraf.lexer.vector_space import VectorSpace

class POSBase:
    def __init__(self)  

    def to_vector(self):
        raise NotImplementedError("Subclasses must implement to_vector")



class POSTerminal(POSBase):
    """
    Represents a terminal part of speech (POS) in the sentence.
    This is the base class for all terminal POS types.
    """
    def __init__(self, word, vector: VectorSpace):
        super().__init__():
        self.vector = vector
        self.word = word

    def __repr__(self):
        return f"POSTerminal({self.word}, vector={self.vector})"

    def __str__(self):
        return f"POSTerminal: {self.word or 'unknown'}"

    def to_vector(self):
        return self.vector

    def modify(self, target: VectorSpace):
        # Optional: scalar multiplier, e.g., adverbs modifying adjectives
        target *= self.vector

class POSNonTerminal(POSBase):
    def __init__(self, children=None):
        self.children = children or []

    def add(self, child: POSBase):
        self.children.append(child)
