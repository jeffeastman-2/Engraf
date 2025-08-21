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
        
        # Static fields to replace dynamic attributes used in Layer 2/3
        self.spatial_vector = None       # Set during Layer 3 processing
        self.vector_text = None          # Expected for vector coordinates
        self.spatial_location = None     # Expected for spatial processing
        self.locX = None                 # Expected for spatial processing
        self.locY = None                 # Expected for spatial processing
        self.locZ = None                 # Expected for spatial processing 

    def __repr__(self):
        return f"PrepositionalPhrase(preposition={self.preposition!r}, noun_phrase={self.noun_phrase!r})"

    def apply_preposition(self, tok):
        self.preposition = tok.word
        self.vector = tok

    def apply_vector(self, tok):
        self.vector += tok

    def apply_np(self, np):
        self.noun_phrase = np