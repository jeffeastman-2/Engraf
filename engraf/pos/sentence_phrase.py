import numpy as np
from engraf.lexer.vector_space import VectorSpace

class SentencePhrase():
    def __init__(self):
        self.vector = VectorSpace()
        self.subject = None
        self.predicate = None
        self.definition_word = None
        self.tobe = None
        self.scale_factor = 1.0


    def to_vector(self):
        # Optional: vector for the entire sentence
        return self.predicate.to_vector()

    def __repr__(self):
        return f"SentenceDeclarative(subject={self.subject}, predicate={self.predicate})"

    def store_definition_word(self, tok):
        self.definition_word = tok.word
        self.vector = tok  # Store the vector representation of the quoted word

    def apply_subject(self, tok):
        print(f"✅ => Applying sentence subject {tok}")
        self.subject = tok

    def apply_predicate(self,tok):
        print(f"✅ => Applying sentence predcate {tok}")
        self.predicate = tok

    def apply_tobe(self, tok):
        print(f"✅ => Applying sentence tobe {tok}")
        self.tobe = tok.word
        self.vector += tok

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        self.scale_vector = getattr(self, "scale_vector", VectorSpace())
        print(f"Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")

    def apply_adjective(self, tok):
        """Apply the adverb-scaled adjective to the NP vector."""
        scale = getattr(self, "scale_vector", None)
        if scale:
            strength = scale.scalar_projection("adv")
            print(f"Scaling adjective {tok.word} by {strength}")
            print(f"Adjective vector before scale: {tok}")
            self.vector += tok * strength
            self.scale_vector = None
        else:
            self.vector += tok

