import numpy as np
from engraf.lexer.vector_space import VectorSpace

class SentencePhrase():
    def __init__(self):
        self.vector = VectorSpace()
        self.subject = None
        self.predicate = None
        self.definition_word = None

    def to_vector(self):
        # Optional: vector for the entire sentence
        return self.predicate.to_vector()

    def __repr__(self):
        return f"SentenceDeclarative(subject={self.subject}, predicate={self.predicate})"

    def store_definition_word(self, tok):
        self.definition_word = tok.word
        self.vector = tok  # Store the vector representation of the quoted word
        return ctx

    def apply_subject(self, tok):
        print(f"✅ => Applying sentence subject {tok}")
        self.subject = tok

    def apply_predicate(self,tok):
        print(f"✅ => Applying sentence predcate {tok}")
        self.predicate = tok
