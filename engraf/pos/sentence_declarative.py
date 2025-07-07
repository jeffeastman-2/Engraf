import numpy as np
from engraf.lexer.vector_space import VectorSpace

class SentenceDeclarative():
    def __init__(self, subject: NounPhrase, predicate: VerbPhrase):
        self.subject = subject
        self.predicate = predicate

    def to_vector(self):
        # Optional: vector for the entire sentence
        return self.predicate.to_vector()

    def __repr__(self):
        return f"SentenceDeclarative(subject={self.subject}, predicate={self.predicate})"