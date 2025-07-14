import numpy as np
from typing import Optional
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase

class VerbPhrase():
    def __init__(self, verb=None):
        super().__init__()
        self.verb = verb
        self.noun_phrase = None
        self.preps = []
        self.adjective_complement = None
        self.amount = None  # For handling measurements like "45 degrees"

    def __repr__(self):
        return f"VerbPhrase(verb={self.verb}, noun_phrase={self.noun_phrase}, PPs={self.preps}, adjective_complement={self.adjective_complement}, amount={self.amount})"

    def to_vector(self) -> VectorSpace:
        # Combine verb meaning with its object’s vector (if present)
        v = self.verb.to_vector().copy()
        if self.noun_phrase:
            v += self.noun_phrase.to_vector()
        return v

    def apply_verb(self, tok):
        self.verb = tok.word
        self.vector = tok

    def apply_np(self, np):
        self.noun_phrase = np

    def apply_amount(self, amount_np):
        """Apply an amount/measure noun phrase like '45 degrees'"""
        self.amount = amount_np
        print(f"✅ VP applying amount: {amount_np}")

    def apply_pp(self, pp_obj):
        print(f"✅ VP applying PP: {pp_obj}")
        self.preps.append(pp_obj)

    def apply_adjective(self, tok):
        self.adjective_complement = tok.word
        print(f"✅ VP applying adjective complement: {tok.word}")

    def is_imperative(self):
        return self.vector.scalar_projection("action") > 0.5

    def verb_has_intent(self, intent: str, threshold=0.5) -> bool:
        return self.verb and self.verb.scalar_projection(intent) > threshold


