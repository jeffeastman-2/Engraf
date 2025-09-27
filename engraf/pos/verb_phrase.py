
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.debug import debug_print

class VerbPhrase():
    def __init__(self):
        # VerbPhrase doesn't need to call super() if it doesn't inherit from anything
        self.verb = None
        self.vector = VectorSpace()
        self.noun_phrase = None
        self.preps = []
        self.adjective_complement = []  # Changed to list for multiple adjectives
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
        if self.verb is None:
            self.verb = tok.word
        else:
            self.verb = (f"{self.verb} {tok.word}")
        self.vector += tok

    def apply_adverb(self, tok):
        if self.verb is None:
            self.verb = tok.word
        else:
            self.verb = (f"{self.verb} {tok.word}")

    def apply_conjunction(self, tok):
        self.verb = (f"{self.verb} {tok.word}")

    def apply_np(self, np):
        self.noun_phrase = np

    def apply_amount(self, amount_np):
        """Apply an amount/measure noun phrase like '45 degrees'"""
        self.amount = amount_np
        debug_print(f"✅ VP applying amount: {amount_np}")

    def apply_pp(self, pp_obj):
        debug_print(f"✅ VP applying PP: {pp_obj}")
        self.preps.append(pp_obj)

    def apply_adjective(self, tok):
        self.adjective_complement.append(tok.word)
        debug_print(f"✅ VP applying adjective complement: {tok.word}")
        debug_print(f"✅ Current adjective complements: {self.adjective_complement}")

    def is_imperative(self):
        return self.vector.scalar_projection("action") > 0.5

    def verb_has_intent(self, intent: str, threshold=0.5) -> bool:
        return self.verb and self.verb.scalar_projection(intent) > threshold

    def printString(self):
        str = self.verb
        if self.preps:
            for pp in self.preps:
                str += f" ({pp.printString()})"
        if self.adjective_complement:
            adjectives = " ".join(self.adjective_complement)
            str += f" ({adjectives})"
        if self.noun_phrase:
            str += f" [{self.noun_phrase.printString()}]"
        if self.amount:
            str += f" [{self.amount.printString()}]"
        return str

    def equals(self, other):
        """Deep equality comparison for VerbPhrase objects."""
        if not isinstance(other, VerbPhrase):
            return False
        if len(self.preps) != len(other.preps):
            return False
        for this_prep, other_prep in zip(self.preps, other.preps):
            if not this_prep.equals(other_prep):
                return False  
        if len(self.adjective_complement) != len(other.adjective_complement):
            return False
        for this_adj, other_adj in zip(self.adjective_complement, other.adjective_complement):
            if this_adj != other_adj:
                return False
        return (
            self.verb == other.verb and
            self.noun_phrase.equals(other.noun_phrase) and
            self.amount == other.amount and
            getattr(self, 'vector', None) == getattr(other, 'vector', None)
        )
