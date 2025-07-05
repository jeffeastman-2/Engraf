

class NP(POSNonTerminal):
    def __init__(self, noun: POSTerminal, determiner=none, adjectives=None, preps=None):
        super().__init__()
        self.noun = noun
        self.determiner = determiner
        self.adjectives = adjectives or []
        self.preps = preps or []

    def to_vector(self):
        v = self.noun.to_vector().copy()
        if self.determiner:
            v += self.determiner.to_vector()
        for adj in self.adjectives:
            adj.modify(v)
        for prep in self.preps:
            v += prep.to_vector()
        return v

    def __repr__(self):
        return f"NP(noun={self.noun}, determiner= {self.determiner}, adjectives={self.adjectives}, preps={self.preps})"    