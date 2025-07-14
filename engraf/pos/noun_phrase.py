from engraf.lexer.vector_space import VectorSpace   


class NounPhrase():
    def __init__(self, noun=None):
        self.vector = VectorSpace()
        self.noun = noun
        self.pronoun = None
        self.determiner = None
        self.preps = []
        self.scale_factor = 1.0

    def apply_determiner(self, tok):
        self.determiner = tok.word
        self.vector += tok

    def apply_vector(self, tok):
        self.noun = "vector"
        self.vector += tok

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        if not hasattr(self, 'scale_vector') or self.scale_vector is None:
            self.scale_vector = VectorSpace()
        print(f"✅ Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")

    def apply_adjective(self, tok):
        """Apply the adverb-scaled adjective to the NP vector."""
        scale = getattr(self, "scale_vector", None)
        if scale:
            strength = scale.scalar_projection("adv")
            print(f"✅ Scaling adjective {tok.word} by {strength}")
            print(f"✅ Adjective vector before scale: {tok}")
            self.vector += tok * strength
            self.scale_vector = None
        else:
            print(f"✅ Setting adjective vector without scale: {tok}")
            self.vector += tok


    def apply_noun(self, tok):
        print(f"✅ NP applying noun {tok.word} with vector {tok}")
        self.noun = tok.word
        self.vector += tok

    def apply_pronoun(self, tok):
        print(f"✅ NP applying pronoun {tok.word} with vector {tok}")
        self.pronoun = tok.word
        self.vector += tok

    def apply_pp(self, pp_obj):
        print(f"✅ Np applying PP: {pp_obj}")
        self.preps.append(pp_obj)
        self.vector += pp_obj.vector

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
        return f"NP(noun={self.noun}, determiner= {self.determiner}, vector={self.vector}, PPs={self.preps})"          