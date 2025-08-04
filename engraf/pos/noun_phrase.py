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
        
        # Check number agreement between determiner and noun using vector dimensions
        if self.determiner and hasattr(self, 'vector') and self.vector:
            # Get number information from vectors
            noun_is_plural = tok["plural"] > 0.0
            noun_is_singular = tok["singular"] > 0.0
            determiner_is_singular = self.vector["singular"] > 0.0
            determiner_number = self.vector["number"]
            
            # Check for number agreement violations using vector dimensions
            if self._has_number_agreement_error(determiner_is_singular, determiner_number, noun_is_plural, noun_is_singular):
                if determiner_is_singular and noun_is_plural:
                    error_msg = f"Number agreement error: singular determiner '{self.determiner}' cannot modify plural noun '{tok.word}'"
                elif determiner_number > 1.0 and noun_is_singular:
                    error_msg = f"Number agreement error: plural determiner '{self.determiner}' (number={determiner_number}) cannot modify singular noun '{tok.word}'"
                else:
                    error_msg = f"Number agreement error between '{self.determiner}' and '{tok.word}'"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
        
        self.noun = tok.word
        self.vector += tok
    
    def _has_number_agreement_error(self, determiner_is_singular, determiner_number, noun_is_plural, noun_is_singular):
        """Check if there's a number agreement error using vector dimensions."""
        # Determine noun number: if not explicitly marked as plural, treat as singular
        is_noun_plural = noun_is_plural > 0.0
        is_noun_singular = not is_noun_plural  # unmarked nouns are singular by default
        
        # Rule 1: Singular determiners (marked with singular=1.0) should only work with singular nouns
        if determiner_is_singular and is_noun_plural:
            return True
            
        # Rule 2: Numeric determiners > 1 should only work with plural nouns
        if determiner_number > 1.0 and is_noun_singular:
            return True
            
        return False

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