from engraf.lexer.vector_space import VectorSpace   


class NounPhrase():
    def __init__(self, noun=None):
        self.vector = VectorSpace()
        self.noun = noun
        self.pronoun = None
        self.determiner = None
        self.preps = []
        self.scale_factor = 1.0
        self.proper_noun = None  # For proper noun names like "called 'Charlie'"
        self.consumed_tokens = []  # All original tokens that were consumed to build this NP
        self.resolved_object = None  # Resolved SceneObject after semantic grounding (LATN Layer 2)

    def apply_determiner(self, tok):
        self.determiner = tok.word
        self.vector += tok
        self.consumed_tokens.append(tok)

    def apply_vector(self, tok):
        self.noun = "vector"
        self.vector += tok
        self.consumed_tokens.append(tok)

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        if not hasattr(self, 'scale_vector') or self.scale_vector is None:
            self.scale_vector = VectorSpace()
        print(f"✅ Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")
        self.consumed_tokens.append(tok)

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
        self.consumed_tokens.append(tok)


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
        self.consumed_tokens.append(tok)
    
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
        self.consumed_tokens.append(tok)

    def apply_proper_noun(self, name_token, has_determiner=False):
        """Apply a proper noun from 'called' or 'named' syntax.
        
        Args:
            name_token: The quoted name token
            has_determiner: True if preceded by 'a/an', making it a type designation
        """
        if has_determiner:
            # "called a 'sun'" - type designation, not proper noun
            print(f"✅ NP applying type designation: {name_token.word}")
            # This will be handled as a regular noun with determiner
        else:
            # "called 'Charlie'" - proper noun
            print(f"✅ NP applying proper noun: {name_token.word}")
            self.proper_noun = name_token.word
        # Don't add to vector as this is a naming directive, not semantic content

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
        consumed_words = [tok.word for tok in self.consumed_tokens]
        parts = [f"noun={self.noun}", f"determiner={self.determiner}", f"vector={self.vector}", f"PPs={self.preps}", f"consumed={consumed_words}"]
        if self.resolved_object:
            parts.append(f"resolved_to={self.resolved_object.object_id}")
        return f"NP({', '.join(parts)})"
    
    def get_consumed_words(self):
        """Return list of words from consumed tokens."""
        return [tok.word for tok in self.consumed_tokens]
    
    def get_original_text(self):
        """Reconstruct the original text from consumed tokens."""
        return " ".join(self.get_consumed_words())
    
    def token_span(self):
        """Return the span (start, end) of consumed tokens for error reporting."""
        if not self.consumed_tokens:
            return (0, 0)
        # Assuming tokens have position information (would need to be added to token structure)
        return (0, len(self.consumed_tokens))  # Placeholder - would use actual positions
    
    def resolve_to_scene_object(self, scene_object):
        """Resolve this NP to a specific SceneObject (LATN Layer 2 semantic grounding).
        
        Args:
            scene_object: The SceneObject this NP refers to
        """
        self.resolved_object = scene_object
        
    def is_resolved(self):
        """Check if this NP has been resolved to a scene object."""
        return self.resolved_object is not None
        
    def get_resolved_object(self):
        """Get the resolved SceneObject, or None if not resolved."""
        return self.resolved_object