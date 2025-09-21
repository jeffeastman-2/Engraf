import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.utils.debug import debug_print

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
        return f"Sentence(subject={self.subject}, predicate={self.predicate})"

    def store_definition_word(self, tok):
        self.definition_word = tok.word
        self.vector = tok  # Store the vector representation of the quoted word

    def apply_subject_conjunction(self, tok):
        debug_print(f"✅ => Applying subject conjunction '{tok.word}'")
        if isinstance(self.subject, ConjunctionPhrase):
            tail = self.subject.get_last()   
            if tail.right is None: 
                tail.right = ConjunctionPhrase(tok)
                debug_print(f"✅ => setting tail.right in {self}")
            else: 
                tail.right = ConjunctionPhrase(tok, left=tail.right)
        else:
            self.subject = ConjunctionPhrase(tok, left=self.subject)
            
    def apply_predicate_conjunction(self, tok):
        debug_print(f"✅ => Applying predicate conjunction '{tok.word}'")
        if isinstance(self.predicate, ConjunctionPhrase):
            tail = self.predicate.get_last()    
            if tail.right is None:
                tail.right = ConjunctionPhrase(tok)
                debug_print(f"✅ => setting tail.right in {self}")
            else: 
                tail.right = ConjunctionPhrase(tok, left=tail.right)
        else:
            self.predicate = ConjunctionPhrase(tok, left=self.predicate)

    def apply_subject_token(self, token):
        self.apply_subject(token._original_np)

    def apply_subject(self, subj):
        debug_print(f"✅ => Applying sentence subject {subj} \n      to {self}")
        if self.subject is None:
            self.subject = subj
        elif self.subject == subj:
            return
        elif isinstance(self.subject, ConjunctionPhrase):
            tail = self.subject.get_last()
            if tail.right is  None:
                tail.right = subj
            elif tail.right == subj:
                return
            else: debug_print(f"⚠️ ERROR: tail.right is not None {self}")
 
    def apply_predicate_token(self, token):
        self.apply_predicate(token._original_vp)

    def apply_predicate(self, pred):
        debug_print(f"✅ => Applying sentence predicate {pred} \n      to {self}")
        if self.predicate is None:
            self.predicate = pred
        elif pred == self.predicate:
            # Avoid wrapping the same predicate twice
            return
        elif isinstance(self.predicate, ConjunctionPhrase):
            tail = self.predicate.get_last()
            if tail.right is None:
                tail.right = pred
            elif tail.right == pred:
                return
            else:
                debug_print(f"⚠️ ERROR: tail.right not None {self}")

    def apply_prepositional_phrase(self, pp_token):
        """Apply a PP token to the sentence structure."""
        if not hasattr(self, 'prepositional_phrases'):
            self.prepositional_phrases = []
        self.prepositional_phrases.append(pp_token)
        return True

    def apply_tobe(self, tok):
        debug_print(f"✅ => Applying sentence tobe {tok}")
        self.tobe = tok.word
        self.vector += tok

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        self.scale_vector = getattr(self, "scale_vector", VectorSpace())
        debug_print(f"Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")

    def apply_adjective(self, tok):
        """Apply the adverb-scaled adjective to the NP vector."""
        scale = getattr(self, "scale_vector", None)
        if scale:
            strength = scale.scalar_projection("adv")
            debug_print(f"Scaling adjective {tok.word} by {strength}")
            debug_print(f"Adjective vector before scale: {tok}")
            self.vector += tok * strength
            self.scale_vector = None
        else:
            self.vector += tok

    def __eq__(self, other):
        """Deep equality comparison for SentencePhrase objects."""
        if not isinstance(other, SentencePhrase):
            return False
        
        return (
            getattr(self, 'subject', None) == getattr(other, 'subject', None) and
            getattr(self, 'predicate', None) == getattr(other, 'predicate', None) and
            getattr(self, 'definition_word', None) == getattr(other, 'definition_word', None) and
            getattr(self, 'tobe', None) == getattr(other, 'tobe', None) and
            getattr(self, 'scale_factor', 1.0) == getattr(other, 'scale_factor', 1.0) and
            getattr(self, 'prepositional_phrases', []) == getattr(other, 'prepositional_phrases', [])
        )

    def __hash__(self):
        """Hash method for SentencePhrase objects."""
        return hash((
            getattr(self, 'subject', None),
            getattr(self, 'predicate', None),
            getattr(self, 'definition_word', None),
            getattr(self, 'tobe', None),
            getattr(self, 'scale_factor', 1.0),
            tuple(getattr(self, 'prepositional_phrases', []))
        ))

    def printString(self):
        """Print a string representation of the sentence."""
        subject_str = self.subject.printString() if self.subject else None
        predicate_str = self.predicate.printString() if self.predicate else "[No Predicate]"
        if subject_str:
            return f"{subject_str} {predicate_str}"
        else:
            return f"{predicate_str}"


