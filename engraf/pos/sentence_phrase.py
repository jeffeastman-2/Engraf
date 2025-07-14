import numpy as np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.conjunction_phrase import ConjunctionPhrase

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
        print(f"✅ => Applying subject conjunction '{tok.word}'")
        if isinstance(self.subject, ConjunctionPhrase):
            tail = self.subject.get_last()   
            if tail.right is None: 
                tail.right = ConjunctionPhrase(tok)
                print(f"✅ => setting tail.right in {self}")
            else: 
                tail.right = ConjunctionPhrase(tok, left=tail.right)
        else:
            self.subject = ConjunctionPhrase(tok, left=self.subject)
            
    def apply_predicate_conjunction(self, tok):
        print(f"✅ => Applying predicate conjunction '{tok.word}'")
        if isinstance(self.predicate, ConjunctionPhrase):
            tail = self.predicate.get_last()    
            if tail.right is None:
                tail.right = ConjunctionPhrase(tok)
                print(f"✅ => setting tail.right in {self}")
            else: 
                tail.right = ConjunctionPhrase(tok, left=tail.right)
        else:
            self.predicate = ConjunctionPhrase(tok, left=self.predicate)

    def apply_subject(self, subj):
        print(f"✅ => Applying sentence subject {subj} \n      to {self}")
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
            else: print(f"⚠️ ERROR: tail.right is not None {self}")
 
    def apply_predicate(self, pred):
        print(f"✅ => Applying sentence predicate {pred} \n      to {self}")
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
                print(f"⚠️ ERROR: tail.right not None {self}")


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


class SentenceImperative(SentencePhrase):
    def __init__(self, subject=None, predicate=None):
        super().__init__(subject, predicate)
        self.action = predicate.verb if predicate else None

    @property
    def object(self):
        return self.predicate.noun_phrase if self.predicate else None
    
    def has_intent(self, intent: str, threshold=0.5) -> bool:
        return self.predicate.verb and self.predicate.verb.scalar_projection(intent) > threshold




def promote_sentence(sentence: SentencePhrase):
    if sentence.subject is None and sentence.predicate and sentence.predicate.is_imperative():
        return SentenceImperative(subject=None, predicate=sentence.predicate)
    return sentence
