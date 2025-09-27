

class ConjunctionPhrase:
    def __init__(self, tok, phrases=[]):
        self.conjunction = tok.word
        self.vector = tok
        self.phrases = phrases # NPs or PPs or VPs or SPs
        self.preps = []  # Add this for PP attachments


    def __repr__(self):
        return f"ConjunctionPhrase({self.conjunction} {self.phrases})"

    def get_last(self):
        return self.phrases[-1] if self.phrases else None

    def printString(self):
        """Print the string representation of the conjunction phrase."""
        parts = [f"{item.printString()}" for item in self.phrases]
        str =  "(" + f" {self.conjunction} ".join(parts) + ")"
        if self.preps:
            str = '(' + str
            str += " " + " ".join(prep.printString() for prep in self.preps) + ")"
        return str
    
def equals(self, other):
    """Deep equality comparison using phrases to compare tree structures."""
    if not isinstance(other, ConjunctionPhrase):
        return False
    self_phrases = self.phrases
    other_phrases = other.phrases
    # Compare the phrases lists
    if len(self_phrases) != len(other_phrases):
        return False
    for i in range(len(self_phrases)):
        if not self_phrases[i].equals(other_phrases[i]):  # Complete the comparison
            return False            
    return True
