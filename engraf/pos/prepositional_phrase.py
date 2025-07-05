import

class PrepositionalPhrase(POSNounPhrase):
    """Represents a prepositional phrase in a sentence.
    This is a non-terminal part of speech (POS) type.
    """
    def __init__(self, preposition: str, noun_phrase=None):
        super().__init__()
        self.preposition = preposition
        self.noun_phrase = noun_phrase if noun_phrase is not None else []   