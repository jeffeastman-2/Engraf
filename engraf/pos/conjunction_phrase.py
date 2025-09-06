

class ConjunctionPhrase:
    def __init__(self, tok, left=None, right= None):
        self.left = left
        self.conjunction = tok.word
        self.vector = tok
        self.right = right

    def __iter__(self):
        yield from self.flatten()

    def flatten(self):
        current = self
        while isinstance(current, ConjunctionPhrase):
            yield current.left
            current = current.right
        yield current  # final right-most item

    def __repr__(self):
        return f"ConjunctionPhrase({self.left}  {self.conjunction}  {self.right})"

    def get_last(self):
        current = self
        while isinstance(current.right, ConjunctionPhrase):
            current = current.right
        print(f"++ get_last returning {current}")
        return current

    def printString(self):
        """Print the string representation of the conjunction phrase."""
        flattened = self.flatten()
        parts = [f"{item.printString()}" for item in flattened]
        str =  "(" + f" {self.conjunction} ".join(parts) + ")"
        return str