

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
    
def __eq__(self, other):
    """Deep equality comparison using flatten() to compare tree structures."""
    if not isinstance(other, ConjunctionPhrase):
        return False
    
    # Use flatten() to get the linear representation of both trees
    self_flattened = list(self.flatten())  # Convert generator to list
    other_flattened = list(other.flatten())  # Convert generator to list

    # Compare the flattened lists
    if len(self_flattened) != len(other_flattened):
        return False
        
    for i in range(len(self_flattened)):
        if self_flattened[i] != other_flattened[i]:  # Complete the comparison
            return False
            
    return True

def __hash__(self):
    """Hash method using flattened representation."""
    # Hash the flattened structure - convert generator to tuple
    flattened = tuple(self.flatten())  # Convert generator to tuple for hashing
    return hash(flattened)