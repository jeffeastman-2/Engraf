class TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        print(f"TokenStream initialized with tokens: {tokens}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def get(self):
        if self.position < len(self.tokens):
            tok = self.tokens[self.position]
            self.position += 1
            return tok
        return None

    def mark(self):
        return self.position

    def rewind(self, mark):
        self.position = mark
        
import re

def tokenize(sentence):
    pattern = re.compile(
        r"""\s*(
            \[             # opening bracket
            \s*-?\d+(\.\d+)?\s*,     # x
            \s*-?\d+(\.\d+)?\s*,     # y
            \s*-?\d+(\.\d+)?\s*      # z
            \]             # closing bracket
            | \w+          # normal word
            | [^\w\s]      # punctuation
        )""",
        re.VERBOSE,
    )

    tokens = pattern.findall(sentence)
    flat_tokens = [t[0] for t in tokens]

    # Post-process vector literals into structured tokens
    result = []
    for tok in flat_tokens:
        if tok.startswith("[") and tok.endswith("]"):
            nums = re.findall(r"-?\d+(?:\.\d+)?", tok)
            result.append({"type": "VECTOR", "value": list(map(float, nums))})
        else:
            result.append(tok.lower())
    return result
