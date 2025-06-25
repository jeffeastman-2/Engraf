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
        
