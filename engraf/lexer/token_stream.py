from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE  
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.vocabulary import get_from_vocabulary
import re

class TokenStream:
    def __init__(self, tokens: list[VectorSpace]):
        self.tokens = tokens
        self.position = 0
        #print("TokenStream initialized with VectorSpace tokens:", self.tokens.__repr__())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def _to_vector(self, token: str) -> VectorSpace:
        if token in SEMANTIC_VECTOR_SPACE:
            return SEMANTIC_VECTOR_SPACE[token]
        else:
            # Default: return empty vector with no POS dimensions set
            return VectorSpace(pos="")  # neutral/no POS

    def __repr__(self):
        return f"TokenStream(pos={self.position}, total={len(self.tokens)})"

    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def next(self):
        if self.position < len(self.tokens):
            tok = self.tokens[self.position]
            self.position += 1
            return tok
        return None

    def reset(self):
        self.position = 0

    def advance(self):
        if self.position < len(self.tokens):
            self.position += 1
        else:
            raise IndexError("TokenStream position out of bounds")

import re
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import VectorSpace, vector_from_features

def tokenize(sentence):
    pattern = re.compile(
        r"""\s*(
            \[             # opening bracket
            \s*-?\d+(\.\d+)?\s*,     # x
            \s*-?\d+(\.\d+)?\s*,     # y
            \s*-?\d+(\.\d+)?\s*      # z
            \]             # closing bracket
            | -?\d+(?:\.\d+)?         # standalone number (int or float)
            | '[\w\s]+'     # quoted words (single quotes)
            | \w+           # normal word
            | [^\w\s]       # punctuation
        )""",
        re.VERBOSE,
    )
    tokens = pattern.findall(sentence)
    flat_tokens = [t[0] for t in tokens]

    result = []
    for tok in flat_tokens:
        if tok.startswith("'") and tok.endswith("'"):
            tok = tok[1:-1]
            vs = vector_from_features(pos="quoted")  # empty vector with unknown POS
            vs.word = tok
            result.append(vs)
        elif tok.startswith("[") and tok.endswith("]"):
            nums = re.findall(r"-?\d+(?:\.\d+)?", tok)
            x, y, z = map(float, nums)
            result.append(vector_from_features(loc=[x, y, z], pos="vector"))  
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", tok):  # match integers and floats
            vs = vector_from_features(pos="det def number")
            vs["number"] = float(tok)
            result.append(vs)
        else:
                vs = get_from_vocabulary(tok.lower())
                if vs is None:
                    raise ValueError(f"Unknown token: {tok}")
                vs.word = tok.lower()  # Store the original word for reference
                result.append(vs)
    return result
