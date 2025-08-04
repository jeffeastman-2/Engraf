from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE, vector_from_word
from engraf.lexer.vector_space import VectorSpace, vector_from_features
from engraf.utils.noun_inflector import singularize_noun
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
    i = 0
    while i < len(flat_tokens):
        tok = flat_tokens[i]
        
        # Check for compound words (look ahead for multi-word tokens in vocabulary)
        if i + 1 < len(flat_tokens) and not tok.startswith("'") and not tok.startswith("[") and not re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
            # Try two-word compound
            two_word = f"{tok} {flat_tokens[i+1]}"
            if two_word.lower() in SEMANTIC_VECTOR_SPACE:
                vs = vector_from_word(two_word.lower())
                if vs is not None:
                    # For compound nouns, store singular form
                    if vs["noun"] > 0:
                        singular_form, was_plural = singularize_noun(two_word)
                        vs.word = singular_form
                    else:
                        vs.word = two_word.lower()
                    result.append(vs)
                    i += 2  # Skip both tokens
                    continue
            
            # Try three-word compound if we have enough tokens
            if i + 2 < len(flat_tokens):
                three_word = f"{tok} {flat_tokens[i+1]} {flat_tokens[i+2]}"
                if three_word.lower() in SEMANTIC_VECTOR_SPACE:
                    vs = vector_from_word(three_word.lower())
                    if vs is not None:
                        # For compound nouns, store singular form
                        if vs["noun"] > 0:
                            singular_form, was_plural = singularize_noun(three_word)
                            vs.word = singular_form
                        else:
                            vs.word = three_word.lower()
                        result.append(vs)
                        i += 3  # Skip all three tokens
                        continue
        
        # Process single token
        # Process single token
        if tok.startswith("'") and tok.endswith("'"):
            tok = tok[1:-1]
            vs = vector_from_features(pos="quoted")  # empty vector with unknown POS
            vs.word = tok
            result.append(vs)
        elif tok.startswith("[") and tok.endswith("]"):
            nums = re.findall(r"-?\d+(?:\.\d+)?", tok)
            x, y, z = map(float, nums)
            vs = vector_from_features(loc=[x, y, z], pos="vector")
            vs.word = tok  # Preserve the original vector word
            result.append(vs)  
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", tok):  # match integers and floats
            vs = vector_from_features(pos="det def number")
            vs.word = tok  # Preserve the original number word
            vs["number"] = float(tok)
            result.append(vs)
        else:
            vs = vector_from_word(tok.lower())
            if vs is None:
                raise ValueError(f"Unknown token: {tok}")
            
            # For nouns, store the singular form in vs.word for consistent object matching
            if vs["noun"] > 0:  # This is a noun
                singular_form, was_plural = singularize_noun(tok)
                vs.word = singular_form
            else:
                vs.word = tok.lower()  # Preserve original for non-nouns
            
            result.append(vs)
        
        i += 1
    
    return result
