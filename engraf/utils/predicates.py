

def is_determiner(tok): return tok is not None and tok.isa("det")
def is_pronoun(tok): return tok is not None and tok.isa("pronoun")
def is_adjective(tok): return tok is not None and tok.isa("adj")
def is_adverb(tok): return tok is not None and tok.isa("adv")
def is_noun(tok): return tok is not None and tok.isa("noun")
def is_verb(tok): return tok is not None and tok.isa("verb")
def is_tobe(tok): return tok is not None and tok.isa("tobe")
def is_preposition(tok): return tok is not None and tok.isa("prep")
def is_number(tok): return tok is not None and tok["number"] > 0.0
def is_none(tok): return tok is None
def is_vector(tok): return tok is not None and tok.isa("vector")
def is_quoted(tok): return tok is not None and tok.isa("quoted")
def is_adjective_or_adverb(tok): return tok is not None and (tok.isa("adj") or tok.isa("adv"))
def any_of(*predicates):
    def combined_predicate(tok):
        return any(pred(tok) for pred in predicates)
    return combined_predicate
def is_conjunction(tok): return tok is not None and tok.isa("conj")
def is_anything(tok): return True
def is_anything_no_consume(tok):
    return True, False
def is_conjunction_no_consume(tok):
    return is_conjunction, False
def is_np_head(tok):
    """Returns True if the token starts a noun phrase."""
    return any_of(is_determiner, is_pronoun, is_vector)(tok)

