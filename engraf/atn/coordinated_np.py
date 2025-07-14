import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, noop
from engraf.utils.predicates import is_conjunction, is_none, is_np_head, any_of, is_verb, is_tobe, is_preposition
from engraf.utils.actions import make_run_np_into_atn, make_run_np_into_conjunction
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase

def build_coordinated_np_atn(coord_np: ConjunctionPhrase, ts: TokenStream):
    """
    Build an ATN for coordinated noun phrases like "a blue box and a green sphere".
    This ATN creates a ConjunctionPhrase that contains multiple NounPhrase objects.
    """
    start = ATNState("COORD-NP-START")
    after_first_np = ATNState("COORD-NP-AFTER-FIRST")
    after_conj = ATNState("COORD-NP-AFTER-CONJ")
    end = ATNState("COORD-NP-END")

    # Parse the first noun phrase
    start.add_arc(is_np_head, make_run_np_into_atn(ts, fieldname="left"), after_first_np)
    
    # After first NP, look for conjunction
    after_first_np.add_arc(is_conjunction, lambda _, tok: coord_np.apply_conjunction(tok), after_conj)
    # If no conjunction, this is just a simple NP - end
    after_first_np.add_arc(any_of(is_verb, is_tobe, is_preposition, is_none), noop, end)
    
    # After conjunction, parse the next noun phrase
    after_conj.add_arc(is_np_head, make_run_np_into_conjunction(ts), after_first_np)
    
    # End state
    end.add_arc(is_none, noop, end)
    
    return start, end
