from engraf.lexer.token_stream import TokenStream
from engraf.utils.predicates import is_preposition, is_noun_phrase_token
from engraf.atn.core import ATNState
from engraf.pos.prepositional_phrase import PrepositionalPhrase

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(pp:PrepositionalPhrase, ts:TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-AFTER-PREP")
    end = ATNState("PP-END")

    # PREP
    start.add_arc(is_preposition, lambda _, tok: pp.apply_preposition(tok), after_prep)
    
    # Handle NounPhrase tokens from Layer 2 directly
    after_prep.add_arc(is_noun_phrase_token, lambda _, tok: pp.apply_np(tok._original_np), end)

    return start, end
