from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vector_space import is_preposition, is_vector, is_determiner, is_pronoun, any_of
from engraf.atn.core import ATNState
from engraf.utils.actions import make_run_np_into_atn
from engraf.pos.prepositional_phrase import PrepositionalPhrase

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(pp):

    start = ATNState("PP-START")
    after_prep = ATNState("PP-NP")
    end = ATNState("PP-END")

    # Match PREPOSITION (e.g., "on", "under")
    start.add_arc(is_preposition, lambda tok: pp.apply_preposition(tok), after_prep)

    # Match NP (subnetwork)
    action = make_run_np_into_atn(ts)
    after_prep.add_arc(any_of(is_determiner, is_pronoun), action, end)

    # Match VECTOR (e.g., "[x, y, z]")
    after_prep.add_arc(is_vector, apply_vector, end)

    return start, end

