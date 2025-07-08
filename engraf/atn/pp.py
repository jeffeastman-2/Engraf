from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vector_space import is_preposition, is_vector, is_determiner, is_pronoun, any_of, is_none
from engraf.atn.core import ATNState
from engraf.utils.actions import make_run_np_into_atn
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.utils.actions import make_run_np_into_atn
from engraf.utils.actions import apply_from_subnet

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(pp, ts):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-AFTER-PREP")
    after_np = ATNState("PP-NP-RESULT")
    end = ATNState("PP-END")

    # PREP
    start.add_arc(is_preposition, lambda _, tok: pp.apply_preposition(tok), after_prep)

    # NP as subnetwork
    after_np.add_arc(is_none, apply_from_subnet("noun_phrase",pp.apply_np), end)

    # Use NP object from result directly
    after_np.add_arc(is_none, lambda _, np_obj: pp.apply_np(np_obj), end)

    # match an NP
    after_prep.add_arc(is_determiner, make_run_np_into_atn(ts, fieldname="noun_phrase"), end)
    # Or: match a vector instead of NP
    after_prep.add_arc(is_vector, lambda _, tok: pp.apply_vector(tok), end)

    return start, end
