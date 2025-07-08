from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vector_space import is_verb, is_tobe, is_determiner, is_pronoun, is_none, any_of
from engraf.atn.core import noop
from engraf.utils.actions import make_run_np_into_atn
from engraf.pos.verb_phrase import VerbPhrase
from engraf.utils.actions import make_run_np_into_atn, apply_from_subnet


# --- Build the Verb Phrase ATN ---
def build_vp_atn(vp: VerbPhrase, ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    after_np = ATNState("VP-AFTER-NP")
    end = ATNState("VP-END")

    # VERB
    start.add_arc(is_verb, lambda _, tok: vp.apply_verb(tok), after_verb)
    # NP (subnetwork)
    after_verb.add_arc(any_of(is_determiner, is_pronoun), make_run_np_into_atn(ts,fieldname="noun_phrase"), after_np)    # Allow final transition if stream is exhausted
    after_verb.add_arc(is_none, noop, end)

    after_np.add_arc(is_none, apply_from_subnet("noun_phrase", vp.apply_np), end)
    return start, end


