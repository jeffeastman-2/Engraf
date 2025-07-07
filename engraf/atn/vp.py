from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vector_space import is_verb, is_tobe, is_determiner, is_pronoun, is_none
from engraf.atn.core import noop
from engraf.utils.actions import make_run_np_into_atn
from engraf.pos.verb_phrase import VerbPhrase


# --- Build the Verb Phrase ATN ---
def build_vp_atn(ts: TokenStream):
    vp = VerbPhrase()

    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    end = ATNState("VP-END")

    # VERB
    start.add_arc(is_verb, vp.verb_action(ts), after_verb)
    # NP (subnetwork)
    action = make_run_np_into_atn(ts)

    after_verb.add_arc(any_of(is_determiner, is_pronoun), action, end)    # Allow final transition if stream is exhausted
    after_verb.add_arc(is_none, noop, end)

    return start, end


