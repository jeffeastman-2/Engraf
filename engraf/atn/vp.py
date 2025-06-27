from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import noop
from engraf.utils.actions import make_run_np_into_ctx
from engraf.atn.np import is_determiner, is_none, is_pronoun

def is_verb(tok):
    return tok is not None and tok.isa("verb")

def verb_action(ctx, tok):
    ctx['verb'] = tok.word

def any_of(*predicates):
    return lambda tok: any(pred(tok) for pred in predicates)

# --- Build the Verb Phrase ATN ---
def build_vp_atn(ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    end = ATNState("VP-END")

    # VERB
    start.add_arc(is_verb, verb_action, after_verb)
    # NP (subnetwork)
    action = make_run_np_into_ctx(ts)

    after_verb.add_arc(any_of(is_determiner, is_pronoun), action, end)    # Allow final transition if stream is exhausted
    after_verb.add_arc(is_none, noop, end)
    return start, end


