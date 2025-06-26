from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import noop
from engraf.utils.actions import make_run_np_into_ctx


# --- Build the Verb Phrase ATN ---
def build_vp_atn(ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    end = ATNState("VP-END")
    # VERB
    start.add_arc(
        lambda tok: tok.isa("verb"),
        lambda ctx, tok: ctx.update({'verb': tok.word}),
        after_verb
    )
    # NP (subnetwork)
    action = make_run_np_into_ctx(ts)
    after_verb.add_arc(
        lambda tok: tok.isa("det"),  # if determiner found, assume NP starts here
        action,  
        end
    )
    # Allow final transition if stream is exhausted
    after_verb.add_arc(
        lambda tok: tok is None,
        noop,
        end
    )
    return start, end


