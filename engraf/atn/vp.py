from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.pos_tags import POS_TAGS
from engraf.atn.core import noop


# --- Build the Verb Phrase ATN ---
def build_vp_atn(ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    end = ATNState("VP-END")
    # VERB
    start.add_arc(
        lambda tok: POS_TAGS.get(tok) == 'VERB',
        lambda ctx, tok: ctx.update({'verb': tok}),
        after_verb
    )
    # NP (subnetwork)
    after_verb.add_arc(
        lambda tok: POS_TAGS.get(tok) in ('DET', 'ADJ', 'NOUN'),
        None,  
        end
    )
    # Allow final transition if stream is exhausted
    after_verb.add_arc(
        lambda tok: tok is None,
        noop,
        end
    )
    return start, end


