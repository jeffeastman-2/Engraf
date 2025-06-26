from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState
from engraf.utils.actions import make_run_np_into_ctx

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(ts: TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-NP")
    end = ATNState("PP-END")

    # Match PREPOSITION (e.g., "on", "under")
    start.add_arc(
        lambda tok: tok.isa("prep"),
        lambda ctx, tok: ctx.update({'prep': tok.word}),
        after_prep
    )

    # Match NP (subnetwork)
    action = make_run_np_into_ctx(ts)
    after_prep.add_arc(
        lambda tok: tok.isa("det"),
        action,  
        end
    )

    # Match VECTOR (e.g., "[x, y, z]")
    after_prep.add_arc(
        lambda tok: tok.isa("vector"),
        lambda ctx, tok: ctx.update({"object": tok}),
        end
    )

    return start, end

