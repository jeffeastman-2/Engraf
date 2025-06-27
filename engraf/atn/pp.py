from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState
from engraf.utils.actions import make_run_np_into_ctx
from engraf.atn.np import is_determiner, is_prep, noop, is_pronoun

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(ts: TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-NP")
    end = ATNState("PP-END")

    def is_vector(tok):
        return tok is not None and tok.isa("vector")

    def apply_preposition(ctx, tok):
        # Initialize the context with the preposition
        ctx['prep'] = tok.word

    def apply_vector(ctx, tok):
        # Update the context with the vector
        ctx['object'] = tok

    # Match PREPOSITION (e.g., "on", "under")
    start.add_arc(is_prep, apply_preposition, after_prep)

    # Match NP (subnetwork)
    action = make_run_np_into_ctx(ts)
    after_prep.add_arc(is_determiner or is_pronoun, action, end)

    # Match VECTOR (e.g., "[x, y, z]")
    after_prep.add_arc(is_vector, apply_vector, end)

    return start, end

