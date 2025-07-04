from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vector_space import is_preposition, is_vector, is_determiner, is_pronoun, any_of
from engraf.atn.core import ATNState
from engraf.utils.actions import make_run_np_into_ctx

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(ts: TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-NP")
    end = ATNState("PP-END")

    def apply_preposition(ctx, tok):
        # Initialize the context with the preposition
        ctx['prep'] = tok.word

    def apply_vector(ctx, tok):
        # Update the context with the vector
        ctx['object'] = tok

    # Match PREPOSITION (e.g., "on", "under")
    start.add_arc(is_preposition, apply_preposition, after_prep)

    # Match NP (subnetwork)
    action = make_run_np_into_ctx(ts)
    after_prep.add_arc(any_of(is_determiner, is_pronoun), action, end)

    # Match VECTOR (e.g., "[x, y, z]")
    after_prep.add_arc(is_vector, apply_vector, end)

    return start, end

