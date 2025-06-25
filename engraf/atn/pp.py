from engraf.lexer.token_stream import TokenStream
from engraf.lexer.pos_tags import POS_TAGS
from engraf.atn.core import ATNState

# --- Build the Prepositional Phrase ATN ---

def build_pp_atn(ts: TokenStream):
    start = ATNState("PP-START")
    after_prep = ATNState("PP-NP")
    end = ATNState("PP-END")

    # Match PREPOSITION (e.g., "on", "under")
    start.add_arc(
        lambda tok: POS_TAGS.get(tok) == 'PREP',
        lambda ctx, tok: ctx.update({'prep': tok}),
        after_prep
    )

    # Match NP (subnetwork)
    after_prep.add_arc(
        lambda tok: POS_TAGS.get(tok) in ('DET', 'ADJ', 'NOUN'),
        None,  # patched with PP runner
        end
    )
    return start, end

