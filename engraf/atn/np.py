import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.pos_tags import POS_TAGS
from engraf.lexer.vector_space import VECTOR_SPACE
from engraf.atn.core import ATNState,noop
from engraf.utils.actions import diagnostic_stub


# --- Build the Noun Phrase ATN ---
def build_np_atn(ts: TokenStream):
    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adj = ATNState("NP-ADJ")
    noun = ATNState("NP-NOUN")
    pp = ATNState("NP-PP")
    end = ATNState("NP-END")

    # DET → ADJ / NOUN
    start.add_arc(lambda tok: POS_TAGS.get(tok) == 'DET', lambda ctx, tok: None, det)

    # ADJ → ADJ / NOUN
    det.add_arc(lambda tok: POS_TAGS.get(tok) == 'ADJ',
                lambda ctx, tok: ctx.setdefault('vector', np.zeros(6)).__iadd__(VECTOR_SPACE.get(tok, np.zeros(6))), adj)
    adj.add_arc(lambda tok: POS_TAGS.get(tok) == 'ADJ',
                lambda ctx, tok: ctx.setdefault('vector', np.zeros(6)).__iadd__(VECTOR_SPACE.get(tok, np.zeros(6))), adj)

    # ADJ or DET → NOUN
    for state in [det, adj]:
        state.add_arc(lambda tok: POS_TAGS.get(tok) == 'NOUN',
                      lambda ctx, tok: (
                          ctx.setdefault('vector', np.zeros(6)).__iadd__(VECTOR_SPACE.get(tok, np.zeros(6))),
                          ctx.update({'noun': tok})
                      ), noun)

    # NOUN → END (simple NP)
    noun.add_arc(lambda tok: tok is None, noop, end)

    # NOUN → PP (subnetwork)
    noun.add_arc(lambda tok: POS_TAGS.get(tok) == 'PREP', diagnostic_stub, pp)

    # PP → END
    pp.add_arc(lambda tok: tok is None, noop, end)

    return start, end


