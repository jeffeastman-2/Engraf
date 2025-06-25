import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.pos_tags import POS_TAGS
from engraf.lexer.vector_space import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features
from engraf.atn.core import ATNState,noop
from engraf.utils.actions import make_run_pp_into_ctx


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
                lambda ctx, tok: ctx.setdefault('vector', 
                vector_from_features()).__iadd__(SEMANTIC_VECTOR_SPACE.get(tok, vector_from_features())), adj)
    adj.add_arc(lambda tok: POS_TAGS.get(tok) == 'ADJ',
                lambda ctx, tok: ctx.setdefault('vector', 
                vector_from_features()).__iadd__(SEMANTIC_VECTOR_SPACE.get(tok, vector_from_features())), adj)

    # ADJ or DET → NOUN
    for state in [det, adj]:
        state.add_arc(lambda tok: POS_TAGS.get(tok) == 'NOUN',
                      lambda ctx, tok: (
                          ctx.setdefault('vector', 
                          vector_from_features()).__iadd__(SEMANTIC_VECTOR_SPACE.get(tok, vector_from_features())),
                          ctx.update({'noun': tok})
                      ), noun)

    # NOUN → END (simple NP)
    noun.add_arc(lambda tok: tok is None, noop, end)

    # NOUN → PP (subnetwork)
    action = make_run_pp_into_ctx(ts)
    noun.add_arc(lambda tok: POS_TAGS.get(tok) == 'PREP', action, pp)

    # PP → END
    pp.add_arc(lambda tok: tok is None, noop, end)

    return start, end


