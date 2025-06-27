from engraf.lexer.token_stream import TokenStream
from engraf.lexer.pos_tags import POS_TAGS
from . import ATNState, run_atn, noop
from .np import make_run_np_into_ctx, is_determiner, is_none
from .vp import make_run_vp_into_ctx, is_verb   

def build_sentence_atn(ts: TokenStream):
    start = ATNState("SENTENCE-START")
    after_np = ATNState("SENTENCE-VP")
    end = ATNState("SENTENCE-END")

    # Optional subject NP (ignored in output, just consumes NP if found)
    start.add_arc(isDeterminer, make_run_np_into_ctx(ts), after_np )

    # If subject omitted, proceed directly to VP
    start.add_arc(is_verb, noop,  after_np)

    # Main verb phrase
    after_np.add_arc(is_verb, make_run_vp_into_ctx(ts), end)

    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)

    return start, end

def run_sentence(tokens):
    ts = TokenStream(tokens)
    print("TokenStream initialized with tokens:", tokens)
    start, end = build_sentence_atn(ts)
    ctx = {}
    return run_atn(start, end, ts, ctx)
