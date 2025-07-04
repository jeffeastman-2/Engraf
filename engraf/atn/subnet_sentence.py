from engraf.atn.core import ATNState, run_atn, noop
from engraf.utils.actions import make_run_np_into_ctx
from engraf.atn.sentence import build_sentence_atn
from engraf.lexer.token_stream import TokenStream



def run_sentence(tokens):
    ts = TokenStream(tokens)
    ctx = {}
    s_start, s_end = build_sentence_atn(ts)
    return run_atn(s_start, s_end, ts, ctx)
