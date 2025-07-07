from engraf.atn.core import ATNState, run_atn, noop
from engraf.atn.sentence import build_sentence_atn
from engraf.lexer.token_stream import TokenStream



def run_sentence(tokens):
    ts = TokenStream(tokens)
    s_start, s_end = build_sentence_atn(ts)
    return run_atn(s_start, s_end, ts)
