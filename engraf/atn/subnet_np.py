# engraf/atn/subnet_np.py
from engraf.atn.np import build_np_atn
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn
from engraf.pos.noun_phrase import NounPhrase


def run_np(tokens):
    ts = TokenStream(tokens)
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    return run_atn(np_start, np_end, ts, np)
