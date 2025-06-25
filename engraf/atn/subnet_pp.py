# engraf/atn/subnet_pp.py
from engraf.atn.pp import build_pp_atn
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn

def run_pp(tokens):
    ts = TokenStream(tokens)
    ctx = {}
    pp_start, pp_end = build_pp_atn(ts)
    return run_atn(pp_start, pp_end, ts, ctx)
