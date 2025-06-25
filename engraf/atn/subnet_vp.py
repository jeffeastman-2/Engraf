# engraf/atn/subnet_pp.py
from engraf.atn.vp import build_vp_atn
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn

def run_vp(tokens):
    ts = TokenStream(tokens)
    ctx = {}
    vp_start, vp_end = build_vp_atn(ts)
    return run_atn(vp_start, vp_end, ts, ctx)
