# engraf/atn/subnet_pp.py
from engraf.atn.pp import build_pp_atn
from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn
from engraf.pos.prepositional_phrase import PrepositionalPhrase

def run_pp(tokens):
    ts = TokenStream(tokens)
    pp = PrepositionalPhrase()
    pp_start, pp_end = build_pp_atn(pp,ts)
    result = run_atn(pp_start, pp_end, ts, pp)
    print(f"DEBUG: noun_phrase inside result = {result.noun_phrase}")
    print(f"DEBUG: is pp the same object as result? {pp is result}")
    return result
