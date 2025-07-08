from engraf.atn.core import ATNState, run_atn, noop
from engraf.atn.sentence import build_sentence_atn
from engraf.lexer.token_stream import TokenStream
from engraf.pos.sentence_phrase import SentencePhrase



def run_sentence(tokens):
    ts = TokenStream(tokens)
    sent = SentencePhrase()
    s_start, s_end = build_sentence_atn(sent, ts)
    return run_atn(s_start, s_end, ts, sent)
