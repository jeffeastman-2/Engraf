from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, run_atn, noop
from engraf.lexer.vector_space import is_quoted, is_tobe, is_verb, is_adjective_or_adverb
from engraf.utils.actions import make_run_np_into_ctx
from engraf.utils.actions import make_run_vp_into_ctx

def store_definition_word(ctx, tok):
    ctx["definition_word"] = tok.word
    ctx["vector"] = tok  # Store the vector representation of the quoted word
    return ctx

def store_tobe_word(ctx, tok):
    ctx["tobe_word"] = tok.word
    ctx["vector"] = tok  # Store the vector representation of the 'is' word
    return ctx

def build_sentence_atn(ts: TokenStream):
    start = ATNState("SENTENCE-START")
    after_np = ATNState("SENTENCE-VP")
    tobe = ATNState("SENTENCE-TOBE")
    adjective_phrase = ATNState("SENTENCE-ADJECTIVE-PHRASE")
    after_tobe = ATNState("SENTENCE-TOBE")
    end = ATNState("SENTENCE-END")

    # Optional subject NP (ignored in output, just consumes NP if found)
    start.add_arc(any_of(is_determiner, is_pronoun), make_run_np_into_ctx(ts), after_np)
    start.add_arc(is_quoted, store_definition_word, tobe)   # Recognize: 'quoted' → 'is' → adjective/adverb phrase
    start.add_arc(is_verb, noop,  after_np)    # If subject omitted, proceed directly to VP

   # Recognize: 'quoted' → 'is' → adjective/adverb phrase
    tobe.add_arc(is_tobe, store_tobe_word, adjective_phrase)

    # Let NP handle adjective phrase (you may reuse np_atn or a subset)
    adjective_phrase.add_arc(is_adjective_or_adverb, make_run_np_into_ctx(ts), adjective_phrase)
    adjective_phrase.add_arc(is_none, noop, end)

    # Main verb phrase
    after_np.add_arc(is_verb, make_run_vp_into_ctx(ts), end)
    after_np.add_arc(is_tobe, store_tobe_word, adjective_phrase)

    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)

    return start, end

def run_sentence(tokens):
    ts = TokenStream(tokens)
    print("TokenStream initialized with tokens:", tokens)
    start, end = build_sentence_atn(ts)
    ctx = {}
    result = run_atn(start, end, ts, ctx)
    if result is None:
        print("No valid parse found for the sentence.")
        return None
    print("Parsed sentence context:", ctx)
    return result
