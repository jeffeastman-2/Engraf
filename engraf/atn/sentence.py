from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, run_atn, noop
from engraf.lexer.vector_space import is_quoted, is_tobe, is_verb, is_adjective_or_adverb, \
    any_of, is_determiner, is_pronoun, is_none, is_adverb, is_adjective, is_conjunction
from engraf.utils.actions import make_run_np_into_atn
from engraf.utils.actions import make_run_vp_into_atn

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
    adjective = ATNState("SENTENCE-ADJECTIVE")
    adverb = ATNState("SENTENCE-ADVERB")
    conj = ATNState("SENTENCE-CONJ")  # e.g., "and transparent"
    after_tobe = ATNState("SENTENCE-TOBE")
    end = ATNState("SENTENCE-END")

    # Optional subject NP (ignored in output, just consumes NP if found)
    start.add_arc(any_of(is_determiner, is_pronoun), make_run_np_into_atn(ts), after_np)
    start.add_arc(is_quoted, store_definition_word, tobe)   # Recognize: 'quoted' → 'is' → adjective/adverb phrase
    start.add_arc(is_verb, noop,  after_np)    # If subject omitted, proceed directly to VP

   # Recognize: 'quoted' → 'is' → adjective/adverb phrase
    tobe.add_arc(is_tobe, store_tobe_word, adjective_phrase)

    # Start by accepting an adverb (optional) or adjective
    adjective_phrase.add_arc(is_adverb, apply_adverb, adverb)
    adjective_phrase.add_arc(is_adjective, apply_adjective, adjective)

    # Allow chain of adverbs before adjectives
    adverb.add_arc(is_adverb, apply_adverb, adverb)
    adverb.add_arc(is_adjective, apply_adjective, adjective)

    # After adjective, allow conjunction for chained descriptions (e.g. "and rough")
    adjective.add_arc(is_conjunction, noop, conj)
    conj.add_arc(is_adjective, apply_adjective, adjective)

    # End of adjective phrase
    adjective.add_arc(is_none, noop, end)

    # Main verb phrase
    after_np.add_arc(is_verb, make_run_vp_into_atn(ts), end)
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
