from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, run_atn, noop
from engraf.lexer.vector_space import is_quoted, is_tobe, is_verb, is_adjective_or_adverb, \
    any_of, is_determiner, is_pronoun, is_none, is_adverb, is_adjective, is_conjunction
from engraf.utils.actions import make_run_np_into_atn
from engraf.utils.actions import make_run_vp_into_atn
from engraf.pos.sentence_phrase import SentencePhrase

def build_sentence_atn(sent: SentencePhrase, ts: TokenStream):
    start = ATNState("SENTENCE-START")
    after_np = ATNState("SENTENCE-AFTER-NP")
    after_no_np = ATNState("SENTENCE-AFTER-NO-NP")
    predicate = ATNState("SENTENCE-PREDICATE")
    after_predicate = ATNState("SENTENCE-AFTER_PREDICATE")
    # TBD:
    adjective_phrase = ATNState("SENTENCE-ADJECTIVE-PHRASE")
    adjective = ATNState("SENTENCE-ADJECTIVE")
    adverb = ATNState("SENTENCE-ADVERB")
    conj = ATNState("SENTENCE-CONJ")  # e.g., "and transparent"
    end = ATNState("SENTENCE-END")

    # Optional subject NP 
    start.add_arc(any_of(is_determiner, is_pronoun), make_run_np_into_atn(ts, fieldname="subject"), after_np)
    start.add_arc(is_quoted, lambda _, tok: sent.store_definition_word(tok), predicate) 
    start.add_arc(is_verb, noop,  predicate) 

    after_np.add_arc(is_none, lambda _, np_obj: sent.apply_subject(np_obj), predicate)

    predicate.add_arc(is_verb, make_run_vp_into_atn(ts, fieldname="predicate"), after_predicate)

    after_predicate.add_arc(is_none, lambda _, vp_obj: sent.apply_predicate(vp_obj), end)


#TBD
    # Start by accepting an adverb (optional) or adjective
    adjective_phrase.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adjective_phrase.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # Allow chain of adverbs before adjectives
    adverb.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adverb.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # After adjective, allow conjunction for chained descriptions (e.g. "and rough")
    adjective.add_arc(is_conjunction, noop, conj)
    conj.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # End of adjective phrase
    adjective.add_arc(is_none, noop, end)

    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)

    return start, end
