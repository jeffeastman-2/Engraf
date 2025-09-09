from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, noop
from engraf.utils.predicates import is_quoted, is_tobe, is_verb, any_of, is_conjunction, \
    is_none, is_adverb, is_adjective, is_np_token, is_vp_token
from engraf.pos.sentence_phrase import SentencePhrase

def build_sentence_atn(sent: SentencePhrase, ts: TokenStream):
    start = ATNState("SENTENCE-START")
    after_np = ATNState("SENTENCE-AFTER-NP")
    predicate = ATNState("SENTENCE-PREDICATE")
    after_predicate = ATNState("SENTENCE-AFTER_PREDICATE")
    adjective_phrase = ATNState("SENTENCE-ADJECTIVE-PHRASE")
    adjective = ATNState("SENTENCE-ADJECTIVE")
    adverb = ATNState("SENTENCE-ADVERB")
    conj = ATNState("SENTENCE-CONJ")  # e.g., "and transparent"
    end = ATNState("SENTENCE-END")

    # Optional subject NP at start
    start.add_arc(is_np_token, lambda _, tok: sent.apply_subject_token(tok), after_np)
    start.add_arc(is_quoted, lambda _, tok: sent.store_definition_word(tok), predicate) 
    start.add_arc(any_of(is_verb, is_tobe), noop,  predicate) 

    predicate.add_arc(is_tobe, lambda _, tok: sent.apply_tobe(tok), adjective_phrase)
    predicate.add_arc(is_vp_token, lambda _, tok: sent.apply_predicate_token(tok), end)

    # NEW: Handle predicate conjunctions

    # Start by accepting an adverb (optional) or adjective
    adjective_phrase.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adjective_phrase.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # Allow chain of adverbs before adjectives
    adverb.add_arc(is_adverb, lambda _, tok: sent.apply_adverb(tok), adverb)
    adverb.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # After adjective, allow conjunction for chained descriptions (e.g. "and rough")
    adjective.add_arc(is_conjunction, lambda _, tok: None, conj)  # Consume the conjunction
    conj.add_arc(is_adjective, lambda _, tok: sent.apply_adjective(tok), adjective)

    # End of adjective phrase
    adjective.add_arc(is_none, noop, end)

    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)

    return start, end
