from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import ATNState, noop
from engraf.utils.predicates import is_quoted, is_tobe, is_verb, any_of, is_np_head, \
    is_none, is_adverb, is_adjective, is_conjunction, is_anything_no_consume
from engraf.utils.actions import make_run_np_into_atn, make_run_vp_into_atn, make_run_vp_into_conjunction
from engraf.pos.sentence_phrase import SentencePhrase

def build_sentence_atn(sent: SentencePhrase, ts: TokenStream):
    start = ATNState("SENTENCE-START")
    after_np = ATNState("SENTENCE-AFTER-NP")
    subject_conj = ATNState("SENTENCE-SUBJECT-CONJ")  # New state for subject conjunctions
    predicate = ATNState("SENTENCE-PREDICATE")
    after_predicate = ATNState("SENTENCE-AFTER_PREDICATE")
    predicate_conj = ATNState("SENTENCE-PREDICATE-CONJ")  # New state for predicate conjunctions
    adjective_phrase = ATNState("SENTENCE-ADJECTIVE-PHRASE")
    adjective = ATNState("SENTENCE-ADJECTIVE")
    adverb = ATNState("SENTENCE-ADVERB")
    conj = ATNState("SENTENCE-CONJ")  # e.g., "and transparent"
    end = ATNState("SENTENCE-END")

    # Optional subject NP 
    start.add_arc(is_np_head, make_run_np_into_atn(ts, fieldname="subject"), after_np)
    start.add_arc(is_quoted, lambda _, tok: sent.store_definition_word(tok), predicate) 
    start.add_arc(any_of(is_verb, is_tobe), noop,  predicate) 

    # NEW: Handle subject conjunctions
    after_np.add_arc(is_conjunction, lambda _, tok: sent.apply_subject_conjunction(tok), subject_conj)
    after_np.add_arc(is_anything_no_consume, lambda _, __: sent.apply_subject(sent.subject), predicate)

    # NEW: After subject conjunction, expect another noun phrase
    subject_conj.add_arc(is_np_head, make_run_np_into_atn(ts, fieldname="subject"), after_np)

    predicate.add_arc(is_verb, make_run_vp_into_atn(ts, fieldname="predicate"), after_predicate)
    predicate.add_arc(is_tobe, lambda _, tok: sent.apply_tobe(tok), adjective_phrase)

    # NEW: Handle predicate conjunctions
    after_predicate.add_arc(is_conjunction, lambda _, tok: sent.apply_predicate_conjunction(tok), predicate_conj)
    after_predicate.add_arc(is_none, lambda _, __: sent.apply_predicate(sent.predicate), end)

    # NEW: After conjunction, expect another verb phrase
    predicate_conj.add_arc(is_verb, make_run_vp_into_conjunction(ts), after_predicate)

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
