import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.utils.predicates import any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none, is_anything, is_vector, is_conjunction, is_number
from engraf.atn.core import ATNState, noop
from engraf.utils.actions import make_run_pp_into_atn, apply_from_subnet
from engraf.pos.noun_phrase import NounPhrase

# --- Build the Noun Phrase ATN ---
def build_np_atn(np: NounPhrase, ts: TokenStream):
    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adj = ATNState("NP-ADJ")
    adj_conj = ATNState("NP-ADJ-CONJ")  # New state for handling conjunctions between adjectives
    adj_after_pronoun = ATNState("NP-ADJ-AFTER-PRONOUN")
    noun = ATNState("NP-NOUN")
    pp = ATNState("NP-PP")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, lambda _, tok: np.apply_determiner(tok), det)
    start.add_arc(is_pronoun, lambda _, tok: np.apply_pronoun(tok), adj_after_pronoun)
    start.add_arc(is_vector, lambda _, tok: np.apply_vector(tok), end)

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)
    det.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)

    adj.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)
    # Handle conjunctions between adjectives (e.g., "tall and red")
    adj.add_arc(is_conjunction, lambda _, tok: None, adj_conj)  # Consume the conjunction
    
    # After conjunction, expect another adjective (possibly with adverb modifier)
    adj_conj.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_conj)
    adj_conj.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)

    adj_after_pronoun.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_after_pronoun)
    adj_after_pronoun.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj_after_pronoun)
    # Handle prepositional phrases after pronouns
    adj_after_pronoun.add_arc(is_preposition, make_run_pp_into_atn(ts), pp)
    # End on various boundary conditions but don't consume tokens
    adj_after_pronoun.add_arc(any_of(is_verb, is_tobe, is_conjunction), noop, end)
    adj_after_pronoun.add_arc(is_none, noop, end)

    # ADJ or DET → NOUN
    for state in [det, adj, adj_conj]:
        state.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)
    # NOUN → END (when conjunction is encountered - don't consume the conjunction)
    noun.add_arc(is_conjunction, noop, end)

    # NOUN → PP (subnetwork)
    # Add the subnetwork runner on its own state transition
    action = make_run_pp_into_atn(ts)
    noun.add_arc(is_preposition, action, pp)
    noun.add_arc(any_of(is_verb, is_tobe, is_conjunction, is_adjective), noop, end)
    # REMOVED: noun.add_arc(is_anything, lambda _, tok: print(f"Unexpected token in NP: {tok}"), end)
    # Let the parser naturally end instead of consuming unexpected tokens

    pp.add_arc(is_none, apply_from_subnet("noun_phrase", np.apply_pp), end)
    pp.add_arc(any_of(is_verb, is_tobe, is_conjunction), noop, end)

    return start, end


