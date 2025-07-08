import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features, VectorSpace, any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none
from engraf.atn.core import ATNState, noop
from engraf.utils.actions import make_run_pp_into_atn, apply_from_subnet
from engraf.pos.noun_phrase import NounPhrase

# --- Build the Noun Phrase ATN ---
def build_np_atn(np: NounPhrase, ts: TokenStream):
    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adv = ATNState("NP-ADV")
    adj = ATNState("NP-ADJ")
    adj_after_pronoun = ATNState("NP-ADJ-AFTER-PRONOUN")
    noun = ATNState("NP-NOUN")
    pp = ATNState("NP-PP")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, lambda _, tok: np.apply_determiner(tok), det)
    start.add_arc(is_pronoun, lambda _, tok: np.apply_pronoun(tok), adj_after_pronoun)

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)
    det.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)

    adj.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)

    adj_after_pronoun.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_after_pronoun)
    adj_after_pronoun.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj_after_pronoun)
    adj_after_pronoun.add_arc(lambda _, tok: True, noop, end)

    # ADJ or DET → NOUN
    for state in [det, adj]:
        state.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)

    # NOUN → PP (subnetwork)
    # Add the subnetwork runner on its own state transition
    action = make_run_pp_into_atn(ts)
    noun.add_arc(is_preposition, action, pp)
    noun.add_arc(any_of(is_verb, is_tobe), action, pp)

    pp.add_arc(is_none, apply_from_subnet("noun_phrase", np.apply_pp), end)

    return start, end


