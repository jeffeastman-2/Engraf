import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features, VectorSpace, any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none
from engraf.atn.core import ATNState, noop
from engraf.utils.actions import make_run_pp_into_atn
from engraf.pos.noun_phrase import NounPhrase


# --- Build the Noun Phrase ATN ---
def build_np_atn(ts: TokenStream):
    np = NounPhrase()

    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adv = ATNState("NP-ADV")
    adj = ATNState("NP-ADJ")
    adj_after_pronoun = ATNState("NP-ADJ-AFTER-PRONOUN")
    noun = ATNState("NP-NOUN")
    pp = ATNState("NP-PP")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, apply_determiner, det)
    start.add_arc(is_pronoun, apply_pronoun, adj_after_pronoun)

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, apply_adverb, det)
    det.add_arc(is_adjective, apply_adjective, adj)

    adj.add_arc(is_adjective, apply_adjective, adj)

    adj_after_pronoun.add_arc(is_adverb, apply_adverb, adj_after_pronoun)
    adj_after_pronoun.add_arc(is_adjective, apply_adjective, adj_after_pronoun)
    adj_after_pronoun.add_arc(lambda tok: True, noop, end)

    # ADJ or DET → NOUN
    for state in [det, adj]:
        state.add_arc(is_noun, apply_noun, noun)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)

    # NOUN → PP (subnetwork)
    action = make_run_pp_into_atn(ts)
    noun.add_arc(is_preposition, action, pp)

    # NOUN → VERB or ISA 
    noun.add_arc(any_of(is_verb, is_tobe), action, end)

    # PP → END
    pp.add_arc(is_none, noop, end)

    return start, end


