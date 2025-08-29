import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.utils.predicates import any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none, is_vector, is_conjunction, is_number, is_unknown
from engraf.atn.core import ATNState, noop
from engraf.pos.noun_phrase import NounPhrase

# --- Build the Noun Phrase ATN ---
def build_np_atn(np: NounPhrase, ts: TokenStream):
    # Helper function to check if token is a non-comparative adjective
    def is_non_comparative_adjective(tok):
        return is_adjective(tok) and tok.scalar_projection("comp") == 0.0
    
    start = ATNState("NP-START")
    det = ATNState("NP-DET")
    adj = ATNState("NP-ADJ")
    adj_conj = ATNState("NP-ADJ-CONJ")  # New state for handling conjunctions between adjectives
    adj_after_pronoun = ATNState("NP-ADJ-AFTER-PRONOUN")
    noun = ATNState("NP-NOUN")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, lambda _, tok: np.apply_determiner(tok), det)
    start.add_arc(is_pronoun, lambda _, tok: np.apply_pronoun(tok), adj_after_pronoun)
    start.add_arc(is_vector, lambda _, tok: np.apply_vector(tok), end)
    
    # LATN Extension: Allow NPs to start with adjectives, adverbs, or nouns
    start.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)
    start.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)  # "very" -> DET state to handle "very big sphere"
    start.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)     # Allow bare nouns like "sphere"

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), det)
    det.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)
    # Terminate on unknown tokens after determiners
    det.add_arc(is_unknown, noop, end)

    adj.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)
    # Handle adverbs that modify subsequent adjectives (e.g., "small bright blue")
    adj.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj)
    # Handle conjunctions between adjectives (e.g., "tall and red") - but NOT noun phrase coordination
    def is_adjective_conjunction(tok):
        if not is_conjunction(tok):
            return False
        # Peek ahead to see if next token is an adjective (not a determiner starting new NP)
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        return next_tok and is_adjective(next_tok) and not is_determiner(next_tok)
    
    adj.add_arc(is_adjective_conjunction, lambda _, tok: None, adj_conj)  # Consume the conjunction
    
    # After conjunction, expect another adjective (possibly with adverb modifier)
    adj_conj.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_conj)
    adj_conj.add_arc(is_adjective, lambda _, tok: np.apply_adjective(tok), adj)

    adj_after_pronoun.add_arc(is_adverb, lambda _, tok: np.apply_adverb(tok), adj_after_pronoun)
    # Allow adjectives after pronouns, but NOT comparative adjectives (let VP handle those)
    adj_after_pronoun.add_arc(is_non_comparative_adjective, lambda _, tok: np.apply_adjective(tok), adj_after_pronoun)
    # Handle conjunctions between adjectives after pronouns - but only if followed by adjective
    def is_conjunction_followed_by_adjective(tok):
        if not is_conjunction(tok):
            return False
        # Peek ahead to see if next token is an adjective
        current_pos = ts.position
        next_tok = None
        if current_pos + 1 < len(ts.tokens):
            next_tok = ts.tokens[current_pos + 1]
        return next_tok and is_adjective(next_tok)
    
    adj_after_pronoun.add_arc(is_conjunction_followed_by_adjective, lambda _, tok: None, adj_conj)
    # End on various boundary conditions but don't consume tokens
    adj_after_pronoun.add_arc(any_of(is_verb, is_tobe, is_conjunction, is_adjective), noop, end)
    adj_after_pronoun.add_arc(is_none, noop, end)
    # Terminate on unknown tokens after pronouns - this ensures pronouns can end properly
    adj_after_pronoun.add_arc(is_unknown, noop, end)
    # Also end on prepositions and numbers that might follow pronouns
    adj_after_pronoun.add_arc(any_of(is_preposition, is_number), noop, end)

    # ADJ or DET → NOUN
    for state in [det, adj, adj_conj]:
        state.add_arc(is_noun, lambda _, tok: np.apply_noun(tok), noun)

    # Allow ADJ state to end on various conditions
    adj.add_arc(is_none, noop, end)
    adj.add_arc(any_of(is_verb, is_tobe, is_conjunction), noop, end)
    # Terminate on unknown tokens after adjectives
    adj.add_arc(is_unknown, noop, end)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)
    # NOUN → END (when conjunction is encountered - don't consume the conjunction)
    noun.add_arc(is_conjunction, noop, end)
    # NOUN → END (when verb, tobe, adjective, or preposition is encountered - don't consume)
    noun.add_arc(any_of(is_verb, is_tobe, is_adjective, is_preposition), noop, end)
    # NOUN → END (when determiner is encountered - don't consume, starts new NP)
    noun.add_arc(is_determiner, noop, end)
    # Terminate on unknown tokens after nouns - this is the key fix
    noun.add_arc(is_unknown, noop, end)

    return start, end


def run_np(tokens):
    """Run the NP ATN on a sequence of tokens.
    
    Args:
        tokens: List of VectorSpace tokens
        
    Returns:
        NounPhrase object if successful, None if parsing fails
    """
    from engraf.atn.core import run_atn
    
    ts = TokenStream(tokens)
    np = NounPhrase()
    np_start, np_end = build_np_atn(np, ts)
    result = run_atn(np_start, np_end, ts, np)
    return result


