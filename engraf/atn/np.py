import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features, VectorSpace, any_of, is_verb, is_adverb, is_noun, is_tobe, \
    is_determiner, is_pronoun, is_adjective, is_preposition, is_none
from engraf.atn.core import ATNState, noop
from engraf.utils.actions import make_run_pp_into_ctx


# --- Action functions ---
def apply_determiner(ctx, tok):
    # Initialize the vector space for the noun phrase
    ctx["vector"] = ctx.setdefault("vector", VectorSpace())
    # Add the determiner vector to the context
    ctx["vector"] += tok

def apply_pronoun(ctx, tok):
    ctx["vector"] = ctx.setdefault("vector", VectorSpace())
    ctx["pronoun"] = tok.word
    ctx["object"] = tok.word  # placeholder, real referent resolved later
    ctx["noun_phrase"] = {"pronoun": tok.word}

def apply_adverb(ctx, tok):
    # Multiply any upcoming adjective by 1.5 (or another scalar)
    ctx["scale_factor"] = ctx.get("scale_factor", 1.0) * 1.5

def apply_adjective(ctx, tok):
    vec = ctx.setdefault("vector")
    scale = ctx.pop("scale_factor", 1.0)
    vec += tok * scale

def apply_noun(ctx, tok):
    ctx.setdefault('vector', 
    vector_from_features("")).__iadd__(tok),
    ctx.update({'noun': tok.word})

# --- Build the Noun Phrase ATN ---
def build_np_atn(ts: TokenStream):
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
    action = make_run_pp_into_ctx(ts)
    noun.add_arc(is_preposition, action, pp)

    # NOUN → VERB or ISA 
    noun.add_arc(any_of(is_verb, is_tobe), action, end)

    # PP → END
    pp.add_arc(is_none, noop, end)

    return start, end


