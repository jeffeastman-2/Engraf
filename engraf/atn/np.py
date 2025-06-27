import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features, VectorSpace
from engraf.atn.core import ATNState,noop
from engraf.utils.actions import make_run_pp_into_ctx

# --- Helper predicate functions ---
def is_determiner(tok): 
    return tok.isa("det")   

def is_adjective(tok):
    return tok.isa("adj")

def is_adverb(tok):
    return tok.isa("adv")

def is_noun(tok):
    return tok.isa("noun")  

def is_preposition(tok):
    return tok.isa("prep")

def is_none(tok):
    return tok is None

# --- Action functions ---
def apply_determiner(ctx, tok):
    # Initialize the vector space for the noun phrase
    ctx["vector"] = ctx.setdefault("vector", VectorSpace())
    # Add the determiner vector to the context
    ctx["vector"] += tok

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
    noun = ATNState("NP-NOUN")
    pp = ATNState("NP-PP")
    end = ATNState("NP-END")

    start.add_arc(is_determiner, apply_determiner, det)

    # ADJ → ADJ / NOUN
    det.add_arc(is_adverb, apply_adverb, adj)
    det.add_arc(is_adjective, apply_adjective, adj)
    adj.add_arc(is_adjective, apply_adjective, adj)

    # ADJ or DET → NOUN
    for state in [det, adj]:
        state.add_arc(is_noun, apply_noun, noun)

    # NOUN → END (simple NP)
    noun.add_arc(is_none, noop, end)

    # NOUN → PP (subnetwork)
    action = make_run_pp_into_ctx(ts)
    noun.add_arc(is_preposition, action, pp)

    # PP → END
    pp.add_arc(is_none, noop, end)

    return start, end


