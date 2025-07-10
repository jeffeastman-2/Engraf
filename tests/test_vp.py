import numpy as np
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_vp import run_vp
from engraf.lexer.vector_space import VectorSpace
from pprint import pprint

def test_simple_vp():
    vp = run_vp(TokenStream(tokenize("draw a red cube")))
    #print(result)  # For debugging purposes, can be removed later
    assert vp is not None
    assert vp.verb == "draw"
    assert vp.noun_phrase.noun == "cube"
    assert isinstance(vp.noun_phrase.vector, VectorSpace)

def test_vp_with_nested_pp():
    vp = run_vp(TokenStream(tokenize("draw a cube over the sphere")))
    assert vp is not None

    # Top-level checks
    assert vp.verb == "draw"
    assert isinstance(vp.vector, VectorSpace)

    # Noun phrase checks
    noun_phrase = vp.noun_phrase
    assert noun_phrase is not None
    assert noun_phrase.noun == "cube"
    assert isinstance(noun_phrase.vector, VectorSpace)

    # Safely check prepositions
    preps = noun_phrase.preps
    assert preps is not None, "Expected noun_phrase to have prepositions"
    assert isinstance(preps, list)
    assert len(preps) == 1

    prep = preps[0]
    assert prep.preposition == "over"
    assert isinstance(prep.vector, VectorSpace)

    inner_np = prep.noun_phrase
    assert inner_np.noun == "sphere"
    assert isinstance(inner_np.vector, VectorSpace)
    assert len(inner_np.preps) == 0


def test_vp_with_explicit_vector():
    tokens = TokenStream(tokenize("draw a tall blue cube at [3.0, 4.0, 5.0]"))
    result = run_vp(tokens)
    pprint(result)  # For debugging purposes, can be removed later
    assert result is not None
    assert result.verb == "draw"

    np = result.noun_phrase
    assert np.noun == "cube"
    assert isinstance(np.vector, VectorSpace)
    
    pp = np.preps[0]
    assert pp.preposition == "at"
    pp_np = pp.noun_phrase
    assert pp_np.vector["locX"] == 3.0
    assert pp_np.vector["locY"] == 4.0
    assert pp_np.vector["locZ"] == 5.0

    # Optional: check if the vector is untouched by the position spec
    v = np.vector
    assert v["scaleY"] == 2.5  # from "tall"
    assert v["blue"] == 1.0    # from "blue"

