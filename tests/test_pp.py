import numpy as np
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_pp import run_pp
from engraf.lexer.vector_space import VectorSpace, vector_from_features, is_preposition, is_vector, is_determiner, is_pronoun, is_adverb

def test_pp_over_red_cube():
    result = run_pp(TokenStream(tokenize("over the red cube")))
    assert result is not None
    print(f" *** Result = {result}")
    assert result.preposition == "over"
    assert result.noun_phrase is not None
    assert result.noun_phrase.noun == "cube"
    assert isinstance(result.vector, VectorSpace)


def test_pp_near_green_sphere():
    result = run_pp(TokenStream(tokenize("near the green sphere")))
    assert result is not None
    assert result.preposition == "near"
    assert result.noun_phrase.noun == "sphere"


def test_pp_under_large_blue_box():
    result = run_pp(TokenStream(tokenize("under the large blue box")))
    assert result is not None
    assert result.preposition == 'under'
    assert result.noun_phrase.noun == 'box'

def test_pp_at_vector():
    tokens = tokenize("at [3.0, 4.0, 5.0]")
    result = run_pp(TokenStream(tokens))

    assert result is not None
    assert result.preposition == 'at'
    assert result.vector is not None
    assert result.vector["locX"] == 3.0
    assert result.vector["locY"] == 4.0
    assert result.vector["locZ"] == 5.0
