import pytest
from engraf.lexer.token_stream import tokenize
from engraf.lexer.vector_space import VectorSpace

def assert_vector_has_pos(vs: VectorSpace, expected_pos: str):
    assert isinstance(vs, VectorSpace)
    for pos in expected_pos.split():
        assert pytest.approx(vs[pos], abs=1e-6) == 1.0, f"Expected POS '{pos}' not found in vector: {vs}"
        

def test_tokenize_simple_words():
    tokens = tokenize("draw a red cube")
    assert len(tokens) == 4
    assert_vector_has_pos(tokens[0], "verb")
    assert_vector_has_pos(tokens[1], "det")
    assert_vector_has_pos(tokens[2], "adj")
    assert_vector_has_pos(tokens[3], "noun")

def test_tokenize_vector_literal():
    tokens = tokenize("at [3.0, 4.0, 5.0]")
    assert len(tokens) == 2
    assert_vector_has_pos(tokens[0], "prep")
    assert_vector_has_pos(tokens[1], "vector")
    assert tokens[1]["locX"] == 3.0
    assert tokens[1]["locY"] == 4.0
    assert tokens[1]["locZ"] == 5.0

def test_tokenize_number():
    tokens = tokenize("draw 2 cube")
    assert len(tokens) == 3
    assert_vector_has_pos(tokens[0], "verb")
    assert_vector_has_pos(tokens[1], "det def number")
    assert tokens[1]["number"] == 2.0
    assert_vector_has_pos(tokens[2], "noun")

def test_tokenize_with_float_number():
    tokens = tokenize("draw 3.5 sphere")
    assert len(tokens) == 3
    assert_vector_has_pos(tokens[1], "det def number")
    assert tokens[1]["number"] == 3.5

def test_tokenize_unknown_token_raises():
    with pytest.raises(ValueError, match="Unknown token: foozle"):
        tokenize("draw foozle")
