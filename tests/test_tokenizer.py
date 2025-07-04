import pytest
from engraf.lexer.token_stream import tokenize
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vocabulary import add_to_vocabulary, get_from_vocabulary, has_word

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
    assert_vector_has_pos(tokens[1], "det def")
    print(tokens[1])
    # Check if the number is correctly parsed as a float
    assert isinstance(tokens[1]["number"], float)
    assert tokens[1]["number"] == 2.0
    assert_vector_has_pos(tokens[2], "noun")

def test_tokenize_with_float_number():
    tokens = tokenize("draw 3.5 sphere")
    assert len(tokens) == 3
    assert_vector_has_pos(tokens[1], "det def")
    assert tokens[1]["number"] == 3.5

def test_tokenize_unknown_token_raises():
    with pytest.raises(ValueError, match="Unknown token: foozle"):
        tokenize("draw foozle")

def test_tokenize_quoted_word():
    tokens = tokenize("'sky blue' is blue and green")
    assert len(tokens) == 5
    assert_vector_has_pos(tokens[0], "unknown")
    assert tokens[0].word == "sky blue"
    assert_vector_has_pos(tokens[1], "tobe")
    assert_vector_has_pos(tokens[2], "adj")
    assert_vector_has_pos(tokens[3], "conj")
    assert_vector_has_pos(tokens[4], "adj") 
    add_to_vocabulary(tokens[0].word, tokens[0])
    print(f"Added '{tokens[0].word}' to vocabulary: {SEMANTIC_VECTOR_SPACE[tokens[0].word]}")
    assert has_word("sky blue")