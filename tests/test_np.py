import numpy as np
from engraf.atn.np import build_np_atn
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace, vector_from_features

 
def test_simple_np():
    result = run_np(TokenStream(tokenize("the red cube")))
   
    assert result is not None
    assert result["noun"] == "cube"
    assert isinstance(result["vector"], VectorSpace)

def test_np_with_adjectives():
    result = run_np(TokenStream(tokenize("a large blue sphere")))
   
    assert result is not None
    assert result["noun"] == "sphere"
    assert isinstance(result["vector"], VectorSpace)

def test_np_failure():
    result = run_np(TokenStream(tokenize("draw red cube")))

    assert result is None

def test_np_with_preposition():
    result = run_np(TokenStream(tokenize("a large blue sphere over the green cube")))
   
    assert result is not None
    assert result["noun"] == "sphere"
    assert isinstance(result["vector"], VectorSpace)
    assert result["modifiers"] is not None
    assert len(result["modifiers"]) == 1
    assert result["modifiers"][0]["prep"] == "over"
    assert result["modifiers"][0]["object"] == "cube"

