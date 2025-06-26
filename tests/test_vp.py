import numpy as np
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_vp import run_vp
from engraf.lexer.vector_space import VectorSpace, vector_from_features
from pprint import pprint

def test_simple_vp():
    result = run_vp(TokenStream(tokenize("draw a red cube")))
    #print(result)  # For debugging purposes, can be removed later
    assert result is not None
    assert result["verb"] == "draw"
    assert result["noun_phrase"]["noun"] == "cube"
    assert isinstance(result["noun_phrase"]["vector"], VectorSpace)
    assert result["modifiers"] is None

def test_vp_with_nested_pp():
    result = run_vp(TokenStream(tokenize("draw a cube over the sphere")))
    assert result is not None

    # Top-level checks
    assert result["verb"] == "draw"
    assert result["object"] == "cube"
    assert isinstance(result["vector"], VectorSpace)

    # Noun phrase checks
    noun_phrase = result["noun_phrase"]
    assert isinstance(noun_phrase, dict)
    assert noun_phrase["noun"] == "cube"
    assert isinstance(noun_phrase["vector"], VectorSpace)

    # Safely check modifiers
    np_mods = noun_phrase.get("modifiers")
    assert np_mods is not None, "Expected noun_phrase to have modifiers"
    assert isinstance(np_mods, list)
    assert len(np_mods) == 1

    mod = np_mods[0]
    assert mod["prep"] == "over"
    assert mod["object"] == "sphere"
    assert isinstance(mod["vector"], VectorSpace)

    inner_np = mod["noun_phrase"]
    assert isinstance(inner_np, dict)
    assert inner_np["noun"] == "sphere"
    assert isinstance(inner_np["vector"], VectorSpace)
    assert inner_np.get("modifiers") is None

    # Optional: ensure top-level and nested modifiers are the same object
    assert result["modifiers"] == np_mods

def test_vp_with_explicit_vector():
    tokens = TokenStream(tokenize(tokenize("draw a tall blue cube at [3.0, 4.0, 5.0]")))
    result = run_vp(tokens)
    pprint(result)  # For debugging purposes, can be removed later
    assert result is not None
    assert result["verb"] == "draw"
    assert result["object"] == "cube"

    np_data = result["noun_phrase"]
    assert np_data["noun"] == "cube"
    assert isinstance(np_data["vector"], VectorSpace)
    
    mod = np_data["modifiers"][0]
    assert mod["prep"] == "at"
    assert mod["object"] == [3.0, 4.0, 5.0]

    # Optional: check if the vector is untouched by the position spec
    v = np_data["vector"]
    assert v["scaleY"] == 2.5  # from "tall"
    assert v["blue"] == 1.0    # from "blue"

