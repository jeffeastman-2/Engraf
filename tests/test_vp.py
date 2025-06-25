import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_vp import run_vp

def test_simple_vp():
    result = run_vp(TokenStream("draw a red cube".split()))
    #print(result)  # For debugging purposes, can be removed later
    assert result is not None
    assert result["verb"] == "draw"
    assert result["noun_phrase"]["noun"] == "cube"
    assert result["noun_phrase"]["vector"].shape == (6,)
    assert result["modifiers"] is None

def test_vp_with_nested_pp():
    result = run_vp(TokenStream("draw a cube over the sphere".split()))
    assert result is not None

    # Top-level checks
    assert result["verb"] == "draw"
    assert result["object"] == "cube"
    assert isinstance(result["vector"], np.ndarray)
    assert result["vector"].shape == (6,)

    # Noun phrase checks
    noun_phrase = result["noun_phrase"]
    assert isinstance(noun_phrase, dict)
    assert noun_phrase["noun"] == "cube"
    assert isinstance(noun_phrase["vector"], np.ndarray)
    assert noun_phrase["vector"].shape == (6,)

    # Safely check modifiers
    np_mods = noun_phrase.get("modifiers")
    assert np_mods is not None, "Expected noun_phrase to have modifiers"
    assert isinstance(np_mods, list)
    assert len(np_mods) == 1

    mod = np_mods[0]
    assert mod["prep"] == "over"
    assert mod["object"] == "sphere"
    assert isinstance(mod["vector"], np.ndarray)
    assert mod["vector"].shape == (6,)

    inner_np = mod["noun_phrase"]
    assert isinstance(inner_np, dict)
    assert inner_np["noun"] == "sphere"
    assert isinstance(inner_np["vector"], np.ndarray)
    assert inner_np["vector"].shape == (6,)
    assert inner_np.get("modifiers") is None

    # Optional: ensure top-level and nested modifiers are the same object
    assert result["modifiers"] == np_mods



