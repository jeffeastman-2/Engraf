import numpy as np
from engraf.atn.np import build_np_atn
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet import run_np
 
def test_simple_np():
    result = run_np(TokenStream("the red cube".split()))
   
    assert result is not None
    assert result["noun"] == "cube"
    assert np.allclose(result["vector"], np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

def test_np_with_adjectives():
    result = run_np(TokenStream("a large blue sphere".split()))
   
    assert result is not None
    assert result["noun"] == "sphere"
    assert result["vector"].shape == (6,)

def test_np_failure():
    result = run_np(TokenStream("draw red cube".split()))

    assert result is None
