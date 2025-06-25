import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet import run_pp
from engraf.atn.pp import build_pp_atn


def test_pp_over_red_cube():
    result = run_pp(TokenStream("over the red cube".split()))
    assert result is not None
    assert result['prep'] == 'over'
    assert result['object'] == 'cube'
    assert isinstance(result['vector'], np.ndarray)
    assert result['noun_phrase']['noun'] == 'cube'


def test_pp_near_green_sphere():
    result = run_pp(TokenStream("near the green sphere".split()))
    assert result is not None
    assert result['prep'] == 'near'
    assert result['object'] == 'sphere'
    assert result['noun_phrase']['noun'] == 'sphere'


def test_pp_under_large_blue_box():
    result = run_pp(TokenStream("under the large blue box".split()))
    assert result is not None
    assert result['prep'] == 'under'
    assert result['object'] == 'box'
    assert result['noun_phrase']['noun'] == 'box'
