"""Unit test for the extracted Layer6Structure.

Verifies the move out of TokenizationHypothesis was mechanically correct:
add_phrase emits bracket pairs with vectors+refs, wrap_with_phrase nests,
to_string renders the structural sequence. Self-contained (no torch).
"""

import numpy as np

from latn.lexer.vector_space import VECTOR_LENGTH
from engraf.llm_layer6.structure import Layer6Structure


class _FakeSceneObj:
    def __init__(self, oid):
        self.object_id = oid


def test_initial_state_is_empty():
    l6 = Layer6Structure()
    assert l6.tokens == []
    assert l6.vectors == []
    assert l6.scene_refs == []


def test_add_phrase_emits_bracket_pair_with_vector_and_ref():
    l6 = Layer6Structure()
    phrase_vec = np.ones(VECTOR_LENGTH)
    obj = _FakeSceneObj("sphere_1")

    l6.add_phrase("NP", phrase_vec, scene_object=obj)

    assert l6.tokens == ["[NP", "]NP"]
    assert l6.scene_refs == [None, obj]                   # ref on the CLOSE
    assert len(l6.vectors) == 2
    # Close vector carries the full phrase semantics (a copy, not the same object)
    assert np.array_equal(l6.vectors[1], phrase_vec)
    assert l6.vectors[1] is not phrase_vec
    # Open marker vector has NP dim set to +1, rest zero
    assert np.count_nonzero(l6.vectors[0]) == 1


def test_wrap_with_phrase_nests_around_existing_brackets():
    l6 = Layer6Structure()
    np_vec = np.zeros(VECTOR_LENGTH)
    l6.add_phrase("NP", np_vec)
    # State: ["[NP", "]NP"]

    pp_vec = np.zeros(VECTOR_LENGTH)
    l6.wrap_with_phrase(0, 1, "PP", pp_vec)

    assert l6.tokens == ["[PP", "[NP", "]NP", "]PP"]
    assert l6.scene_refs[0] is None       # opening PP
    assert l6.scene_refs[-1] is None      # closing PP (no scene obj given)


def test_to_string_renders_with_object_ids_on_grounded_brackets():
    l6 = Layer6Structure()
    l6.add_phrase("NP", np.zeros(VECTOR_LENGTH), scene_object=_FakeSceneObj("cube_1"))

    out = l6.to_string()
    # opening [NP has no ref; closing ]NP carries the grounded id
    assert out == "[NP ]NP<cube_1>"
