from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.verb_phrase import VerbPhrase



def test_coordinated_vp():
    """Test Layer 4 VP tokenization with coordinated verb phrases """
    executor = LATNLayerExecutor()

    # Test coordinated VPs: "draw a red box below the blue circle and move the cube behind the octahedron to [3,3,3]"
    result = executor.execute_layer4('draw a red box below the blue circle and move the cube behind the octahedron to [3,3,3]',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 3"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    hyp0 = result.hypotheses[0]    
    # Should have exactly 1 VP token
    assert len(hyp0.tokens) == 1, f"Should have exactly 1 token, got {len(hyp0.tokens)}"
    conj_vp = hyp0.tokens[0].word
    assert conj_vp.startswith("CONJ-VP"), "First token should be a conjunction VP"
    conj = hyp0.tokens[0]
    assert conj is not None
    original_vp = conj._original_vp
    assert isinstance(original_vp, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_vp)}"
    assert isinstance(original_vp.right, VerbPhrase), f"Expected VerbPhrase, got {type(original_vp.right)}"
    assert isinstance(original_vp.left, VerbPhrase), f"Expected ConjunctionPhrase, got {type(original_vp.left)}"

    hyp1 = result.hypotheses[1]
    # Should have exactly 2 VP tokens (one for each verb phrase)
    assert len(hyp1.tokens) == 3, f"Should have exactly 5 tokens, got {len(hyp1.tokens)}"
    main_vp = hyp1.tokens[0].word
    assert main_vp.startswith("VP("), "First token should be a VP"
    conj_vp = hyp1.tokens[1].word
    assert conj_vp.startswith("and"), "Second token should be 'and'"
    other_vp = hyp1.tokens[2].word
    assert other_vp.startswith("VP("), "Third token should be a VP"


def test_coordinated_vp_with_nps():

    executor = LATNLayerExecutor()

    # Test coordinated VPs: "draw a red cube above the table and color the blue sphere below the cylinder green'"
    result = executor.execute_layer4('draw a red cube above the table and color the blue sphere below the cylinder green',report=True)

    assert result.success, "Failed to tokenize coordinated VPs in Layer 4"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
    hyp = result.hypotheses[1]
    assert len(hyp.tokens) == 3, f"Should have exactly 3 tokens, got {len(hyp.tokens)}"

