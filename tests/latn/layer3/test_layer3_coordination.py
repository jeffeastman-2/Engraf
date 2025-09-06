from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase



def test_coordinated_pp():
    """Test Layer 3 PP tokenization with coordinated prepositional phrases """
    executor = LATNLayerExecutor()

    # Test coordinated PPs: "above the red box and below the blue circle and behind the octahedron"
    result = executor.execute_layer3('above the red box and below the blue circle and behind the octahedron',report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    main_hyp = result.hypotheses[1]
    # Should have exactly 3 PP tokens (one for each prepositional phrase)
    assert len(main_hyp.tokens) == 5, f"Should have exactly 5 tokens, got {len(main_hyp.tokens)}"
    main_pp = main_hyp.tokens[0].word
    assert main_pp.startswith("PP("), "First token should be a PP"
    conj_pp = main_hyp.tokens[1].word
    assert conj_pp.startswith("and"), "Second token should be a conjunction"
    other_pp = main_hyp.tokens[2].word
    assert other_pp.startswith("PP("), "Third token should be a PP"
    conj_pp = main_hyp.tokens[3].word
    assert conj_pp.startswith("and"), "Fourth token should be a conjunction"
    other_pp = main_hyp.tokens[4].word
    assert other_pp.startswith("PP("), "Fifth token should be a PP"

    conj_hyp = result.hypotheses[0]    
    # Should have exactly 2 NP tokens (one for each noun phrase)
    assert len(conj_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(conj_hyp.tokens)}"
    conj_np = conj_hyp.tokens[0].word
    assert conj_np.startswith("CONJ-PP"), "First token should be a conjunction PP"
    conj = conj_hyp.tokens[0]
    assert conj is not None
    original_pp = conj._original_pp
    assert isinstance(original_pp, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_pp)}"
    assert isinstance(original_pp.right, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.right)}"
    assert isinstance(original_pp.left, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_pp.left)}"
    assert isinstance(original_pp.left.left, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.left.left)}"
    assert isinstance(original_pp.left.right, PrepositionalPhrase), f"Expected PrepositionalPhrase, got {type(original_pp.left.right)}"

def test_coordinated_pp_with_nps():

    executor = LATNLayerExecutor()

    # Test coordinated PPs: "color the red cube above the table and the blue sphere below the cylinder green"
    # Hypothesis-1: "color the red cube (above the table) and (the blue sphere) below the cylinder green"
    # Hypothesis-2: "color the red cube (above the table and the blue sphere) (below the cylinder) green"
    result = executor.execute_layer3('color the red cube above the table and the blue sphere below the cylinder green',report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 5, f"Should have exactly 5 tokens, got {len(hyp.tokens)}"
    hyp = result.hypotheses[1]
    assert len(hyp.tokens) == 7, f"Should have exactly 7 tokens, got {len(hyp.tokens)}"

