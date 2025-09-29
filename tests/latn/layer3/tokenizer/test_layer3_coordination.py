from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase



def test_coordinated_pp():
    """Test Layer 3 PP tokenization with coordinated prepositional phrases """
    executor = LATNLayerExecutor()

    # Test coordinated PPs: "above the red box and below the blue circle and behind the octahedron"
    result = executor.execute_layer3('above the red box and below the blue circle and behind the octahedron',report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 7, "Should generate 14 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 3 PP tokens (one for each prepositional phrase)
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    main_pp = main_hyp.tokens[0]
    assert main_pp.word == "CONJ-PP", "First token should be a CONJ-PP"
    parts = list(main_pp.phrase.phrases)
    assert len(parts) == 3, f"CONJ-PP should have 3 parts, got {len(parts)}"
    assert all(isinstance(part, PrepositionalPhrase) for part in parts), "All parts should be PrepositionalPhrase instances"
    assert parts[0].preposition == "above", f"First PP should be 'above', got '{parts[0].preposition}'"
    assert parts[1].preposition == "below", f"Second PP should be 'below', got '{parts[1].preposition}'"
    assert parts[2].preposition == "behind", f"Third PP should be 'behind', got '{parts[2].preposition}'"

def test_coordinated_pp_with_nps():

    executor = LATNLayerExecutor()

    result = executor.execute_layer3('the red cube above the table and the blue sphere below the cylinder',report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) == 14, "Should generate 2 hypotheses"

    hyp = result.hypotheses[0]
    assert len(hyp.tokens) == 3, f"Should have exactly 3 tokens, got {len(hyp.tokens)}"
    np = hyp.tokens[0].phrase
    str = np.printString()
    assert hyp.tokens[0].phrase.printString() == "the red cube", f"First token should be NP, got {hyp.tokens[0].word}"
    pp = hyp.tokens[1].phrase
    np = pp.noun_phrase
    assert isinstance(np, ConjunctionPhrase), f"Second token should be CONJ-NP, got {hyp.tokens[1].word}"
    parts = np.phrases
    assert len(parts) == 2, f"CONJ-NP should have 2 parts, got {len(parts)}"
    assert all(isinstance(part, NounPhrase) for part in parts), "All parts should be NounPhrase instances"
    assert parts[0].printString() == "the table", f"First NP part should be 'the table', got '{parts[0].printString()}'"
    assert parts[1].printString() == "the blue sphere", f"Second NP part should be 'the blue sphere', got '{parts[1].printString()}'"
