import numpy as np
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer_layer1 import latn_tokenize_layer1, latn_tokenize_best
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase

 
def test_simple_np():
    result = run_np(TokenStream(latn_tokenize_best("the red cube")))
   
    assert result is not None
    assert result.noun == "cube"
    assert isinstance(result.vector, VectorSpace)

def test_np_with_adjectives():
    result = run_np(TokenStream(latn_tokenize_best("a large blue sphere")))
   
    assert result is not None
    assert result.noun == "sphere"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] > 1.5  # Check if the height is scaled
    assert result.vector["blue"] > 0.5  # Check if the color is blue

def test_np_with_adverbs():
    result = run_np(TokenStream(latn_tokenize_best("a very tall red cylinder")))
   
    assert result is not None
    print("Final vector:", result.vector)
    assert result.noun == "cylinder"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] == 2.25  # "very tall" = 1.5 * 1.5 = 2.25
    assert result.vector["red"] > 0.5  # Check if the color is red

def test_np_failure():
    result = run_np(TokenStream(latn_tokenize_best("draw red cube")))
    assert result is None

def test_np_with_pronoun():
    np = run_np(TokenStream(latn_tokenize_best("it")))
    assert np is not None
    assert np.pronoun == "it"

def test_np_with_vector():
    np = run_np(TokenStream(latn_tokenize_best("[5,6,7]")))
    assert np is not None
    assert np.noun == "vector"
    assert np.vector is not None
    v = np.vector
    assert v["locX"] == 5.0
    assert v["locY"] == 6.0
    assert v["locZ"] == 7.0


# LATN Integration Tests

def test_latn_tokenizer_with_simple_np():
    """Test that LATN tokenizer works with simple NP and provides multiple hypotheses."""
    sentence = "the red cube"
    hypotheses = latn_tokenize_layer1(sentence)
    
    # Should have at least one hypothesis
    assert len(hypotheses) > 0
    
    # Test the best hypothesis with NP ATN
    best_tokens = latn_tokenize_best(sentence)
    result = run_np(TokenStream(best_tokens))
    
    assert result is not None
    assert result.noun == "cube"
    assert isinstance(result.vector, VectorSpace)
    print(f"LATN best hypothesis gave {len(best_tokens)} tokens: {[t.word for t in best_tokens]}")
    print(f"Found {len(hypotheses)} total hypotheses")


def test_latn_multiple_hypotheses_with_np():
    """Test how NP ATN handles different tokenization hypotheses."""
    sentence = "the big red sphere"
    hypotheses = latn_tokenize_layer1(sentence)
    
    print(f"Testing sentence: '{sentence}'")
    print(f"LATN generated {len(hypotheses)} hypotheses:")
    
    successful_parses = []
    
    # Try each hypothesis with the NP ATN
    for i, hypothesis in enumerate(hypotheses):
        print(f"\nHypothesis {i+1} (conf={hypothesis.confidence:.2f}): {hypothesis.description}")
        print(f"  Tokens: {[t.word for t in hypothesis.tokens]}")
        
        try:
            result = run_np(TokenStream(hypothesis.tokens))
            if result is not None:
                successful_parses.append((hypothesis, result))
                print(f"  ✅ NP Parse successful: noun='{result.noun}', vector={result.vector}")
            else:
                print(f"  ❌ NP Parse failed")
        except Exception as e:
            print(f"  ❌ NP Parse error: {e}")
    
    # Should have at least one successful parse
    assert len(successful_parses) > 0
    
    # The best hypothesis should parse successfully
    best_result = run_np(TokenStream(hypotheses[0].tokens))
    assert best_result is not None
    assert best_result.noun == "sphere"


def test_latn_ambiguous_compound_with_np():
    """Test LATN with potentially ambiguous compounds in NP context."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    
    # Set up vocabulary for red/cube vs "red cube" compound ambiguity
    originals = {}
    test_words = {
        'red': "adj",
        'cube': "noun", 
        'red cube': "noun"  # compound noun
    }
    
    for word, features in test_words.items():
        originals[word] = word in SEMANTIC_VECTOR_SPACE
        if not originals[word]:
            if features == "adj":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", red=1.0)
            else:  # noun
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("noun")
    
    try:
        sentence = "a red cube"  # Could be "red cube" (compound) or "red" + "cube"
        hypotheses = latn_tokenize_layer1(sentence)

        print(f"Testing ambiguous sentence: '{sentence}'")
        print(f"LATN generated {len(hypotheses)} hypotheses:")

        successful_parses = []

        for i, hypothesis in enumerate(hypotheses):
            print(f"\nHypothesis {i+1} (conf={hypothesis.confidence:.2f}): {hypothesis.description}")
            print(f"  Tokens: {[t.word for t in hypothesis.tokens]}")

            try:
                result = run_np(TokenStream(hypothesis.tokens))
                if result is not None:
                    successful_parses.append((hypothesis, result))
                    print(f"  ✅ NP Parse successful: noun='{result.noun}'")
                    print(f"    Determiner: {result.determiner}")
                    print(f"    Consumed tokens: {result.get_consumed_words()}")
                    print(f"    Vector: {result.vector}")
                else:
                    print(f"  ❌ NP Parse failed")
            except Exception as e:
                print(f"  ❌ NP Parse error: {e}")

        # Should have at least one successful parse
        assert len(successful_parses) > 0, f"Expected at least one successful parse for '{sentence}'"
        
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]
    print(f"\nTotal successful NP parses: {len(successful_parses)}")


def test_latn_with_adverb_adjective_np():
    """Test LATN tokenizer with adverb-adjective scaling in NP."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    
    # Set up vocabulary for adverb-adjective test
    originals = {}
    test_words = {
        'very': "adv",
        'large': "adj",
        'blue': "adj",
        'sphere': "noun"  # using sphere instead of ball since it exists
    }
    
    for word, features in test_words.items():
        originals[word] = word in SEMANTIC_VECTOR_SPACE
        if not originals[word]:
            if features == "adv":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adv")
            elif features == "adj" and word == "large":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", scaleX=2.0, scaleY=2.0, scaleZ=2.0)
            elif features == "adj" and word == "blue":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", blue=1.0)
            else:  # noun
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("noun")
    
    try:
        sentence = "a very large blue sphere"
        hypotheses = latn_tokenize_layer1(sentence)

        print(f"Testing adverb-adjective sentence: '{sentence}'")
        print(f"LATN generated {len(hypotheses)} hypotheses")

        # Test the best hypothesis
        best_tokens = latn_tokenize_best(sentence)
        result = run_np(TokenStream(best_tokens))

        assert result is not None, f"Expected successful parse for '{sentence}'"
        assert result.noun == "sphere"
        assert result.determiner == "a"
        
        # Check consumed tokens
        consumed = result.get_consumed_words()
        print(f"Consumed tokens: {consumed}")
        assert consumed == ["a", "very", "large", "blue", "sphere"]
        
        # Check that adverb scaling was applied (very should scale large)
        print(f"Final vector: {result.vector}")
        # The vector should show the scaled adjective effects
        assert result.vector["adj"] > 1.0, "Should have adjective components"
        
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]


def test_latn_semantic_grounding_ambiguous_objects():
    """Test LATN Layer 2 semantic grounding with ambiguous scene objects using SceneModel."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    from engraf.visualizer.scene.scene_object import SceneObject
    from engraf.visualizer.scene.scene_model import SceneModel
    
    # Set up vocabulary
    originals = {}
    test_words = {
        'the': "det",
        'red': "adj",
        'green': "adj", 
        'box': "noun",
        'tall': "adj"
    }
    
    for word, features in test_words.items():
        originals[word] = word in SEMANTIC_VECTOR_SPACE
        if not originals[word]:
            if features == "det":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("det", singular=1.0)
            elif features == "adj" and word == "red":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", red=1.0)
            elif features == "adj" and word == "green":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", green=1.0)
            elif features == "adj" and word == "tall":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", scaleY=2.0)
            else:  # noun
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("noun")
    
    try:
        # Create SceneModel with ambiguous objects
        scene = SceneModel()
        
        red_box_vector = vector_from_features("noun", red=1.0)
        green_box_vector = vector_from_features("noun", green=1.0)
        tall_red_box_vector = vector_from_features("noun", red=1.0, scaleY=2.0)
        
        scene_objects = [
            SceneObject("box", red_box_vector, object_id="red_box_1"),
            SceneObject("box", green_box_vector, object_id="green_box_1"), 
            SceneObject("box", tall_red_box_vector, object_id="tall_red_box_1")
        ]
        
        # Add objects to scene
        for obj in scene_objects:
            scene.add_object(obj)
        
        print(f"Created scene with {len(scene_objects)} objects:")
        for obj in scene_objects:
            print(f"  {obj.object_id}: {obj.vector}")
        
        # Test case 1: "the box" should remain ambiguous (3 possible matches)
        ambiguous_sentence = "the box"
        print(f"\nTesting ambiguous reference: '{ambiguous_sentence}'")
        
        hypotheses = latn_tokenize_layer1(ambiguous_sentence)
        print(f"LATN generated {len(hypotheses)} tokenization hypotheses")
        
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            # Parse as NP
            np = run_np(TokenStream(hyp.tokens))
            if np is not None:
                print(f"    ✅ Parsed NP: {np}")
                print(f"    Consumed tokens: {np.get_consumed_words()}")
                print(f"    NP vector for matching: {np.vector}")
                
                # Test semantic grounding using SceneModel
                matched_objects = scene.find_noun_phrase(np)
                assert len(matched_objects) == 3, f"Should match 3 objects, got {len(matched_objects)}"
                # Verify the NP was parsed correctly
                assert np.noun == "box"
                assert np.determiner == "the"
            else:
                print(f"    ❌ Failed to parse as NP")
        
        # Test case 2: "the red box" should match red boxes
        specific_sentence = "the red box"
        print(f"\nTesting more specific reference: '{specific_sentence}'")
        
        hypotheses = latn_tokenize_layer1(specific_sentence)
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            np = run_np(TokenStream(hyp.tokens))
            if np is not None:
                print(f"    ✅ Parsed NP: {np}")
                print(f"    NP vector: {np.vector}")
                
                # Test semantic grounding for red box
                matched_objects = scene.find_noun_phrase(np)
                assert len(matched_objects) == 2, f"Should match 2 red boxes, got {len(matched_objects)}"                    
                    # Should match a red box (either red_box_1 or tall_red_box_1)
                
                # This NP should have red=1.0 to help distinguish from green box
                assert np.vector["red"] > 0.0, "Should have red color for matching"
                assert np.vector["noun"] > 0.0, "Should be a noun"
                
        # Test case 3: "the green box" should have 1 specific match
        specific_green_sentence = "the green box"
        print(f"\nTesting specific green reference: '{specific_green_sentence}'")
        
        hypotheses = latn_tokenize_layer1(specific_green_sentence)
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            np = run_np(TokenStream(hyp.tokens))
            if np is not None:
                print(f"    ✅ Parsed NP: {np}")
                print(f"    NP vector: {np.vector}")
                
                # Test semantic grounding for green box
                matched_objects = scene.find_noun_phrase(np)
                assert len(matched_objects) == 1, f"Should match 1 green box, got {len(matched_objects)}"
                
                # This NP should have green=1.0 for unique matching
                assert np.vector["green"] > 0.0, "Should have green color for matching"
                
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]

def test_latn_vs_original_tokenizer():
    """Compare LATN multi-hypothesis vs original single tokenizer."""
    text = "a big red cube"  # Proper NP with determiner
    
    # Original tokenizer - single hypothesis
    original_tokens = list(latn_tokenize_best(text))
    print(f"Original tokenizer: {len(original_tokens)} tokens")
    for tok in original_tokens:
        print(f"  {tok.word}")
    
    # LATN tokenizer - multiple hypotheses  
    latn_hypotheses = latn_tokenize_layer1(text)
    print(f"LATN tokenizer: {len(latn_hypotheses)} hypotheses")
    for i, hyp in enumerate(latn_hypotheses, 1):
        tokens = [tok.word for tok in hyp.tokens]
        print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
    
    # Both should successfully parse as NP
    np1 = run_np(TokenStream(original_tokens))
    
    assert np1 is not None
    assert np1.noun == "cube"
    assert np1.determiner == "a"
    
    # Check the consumed tokens tracking
    consumed_words = np1.get_consumed_words()
    print(f"Consumed tokens: {consumed_words}")
    assert consumed_words == ["a", "big", "red", "cube"]
    
    # Check original text reconstruction
    original_text = np1.get_original_text()
    print(f"Reconstructed text: '{original_text}'")
    assert original_text == "a big red cube"

def test_latn_ambiguous_light_house():
    """Test LATN tokenizer with the classic 'green box' ambiguity (using available vocabulary)."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    
    # Ensure we have both interpretations in vocabulary
    original_green = 'green' in SEMANTIC_VECTOR_SPACE
    original_box = 'box' in SEMANTIC_VECTOR_SPACE  
    original_greenbox = 'green box' in SEMANTIC_VECTOR_SPACE
    
    if not original_green:
        SEMANTIC_VECTOR_SPACE['green'] = vector_from_features("adj", color="green")
    if not original_box:
        SEMANTIC_VECTOR_SPACE['box'] = vector_from_features("noun")
    if not original_greenbox:
        SEMANTIC_VECTOR_SPACE['green box'] = vector_from_features("noun")  # Compound noun
    
    try:
        sentence = "a green box"
        
        # LATN should give multiple hypotheses
        latn_hypotheses = latn_tokenize_layer1(sentence)
        print(f"LATN hypotheses for '{sentence}': {len(latn_hypotheses)}")
        
        for i, hyp in enumerate(latn_hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            # Test NP parsing for each hypothesis
            print(f"    Testing NP parsing...")
            np = run_np(TokenStream(hyp.tokens))
            
            if np is not None:
                print(f"    ✅ Success: NP={np}")
                print(f"    ✅ Consumed tokens: {np.get_consumed_words()}")
                # Both interpretations should parse successfully
                assert np.noun in ["box", "green box"], f"Expected noun to be 'box' or 'green box', got '{np.noun}'"
                assert np.determiner == "a", f"Expected determiner 'a', got '{np.determiner}'"
            else:
                print(f"    ❌ Failed to parse as NP")
                assert False, f"Hypothesis {i} should parse as valid NP"
        
        # Should have at least 2 hypotheses for this ambiguous case
        assert len(latn_hypotheses) >= 2, "Should have multiple hypotheses for 'green box'"
        
    finally:
        # Clean up test vocabulary
        if not original_green and 'green' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['green']
        if not original_box and 'box' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['box']
        if not original_greenbox and 'green box' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['green box']

def test_latn_three_way_ambiguity_with_np():
    """Test LATN with three-way ambiguity: 'very big box'."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    
    # Set up vocabulary for three-way ambiguity using words we know exist or can add
    originals = {}
    test_words = {
        'very': "adv",
        'big': "adj", 
        'box': "noun",
        'big box': "noun",
        'very big box': "noun"
    }
    
    for word, features in test_words.items():
        originals[word] = word in SEMANTIC_VECTOR_SPACE
        if not originals[word]:
            if features == "adv":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adv")
            elif features == "adj":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", scaleX=2.0, scaleY=2.0, scaleZ=2.0)
            else:  # noun
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("noun")
    
    try:
        sentence = "a very big box"
        
        # LATN should give multiple hypotheses
        latn_hypotheses = latn_tokenize_layer1(sentence)
        print(f"LATN hypotheses for '{sentence}': {len(latn_hypotheses)}")
        
        successful_parses = 0
        for i, hyp in enumerate(latn_hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            # Test NP parsing for each hypothesis
            print(f"    Testing NP parsing...")
            np = run_np(TokenStream(hyp.tokens))
            
            if np is not None:
                print(f"    ✅ Success: NP={np}")
                print(f"    ✅ Consumed tokens: {np.get_consumed_words()}")
                successful_parses += 1
                # Verify the parse makes sense
                assert np.determiner == "a", f"Expected determiner 'a', got '{np.determiner}'"
                expected_nouns = ["box", "big box", "very big box"]
                assert np.noun in expected_nouns, f"Expected noun in {expected_nouns}, got '{np.noun}'"
            else:
                print(f"    ❌ Failed to parse as NP")
        
        # Should have multiple hypotheses and at least some should parse
        assert len(latn_hypotheses) >= 2, "Should have multiple hypotheses for 'very big box'"
        assert successful_parses > 0, "At least one hypothesis should parse as valid NP"
        
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]

def test_coordinated_np():
    """Test Layer 2 NP tokenization with coordinated noun phrases """
    executor = LATNLayerExecutor()

    # Test coordinated NPs: "a red box and a blue circle and a octahedron"
    result = executor.execute_layer2('a red box and a blue circle and a octahedron', report=True, tokenize_only=True)

    assert result.success, "Failed to tokenize coordinated NPs with adverb-adjective sequences"
    assert len(result.hypotheses) == 1, "Should generate 1 hypothesis"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token (the coordinated NP)
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    main_np = main_hyp.tokens[0].word
    assert main_np.startswith("CONJ-NP"), "First token should be a conjunction NP"
    conj_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token (the coordinated NP)
    assert len(conj_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(conj_hyp.tokens)}"
    conj_np = conj_hyp.tokens[0].word
    assert conj_np.startswith("CONJ-NP"), "First token should be a conjunction NP"
    conj = conj_hyp.tokens[0]
    assert conj is not None
    original_np = conj.phrase
    assert isinstance(original_np, ConjunctionPhrase), f"Expected ConjunctionPhrase, got {type(original_np)}"
    parts = [np for np in original_np.phrases]
    assert len(parts) == 3, f"Should have 3 parts, got {len(parts)}"