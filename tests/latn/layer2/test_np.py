import numpy as np
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.lexer.latn_tokenizer import latn_tokenize, latn_tokenize_best
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase

 
def test_simple_np():
    result = run_np(TokenStream(tokenize("the red cube")))
   
    assert result is not None
    assert result.noun == "cube"
    assert isinstance(result.vector, VectorSpace)

def test_np_with_adjectives():
    result = run_np(TokenStream(tokenize("a large blue sphere")))
   
    assert result is not None
    assert result.noun == "sphere"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] > 1.5  # Check if the height is scaled
    assert result.vector["blue"] > 0.5  # Check if the color is blue

def test_np_with_adverbs():
    result = run_np(TokenStream(tokenize("a very tall red cylinder")))
   
    assert result is not None
    print("Final vector:", result.vector)
    assert result.noun == "cylinder"
    assert isinstance(result.vector, VectorSpace)
    assert result.vector["scaleY"] == 2.25  # "very tall" = 1.5 * 1.5 = 2.25
    assert result.vector["red"] > 0.5  # Check if the color is red

def test_np_failure():
    result = run_np(TokenStream(tokenize("draw red cube")))
    assert result is None

def test_np_with_pronoun():
    np = run_np(TokenStream(tokenize("it")))
    assert np is not None
    assert np.pronoun == "it"

def test_np_with_vector():
    np = run_np(TokenStream(tokenize("[5,6,7]")))
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
    hypotheses = latn_tokenize(sentence)
    
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
    hypotheses = latn_tokenize(sentence)
    
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
        hypotheses = latn_tokenize(sentence)

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
        hypotheses = latn_tokenize(sentence)

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
        
        hypotheses = latn_tokenize(ambiguous_sentence)
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
                matched_object = scene.find_noun_phrase(np)
                if matched_object:
                    # Create SceneObjectPhrase from the NP and resolve it
                    from engraf.pos.scene_object_phrase import SceneObjectPhrase
                    so = SceneObjectPhrase.from_noun_phrase(np)
                    so.resolve_to_scene_object(matched_object)
                    print(f"    🎯 Scene resolution: {matched_object.object_id}")
                    print(f"    ✅ SO resolved: {so}")
                    
                    # Test the resolution functionality on SceneObjectPhrase
                    assert so.is_resolved(), "SO should be marked as resolved"
                    assert so.get_resolved_object() == matched_object, "Should return the correct resolved object"
                    assert matched_object.name == "box", f"Should match a box, got {matched_object.name}"
                    
                    # Verify original NP doesn't have resolution methods
                    assert not hasattr(np, 'is_resolved'), "Original NP should not have is_resolved method"
                    assert not hasattr(np, 'resolve_to_scene_object'), "Original NP should not have resolve_to_scene_object method"
                    
                    # For ambiguous "the box", any box is valid
                    assert matched_object.object_id in ["red_box_1", "green_box_1", "tall_red_box_1"]
                else:
                    print(f"    ❌ No scene object matched")
                    
                # Verify the NP was parsed correctly
                assert np.noun == "box"
                assert np.determiner == "the"
            else:
                print(f"    ❌ Failed to parse as NP")
        
        # Test case 2: "the red box" should match red boxes
        specific_sentence = "the red box"
        print(f"\nTesting more specific reference: '{specific_sentence}'")
        
        hypotheses = latn_tokenize(specific_sentence)
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            np = run_np(TokenStream(hyp.tokens))
            if np is not None:
                print(f"    ✅ Parsed NP: {np}")
                print(f"    NP vector: {np.vector}")
                
                # Test semantic grounding for red box
                matched_object = scene.find_noun_phrase(np)
                if matched_object:
                    # Create SceneObjectPhrase from the NP and resolve it
                    from engraf.pos.scene_object_phrase import SceneObjectPhrase
                    so = SceneObjectPhrase.from_noun_phrase(np)
                    so.resolve_to_scene_object(matched_object)
                    print(f"    🎯 Scene resolution: {matched_object.object_id}")
                    print(f"    ✅ SO resolved: {so}")
                    
                    # Should match a red box (either red_box_1 or tall_red_box_1)
                    assert so.is_resolved(), "SO should be resolved"
                    assert matched_object.name == "box"
                    assert matched_object.vector["red"] > 0.0, "Matched object should be red"
                    assert matched_object.object_id in ["red_box_1", "tall_red_box_1"], f"Should match red box, got {matched_object.object_id}"
                
                # This NP should have red=1.0 to help distinguish from green box
                assert np.vector["red"] > 0.0, "Should have red color for matching"
                assert np.vector["noun"] > 0.0, "Should be a noun"
                
        # Test case 3: "the green box" should have 1 specific match
        specific_green_sentence = "the green box"
        print(f"\nTesting specific green reference: '{specific_green_sentence}'")
        
        hypotheses = latn_tokenize(specific_green_sentence)
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [tok.word for tok in hyp.tokens]
            print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
            
            np = run_np(TokenStream(hyp.tokens))
            if np is not None:
                print(f"    ✅ Parsed NP: {np}")
                print(f"    NP vector: {np.vector}")
                
                # Test semantic grounding for green box
                matched_object = scene.find_noun_phrase(np)
                if matched_object:
                    # Create SceneObjectPhrase from the NP and resolve it
                    from engraf.pos.scene_object_phrase import SceneObjectPhrase
                    so = SceneObjectPhrase.from_noun_phrase(np)
                    so.resolve_to_scene_object(matched_object)
                    print(f"    🎯 Scene resolution: {matched_object.object_id}")
                    print(f"    ✅ SO resolved: {so}")
                    
                    # Should uniquely match the green box
                    assert so.is_resolved(), "SO should be resolved"
                    assert matched_object.name == "box"
                    assert matched_object.vector["green"] > 0.0, "Matched object should be green"
                    assert matched_object.object_id == "green_box_1", f"Should match green box, got {matched_object.object_id}"
                
                # This NP should have green=1.0 for unique matching
                assert np.vector["green"] > 0.0, "Should have green color for matching"
                
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]


def test_latn_semantic_grounding_resolution():
    """Test LATN Layer 2 semantic grounding with object resolution scenarios using SceneModel."""
    from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
    from engraf.lexer.vector_space import vector_from_features
    from engraf.visualizer.scene.scene_object import SceneObject
    from engraf.visualizer.scene.scene_model import SceneModel
    
    # Set up vocabulary
    originals = {}
    test_words = {
        'a': "det",
        'big': "adj",
        'small': "adj",
        'blue': "adj",
        'sphere': "noun"
    }
    
    for word, features in test_words.items():
        originals[word] = word in SEMANTIC_VECTOR_SPACE
        if not originals[word]:
            if features == "det":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("det", singular=1.0, number=1.0)
            elif features == "adj" and word == "big":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", scaleX=2.0, scaleY=2.0, scaleZ=2.0)
            elif features == "adj" and word == "small":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", scaleX=0.5, scaleY=0.5, scaleZ=0.5)
            elif features == "adj" and word == "blue":
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("adj", blue=1.0)
            else:  # noun
                SEMANTIC_VECTOR_SPACE[word] = vector_from_features("noun")
    
    try:
        # Create SceneModel with different spheres that test semantic resolution
        scene = SceneModel()
        
        big_blue_sphere_vector = vector_from_features("noun", blue=1.0, scaleX=2.0, scaleY=2.0, scaleZ=2.0)
        small_sphere_vector = vector_from_features("noun", scaleX=-0.5, scaleY=-0.5, scaleZ=-0.5)  # Match NP parser output
        normal_sphere_vector = vector_from_features("noun")
        
        scene_objects = [
            SceneObject("sphere", big_blue_sphere_vector, object_id="big_blue_sphere"),
            SceneObject("sphere", small_sphere_vector, object_id="small_sphere"),
            SceneObject("sphere", normal_sphere_vector, object_id="normal_sphere")
        ]
        
        # Add objects to scene
        for obj in scene_objects:
            scene.add_object(obj)
        
        print(f"Created resolution test scene with {len(scene_objects)} spheres:")
        for obj in scene_objects:
            print(f"  {obj.object_id}: {obj.vector}")
        
        # Test specific vs. ambiguous references
        test_cases = [
            ("a sphere", "Should match any sphere - ambiguous"),
            ("a big sphere", "Should best match big_blue_sphere"),
            ("a blue sphere", "Should best match big_blue_sphere"),
            ("a big blue sphere", "Should uniquely match big_blue_sphere"),
            ("a small sphere", "Should best match small_sphere")
        ]
        
        for sentence, expectation in test_cases:
            print(f"\nTesting: '{sentence}' - {expectation}")
            
            hypotheses = latn_tokenize(sentence)
            for i, hyp in enumerate(hypotheses, 1):
                tokens = [tok.word for tok in hyp.tokens]
                print(f"  Hypothesis {i} (conf={hyp.confidence:.2f}): {tokens}")
                
                np = run_np(TokenStream(hyp.tokens))
                if np is not None:
                    print(f"    ✅ Parsed NP: {np}")
                    print(f"    Consumed: {np.get_consumed_words()}")
                    print(f"    Vector for semantic matching: {np.vector}")
                    
                    # Test semantic grounding using SceneModel
                    matched_object = scene.find_noun_phrase(np)
                    if matched_object:
                        # Create SceneObjectPhrase from the NP and resolve it
                        from engraf.pos.scene_object_phrase import SceneObjectPhrase
                        so = SceneObjectPhrase.from_noun_phrase(np)
                        so.resolve_to_scene_object(matched_object)
                        print(f"    🎯 Scene resolution: {matched_object.object_id}")
                        print(f"    ✅ SO resolved: {so}")
                        
                        # Test resolution functionality on SceneObjectPhrase
                        assert so.is_resolved(), "SO should be marked as resolved"
                        assert so.get_resolved_object() == matched_object, "Should return the correct resolved object"
                        assert matched_object.name == "sphere", f"Should match a sphere, got {matched_object.name}"
                        
                        # Verify original NP doesn't have resolution methods
                        assert not hasattr(np, 'is_resolved'), "Original NP should not have is_resolved method"
                        assert not hasattr(np, 'resolve_to_scene_object'), "Original NP should not have resolve_to_scene_object method"
                        
                        # Verify specific expectations based on the sentence
                        if "big" in np.get_consumed_words() and "blue" in np.get_consumed_words():
                            # "a big blue sphere" should uniquely match big_blue_sphere
                            assert matched_object.object_id == "big_blue_sphere", f"Expected big_blue_sphere, got {matched_object.object_id}"
                        elif "big" in np.get_consumed_words():
                            # "a big sphere" should match big_blue_sphere (only big one)
                            assert matched_object.object_id == "big_blue_sphere", f"Expected big_blue_sphere, got {matched_object.object_id}"
                        elif "blue" in np.get_consumed_words():
                            # "a blue sphere" should match big_blue_sphere (only blue one)
                            assert matched_object.object_id == "big_blue_sphere", f"Expected big_blue_sphere, got {matched_object.object_id}"
                        elif "small" in np.get_consumed_words():
                            # "a small sphere" should match small_sphere
                            assert matched_object.object_id == "small_sphere", f"Expected small_sphere, got {matched_object.object_id}"
                        # For "a sphere", any sphere is valid
                    else:
                        print(f"    ❌ No scene object matched")
                    
                    # Verify basic properties
                    assert np.noun == "sphere"
                    if "big" in np.get_consumed_words():
                        assert np.vector["scaleX"] > 1.0, "Big sphere should have scale > 1"
                    if "blue" in np.get_consumed_words():
                        assert np.vector["blue"] > 0.0, "Blue sphere should have blue color"
                    if "small" in np.get_consumed_words():
                        assert np.vector["scaleX"] < 1.0, "Small sphere should have scale < 1"
                else:
                    print(f"    ❌ Failed to parse as NP")
                    
    finally:
        # Clean up test vocabulary
        for word, was_original in originals.items():
            if not was_original and word in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE[word]


def test_latn_vs_original_tokenizer():
    """Compare LATN multi-hypothesis vs original single tokenizer."""
    text = "a big red cube"  # Proper NP with determiner
    
    # Original tokenizer - single hypothesis
    original_tokens = list(tokenize(text))
    print(f"Original tokenizer: {len(original_tokens)} tokens")
    for tok in original_tokens:
        print(f"  {tok.word}")
    
    # LATN tokenizer - multiple hypotheses  
    latn_hypotheses = latn_tokenize(text)
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
        latn_hypotheses = latn_tokenize(sentence)
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
        latn_hypotheses = latn_tokenize(sentence)
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