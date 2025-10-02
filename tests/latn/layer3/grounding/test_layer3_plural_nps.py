from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from tests.latn.dummy_test_scene import DummyTestScene

def test_boxes_grounding_above_table():
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 
    
    # Test plural NPs: "the boxes above the table"
    result = executor.execute_layer3('the boxes above the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 3, f"NP grounding should reference 3 objects, got {len(grounding_info)}"
    assert grounding_info[0].object_id == "box-1", f"First grounded object should be box-1, got {grounding_info[0].object_id}"
    assert grounding_info[1].object_id == "box-2", f"Second grounded object should be box-2, got {grounding_info[1].object_id}"
    assert grounding_info[2].object_id == "box-3", f"Third grounded object should be box-3, got {grounding_info[2].object_id}"

def test_boxes_grounding_below_table():
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 

    # Test plural NPs: "the boxes below the table"
    result = executor.execute_layer3('the boxes below the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 2, "Should generate 2 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 2, f"NP grounding should reference 2 objects, got {len(grounding_info)}"  
    assert grounding_info[0].object_id == "box-4", f"First grounded object should be box-4, got {grounding_info[0].object_id}"
    assert grounding_info[1].object_id == "box-5", f"Second grounded object should be box-5, got {grounding_info[1].object_id}"

def test_boxes_grounding_below_and_right_of_table():
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 

    # Test plural NPs: "the boxes below the table and right of the table"
    result = executor.execute_layer3('the boxes below the table and right of the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 5, "Should generate 5 hypotheses"

    main_hyp = result.hypotheses[0]
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 1, f"NP grounding should reference 1 object, got {len(grounding_info)}"    
    assert grounding_info[0].object_id == "box-5", f"First grounded object should be box-5, got {grounding_info[0].object_id}"

def test_boxes_grounding_above_and_left_of_table():
    #assert False
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 

    # Test plural NPs: "the boxes above the table and left of the table"
    result = executor.execute_layer3('the boxes above the table and left of the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 5, "Should generate 5 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 1, f"NP grounding should reference 1 object, got {len(grounding_info)}"    
    assert grounding_info[0].object_id == "box-3", f"First grounded object should be box-3, got {grounding_info[0].object_id}"

def test_objects_grounding_left_of_table():
    #assert False
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 

    # Test plural NPs: "the objects left of the table"
    result = executor.execute_layer3('the objects left of the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 3, "Should generate 3 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert hasattr(np, 'grounding'), "NP token should have grounding info"
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 4, f"NP grounding should reference 4 objects, got {len(grounding_info)}"    
    assert grounding_info[0].object_id == "pyramid-1", f"First grounded object should be pyramid-1, got {grounding_info[0].object_id}"
    assert grounding_info[1].object_id == "sphere-1", f"Second grounded object should be sphere-1, got {grounding_info[1].object_id}"
    assert grounding_info[2].object_id == "box-3", f"Third grounded object should be box-3, got {grounding_info[2].object_id}"
    assert grounding_info[3].object_id == "box-4", f"Fourth grounded object should be box-4, got {grounding_info[3].object_id}"

def test_objects_and_spheres_grounding_left_of_table():
    #assert False
    """Test Layer 3 NP tokenization with plural noun phrases and preposition"""
    executor = LATNLayerExecutor(DummyTestScene().get_scene2()) 

    # Test plural NPs: "the objects and spheres left of the table"
    result = executor.execute_layer3('the objects and spheres left of the table',report=True)
    
    assert result.success, "Failed to tokenize plural NPs"
    assert len(result.hypotheses) == 3, "Should generate 3 hypotheses"

    main_hyp = result.hypotheses[0]
    # Should have exactly 1 NP token
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    token = main_hyp.tokens[0]
    assert token.isa("NP"), "First token should be a noun phrase NP"
    np = token.phrase
    assert token.isa("plural"), "NP should be marked as plural"
    assert token.isa("conj")
    grounding_info = np.grounding.get('scene_objects')
    assert len(grounding_info) == 4, f"NP grounding should reference 4 objects, got {len(grounding_info)}"    

    