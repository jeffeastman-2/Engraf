#!/usr/bin/env python3
"""
Layer 3 Spatial Validation Test

This test demonstrates Layer 3's true purpose: validating which spatial attention
relationships are valid based on current scene state and resolving complex
PP attention chains.

Layer 3 Architecture:
- Input: Complex PP chains with resolved SceneObjectPhrases
- Process: Validate spatial relationships and resolve attention combinatorics
- Output: Valid spatial attention chains with rejected invalid relationships

Test sentence: "move the box above the table beside the pyramid under the sphere to [3,4,5]"
Expected Layer 2 output: SO PPSO PPSO PPSO PP (box, table, pyramid, sphere, [3,4,5])
Layer 3 job: Determine which PPSO attends to which SO/PPSO based on scene validity
"""

import unittest
from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features


class TestLayer3SpatialValidation(unittest.TestCase):
    """Test Layer 3 spatial validation and attention resolution."""
    
    def setUp(self):
        """Set up complex scene for Layer 3 spatial validation tests."""
        # Create scene with all objects from "move the box above the table right of the pyramid under the sphere to [3,4,5]"
        self.scene = SceneModel()
        
        # Create scene objects with distinct vectors
        box_vector = vector_from_word("box")
        table_vector = vector_from_word("table")
        pyramid_vector = vector_from_word("pyramid")
        sphere_vector = vector_from_word("sphere")
        
        self.box = SceneObject("box", box_vector, object_id="box-1")
        self.table = SceneObject("table", table_vector, object_id="table-1") 
        self.pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
        self.sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")
        
        # Set reasonable scale values for spatial calculations
        for obj in [self.box, self.table, self.pyramid, self.sphere]:
            obj.vector['scaleX'] = 1.0
            obj.vector['scaleY'] = 1.0 
            obj.vector['scaleZ'] = 1.0
        
        # Add all objects to scene
        self.scene.add_object(self.box)
        self.scene.add_object(self.table)
        self.scene.add_object(self.pyramid) 
        self.scene.add_object(self.sphere)
        
        # Set up Layer 3 executor with scene
        self.layer3_executor = LATNLayerExecutor(self.scene)
        
        # Test sentence with spatial_location prepositions (including multi-word "right of")
        self.sentence = "move the box above the table right of the pyramid under the sphere to [3,4,5]"
    
    def test_layer3_spatial_validation_with_positioned_objects(self):
        """Test Layer 3 spatial validation with positioned objects to create realistic spatial relationships.
        
        This test positions objects to create valid spatial relationships:
        - Table at [0, 0, 0] (reference)
        - Box at [0, 1, 0] (above table - valid for "above table")  
        - Pyramid at [2, 0, 0] (right of table - valid for "right of pyramid")
        - Sphere at [0, -2, 0] (below table - valid for "under")
        
        Expected: Some PP attachment combinations should be filtered out as spatially invalid.
        """
        # Position objects to create realistic spatial relationships
        self.table.position = [0, 0, 0]    # Reference object
        self.box.position = [0, 1, 0]      # Above table (valid for "above table")
        self.pyramid.position = [2, 0, 0]  # Beside table (valid for "beside pyramid")  
        self.sphere.position = [0, -2, 0]  # Below table (valid for "under sphere")
        
        # Update vector positions to match
        self.table.vector['locX'], self.table.vector['locY'], self.table.vector['locZ'] = 0, 0, 0
        self.box.vector['locX'], self.box.vector['locY'], self.box.vector['locZ'] = 0, 1, 0
        self.pyramid.vector['locX'], self.pyramid.vector['locY'], self.pyramid.vector['locZ'] = 2, 0, 0
        self.sphere.vector['locX'], self.sphere.vector['locY'], self.sphere.vector['locZ'] = 0, -2, 0
        
        print(f"📍 Object positions:")
        print(f"  Box: {self.box.position} (above table)")
        print(f"  Table: {self.table.position} (reference)")
        print(f"  Pyramid: {self.pyramid.position} (beside table)")
        print(f"  Sphere: {self.sphere.position} (below table)")
        
        # Execute Layer 3 directly on the sentence
        print(f"\n🔬 Processing through Layer 3 executor...")
        layer3_result = self.layer3_executor.execute_layer3(
            self.sentence,
            enable_semantic_grounding=True
        )
        
        print(f"Layer 3 execution result:")
        print(f"  Success: {layer3_result.success}")
        print(f"  Hypotheses: {len(layer3_result.hypotheses) if hasattr(layer3_result, 'hypotheses') else 0}")
        
        if hasattr(layer3_result, 'hypotheses') and layer3_result.hypotheses:
            print(f"\nLayer 3 hypotheses details:")
            for i, hypothesis in enumerate(layer3_result.hypotheses):
                print(f"  Hypothesis [{i}]:")
                print(f"    Confidence: {getattr(hypothesis, 'confidence', 'N/A')}")
                print(f"    Type: {type(hypothesis).__name__}")
                
                # Print tokens if available
                if hasattr(hypothesis, 'tokens'):
                    print(f"    Tokens ({len(hypothesis.tokens)}):")
                    for j, token in enumerate(hypothesis.tokens):
                        print(f"      [{j}] {token}")
                
                # Print prepositional phrases if available
                if hasattr(hypothesis, 'prepositional_phrases'):
                    print(f"    PPs ({len(hypothesis.prepositional_phrases)}):")
                    for j, pp in enumerate(hypothesis.prepositional_phrases):
                        print(f"      [{j}] {pp}")
                
                # Print any other relevant attributes
                attrs_to_check = ['description', 'spatial_chains', 'attachment_chains']
                for attr in attrs_to_check:
                    if hasattr(hypothesis, attr):
                        value = getattr(hypothesis, attr)
                        # Clean up verbose descriptions
                        if attr == 'description' and isinstance(value, str):
                            # Extract just the grounding summary (after the →)
                            if '→' in value:
                                grounding_part = value.split('→')[1].strip()
                                print(f"    grounding: {grounding_part}")
                            else:
                                # Show a simplified version
                                clean_desc = value.replace('single-word(', '').replace(')', '').replace(' | ', ' ')
                                print(f"    {attr}: {clean_desc}")
                        else:
                            print(f"    {attr}: {value}")
                
                print()  # Empty line between hypotheses
        
        # Test spatial relationship validation results
        assert layer3_result.success, "Layer 3 should process successfully"
        
        # With positioned objects, spatial validation should filter out invalid combinations
        # Expected: Fewer than 24 hypotheses due to spatial pruning
        if hasattr(layer3_result, 'hypotheses') and layer3_result.hypotheses:
            num_hypotheses = len(layer3_result.hypotheses)
            print(f"  Layer 3 hypotheses found: {num_hypotheses}")
            
            # Detailed analysis of Layer 3 hypotheses
            print(f"\n🔍 DETAILED LAYER 3 HYPOTHESIS ANALYSIS:")
            print(f"{'='*80}")
            
            for i, hypothesis in enumerate(layer3_result.hypotheses):
                print(f"\nHYPOTHESIS [{i+1}/{num_hypotheses}]:")
                print(f"  Confidence: {getattr(hypothesis, 'confidence', 'N/A')}")
                print(f"  Type: {type(hypothesis).__name__}")
                
                # Detailed token analysis
                if hasattr(hypothesis, 'tokens'):
                    print(f"  Tokens ({len(hypothesis.tokens)}):")
                    for j, token in enumerate(hypothesis.tokens):
                        token_type = type(token).__name__
                        
                        # VectorSpace tokens (ungrounded)
                        if hasattr(token, 'word') and token_type == 'VectorSpace':
                            token_repr = f"{token.word} (VectorSpace - ungrounded)"
                        
                        # SceneObjectPhrase (grounded noun phrase)
                        elif hasattr(token, 'scene_object') and token.scene_object:
                            obj_id = getattr(token.scene_object, 'object_id', 'unknown')
                            token_repr = f"SceneObjectPhrase: '{obj_id}' (grounded NP)"
                        
                        # PrepositionalPhrase - check if it contains grounded objects
                        elif hasattr(token, 'preposition') and hasattr(token, 'noun_phrase'):
                            prep = getattr(token, 'preposition', 'unknown')
                            np = getattr(token, 'noun_phrase', None)
                            
                            if np and hasattr(np, 'scene_object') and np.scene_object:
                                # PPSO: PP with grounded SceneObjectPhrase
                                obj_id = np.scene_object.object_id
                                token_repr = f"PP: '{prep}' + SceneObjectPhrase('{obj_id}') - PPSO"
                            elif np and hasattr(np, 'vector_literal'):
                                # PP with vector literal
                                vector_val = getattr(np, 'vector_literal', 'unknown')
                                token_repr = f"PP: '{prep}' + VectorLiteral({vector_val})"
                            else:
                                # Regular PP with ungrounded NP
                                np_type = type(np).__name__ if np else 'None'
                                token_repr = f"PP: '{prep}' + {np_type} (ungrounded)"
                        
                        # NounPhrase (ungrounded)
                        elif token_type == 'NounPhrase':
                            if hasattr(token, 'determiner') and hasattr(token, 'noun'):
                                det = getattr(token, 'determiner', '')
                                noun = getattr(token, 'noun', '')
                                token_repr = f"NounPhrase: '{det} {noun}' (ungrounded)"
                            else:
                                token_repr = f"NounPhrase (ungrounded)"
                        
                        # Fallback for other types
                        else:
                            token_repr = f"{str(token)[:50]} ({token_type})"
                        
                        print(f"    [{j}] {token_repr}")
                
                # Prepositional phrase analysis
                if hasattr(hypothesis, 'prepositional_phrases'):
                    print(f"  Prepositional Phrases ({len(hypothesis.prepositional_phrases)}):")
                    for j, pp in enumerate(hypothesis.prepositional_phrases):
                        pp_type = type(pp).__name__
                        if hasattr(pp, 'preposition'):
                            prep = getattr(pp, 'preposition', 'unknown')
                            np = getattr(pp, 'noun_phrase', None)
                            
                            if np and hasattr(np, 'scene_object') and np.scene_object:
                                # PPSO: PP containing grounded SceneObjectPhrase
                                obj_id = np.scene_object.object_id
                                pp_repr = f"PPSO: '{prep}' contains SceneObjectPhrase('{obj_id}')"
                            elif np and hasattr(np, 'vector_literal'):
                                # PP with vector literal
                                vector_val = getattr(np, 'vector_literal', 'unknown')
                                pp_repr = f"PP: '{prep}' contains VectorLiteral({vector_val})"
                            else:
                                # Regular PP with ungrounded NP
                                np_type = type(np).__name__ if np else 'None'
                                np_content = ""
                                if np and hasattr(np, 'determiner') and hasattr(np, 'noun'):
                                    det = getattr(np, 'determiner', '')
                                    noun = getattr(np, 'noun', '')
                                    np_content = f" '{det} {noun}'"
                                pp_repr = f"PP: '{prep}' contains {np_type}{np_content} (ungrounded)"
                        else:
                            pp_repr = f"{str(pp)[:50]} ({pp_type})"
                        print(f"    [{j}] {pp_repr}")
                
                # Scene object phrases analysis
                scene_object_phrases = []
                if hasattr(hypothesis, 'tokens'):
                    for token in hypothesis.tokens:
                        if hasattr(token, 'scene_object') and token.scene_object:
                            scene_object_phrases.append(token)
                
                if scene_object_phrases:
                    print(f"  Scene Object Phrases ({len(scene_object_phrases)}):")
                    for j, sop in enumerate(scene_object_phrases):
                        obj_id = sop.scene_object.object_id if sop.scene_object else 'None'
                        pos = getattr(sop.scene_object, 'position', 'unknown') if sop.scene_object else 'unknown'
                        print(f"    [{j}] {obj_id} at {pos}")
                
                # Grounding result analysis
                if hasattr(hypothesis, 'grounding_result'):
                    gr = hypothesis.grounding_result
                    print(f"  Grounding Result:")
                    print(f"    Success: {getattr(gr, 'success', 'N/A')}")
                    print(f"    Type: {type(gr).__name__}")
                    if hasattr(gr, 'spatial_chains'):
                        print(f"    Spatial chains: {len(getattr(gr, 'spatial_chains', []))}")
                        for k, chain in enumerate(getattr(gr, 'spatial_chains', [])):
                            print(f"      Chain [{k}]: {chain}")
                    if hasattr(gr, 'filtered_combinations'):
                        filtered = getattr(gr, 'filtered_combinations', [])
                        print(f"    Filtered combinations: {len(filtered)}")
                        for k, combo in enumerate(filtered[:3]):  # Show first 3
                            print(f"      [{k}] {combo}")
                        if len(filtered) > 3:
                            print(f"      ... and {len(filtered) - 3} more")
                
                # Other attributes
                other_attrs = ['description', 'spatial_chains', 'attachment_chains', 'metadata']
                for attr in other_attrs:
                    if hasattr(hypothesis, attr):
                        value = getattr(hypothesis, attr)
                        if attr == 'description' and isinstance(value, str) and len(value) > 100:
                            # Truncate long descriptions
                            print(f"  {attr}: {value[:100]}...")
                        else:
                            print(f"  {attr}: {value}")
                
                print(f"  {'-'*60}")
            
            print(f"\n📊 SUMMARY:")
            print(f"  Total hypotheses: {num_hypotheses}")
            
            # Assert that spatial filtering is working - should have fewer than all 24 combinations
            if num_hypotheses == 24:
                print("⚠️  Warning: All 24 combinations passed spatial validation - spatial filtering may not be working properly")
            else:
                print(f"✅ Spatial filtering working: {24 - num_hypotheses} combinations were filtered out")
        
        # FLEXIBLE ASSERTIONS: Focus on what matters - spatial validation working
        print(f"\n🔬 LAYER 3 GROUNDING VALIDATION:")
    
        assert layer3_result.success, "Layer 3 should process successfully"
        assert hasattr(layer3_result, 'hypotheses') and layer3_result.hypotheses, "Should have hypotheses"
        
        # Verify spatial filtering is working - should have fewer than the maximum possible combinations
        num_hypotheses = len(layer3_result.hypotheses)
        print(f"  ✅ Spatial filtering working: {num_hypotheses} valid combinations found")
        
        # Should have multiple valid hypotheses (not all combinations filtered out)
        assert num_hypotheses >= 1, "Should have at least one valid spatial combination"
        assert num_hypotheses < 24, "Should filter out some invalid spatial combinations"
        
        # Get the best hypothesis for structure checking
        best_hypothesis = layer3_result.hypotheses[0]
        
        # Verify the hypothesis has the expected basic structure
        assert hasattr(best_hypothesis, 'tokens'), "Hypothesis should have tokens"
        tokens = best_hypothesis.tokens
        assert len(tokens) >= 2, "Should have at least move + object tokens"
        
        # Check that first token is the verb
        assert 'move' in str(tokens[0]).lower(), "First token should be the move verb"
        
        # Check that we have grounded scene object phrases (NP tokens with _grounded_phrase)
        grounded_objects = 0
        for token in tokens:
            if hasattr(token, '_grounded_phrase'):
                grounded_objects += 1
                print(f"  ✅ Found grounded object: {token._grounded_phrase}")
        
        assert grounded_objects >= 1, "Should have at least one grounded scene object"
        print(f"  ✅ Found {grounded_objects} grounded scene objects")
        
        print(f"  ✅ Layer 3 spatial validation test passed!")
        
        print(f"✅ All Layer 3 grounding assertions passed!")

        