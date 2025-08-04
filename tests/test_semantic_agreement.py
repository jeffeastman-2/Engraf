"""
Test semantic agreement validation.

This module tests whether commands make semantic sense given the current scene state,
such as "move 3 circles" when only 1 circle exists in the scene.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace, vector_from_features
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.interpreter.semantic_validator import SemanticAgreementValidator


def test_semantic_agreement_basic():
    """Test basic semantic agreement validation."""
    # Create a scene with one circle
    scene = SceneModel()
    circle = SceneObject(
        name="circle",
        vector=vector_from_features("noun")
    )
    scene.add_object(circle)
    
    validator = SemanticAgreementValidator(scene)
    
    # Test cases
    test_cases = [
        ("move a circle to [1, 2, 3]", True, "Should succeed - 1 circle requested, 1 available"),
        ("move the circle to [1, 2, 3]", True, "Should succeed - definite article, 1 available"),
        ("move 3 circles to [1, 2, 3]", False, "Should fail - 3 circles requested, only 1 available"),
        ("move two circles to [1, 2, 3]", False, "Should fail - 2 circles requested, only 1 available"),
    ]
    
    print("\n" + "="*80)
    print("SEMANTIC AGREEMENT VALIDATION TEST")
    print("="*80)
    print(f"Scene state: {len(scene.objects)} circle(s) available")
    print()
    
    for sentence_text, expected_valid, description in test_cases:
        print(f"Testing: {sentence_text}")
        print(f"Expected: {'✅ Valid' if expected_valid else '❌ Invalid'} - {description}")
        
        # Parse the sentence
        tokens = TokenStream(tokenize(sentence_text))
        sentence = run_sentence(tokens)
        
        if sentence is None:
            print(f"⚠️  Could not parse sentence")
            continue
            
        # Validate semantics
        is_valid, error_msg = validator.validate_command(sentence, sentence_text)
        
        if is_valid == expected_valid:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"Result: {status} - {'Valid' if is_valid else f'Invalid: {error_msg}'}")
        print("-" * 60)
        
        # Assert for test framework
        assert is_valid == expected_valid, f"Expected {expected_valid}, got {is_valid} for '{sentence_text}'"


def test_semantic_agreement_multiple_objects():
    """Test semantic agreement with multiple objects."""
    # Create a scene with 3 circles and 2 cubes
    scene = SceneModel()
    
    # Add 3 circles
    for i in range(3):
        circle = SceneObject(
            name="circle",
            vector=vector_from_features("noun")
        )
        scene.add_object(circle)
    
    # Add 2 cubes  
    for i in range(2):
        cube = SceneObject(
            name="cube", 
            vector=vector_from_features("noun")
        )
        scene.add_object(cube)
    
    validator = SemanticAgreementValidator(scene)
    
    test_cases = [
        ("move 2 circles to [1, 2, 3]", True, "2 circles requested, 3 available"),
        ("move 3 circles to [1, 2, 3]", True, "3 circles requested, 3 available"), 
        ("move 4 circles to [1, 2, 3]", False, "4 circles requested, only 3 available"),
        ("move 2 cubes to [1, 2, 3]", True, "2 cubes requested, 2 available"),
        ("move 3 cubes to [1, 2, 3]", False, "3 cubes requested, only 2 available"),
        ("move a cube to [1, 2, 3]", True, "1 cube requested, 2 available"),
    ]
    
    print("\n" + "="*80)
    print("SEMANTIC AGREEMENT TEST - MULTIPLE OBJECTS")
    print("="*80)
    print(f"Scene state: {len([obj for obj in scene.objects if obj.name == 'circle'])} circles, {len([obj for obj in scene.objects if obj.name == 'cube'])} cubes")
    print()
    
    for sentence_text, expected_valid, description in test_cases:
        print(f"Testing: {sentence_text}")
        print(f"Expected: {'✅ Valid' if expected_valid else '❌ Invalid'} - {description}")
        
        # Parse the sentence
        tokens = TokenStream(tokenize(sentence_text))
        sentence = run_sentence(tokens)
        
        if sentence is None:
            print(f"⚠️  Could not parse sentence")
            continue
            
        # Validate semantics
        is_valid, error_msg = validator.validate_command(sentence, sentence_text)
        
        if is_valid == expected_valid:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"Result: {status} - {'Valid' if is_valid else f'Invalid: {error_msg}'}")
        print("-" * 60)
        
        # Assert for test framework
        assert is_valid == expected_valid, f"Expected {expected_valid}, got {is_valid} for '{sentence_text}'"


def test_semantic_agreement_edge_cases():
    """Test edge cases for semantic agreement."""
    # Empty scene
    scene = SceneModel()
    validator = SemanticAgreementValidator(scene)
    
    test_cases = [
        ("move a circle to [1, 2, 3]", False, "No circles in empty scene"),
        ("move the circle to [1, 2, 3]", False, "No circles in empty scene"), 
        ("move 2 circles to [1, 2, 3]", False, "No circles in empty scene"),
    ]
    
    print("\n" + "="*80)
    print("SEMANTIC AGREEMENT TEST - EDGE CASES")
    print("="*80)
    print(f"Scene state: Empty scene (0 objects)")
    print()
    
    for sentence_text, expected_valid, description in test_cases:
        print(f"Testing: {sentence_text}")
        print(f"Expected: {'✅ Valid' if expected_valid else '❌ Invalid'} - {description}")
        
        # Parse the sentence
        tokens = TokenStream(tokenize(sentence_text))
        sentence = run_sentence(tokens)
        
        if sentence is None:
            print(f"⚠️  Could not parse sentence")
            continue
            
        # Validate semantics
        is_valid, error_msg = validator.validate_command(sentence, sentence_text)
        
        if is_valid == expected_valid:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"Result: {status} - {'Valid' if is_valid else f'Invalid: {error_msg}'}")
        print("-" * 60)
        
        # Assert for test framework
        assert is_valid == expected_valid, f"Expected {expected_valid}, got {is_valid} for '{sentence_text}'"


if __name__ == "__main__":
    test_semantic_agreement_basic()
    test_semantic_agreement_multiple_objects() 
    test_semantic_agreement_edge_cases()
    print("\n" + "="*80)
    print("🎉 ALL SEMANTIC AGREEMENT TESTS COMPLETED")
    print("="*80)
