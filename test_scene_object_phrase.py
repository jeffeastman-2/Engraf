#!/usr/bin/env python3
"""
Test SceneObjectPhrase functionality
"""

from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS


def test_scene_object_phrase_creation():
    """Test creating SceneObjectPhrase from NounPhrase."""
    print("=== Testing SceneObjectPhrase Creation ===")
    
    # Create a simple NounPhrase
    np = NounPhrase()
    
    # Add a noun token
    noun_token = VectorSpace(word="cube")
    noun_token["noun"] = 1.0
    noun_token["singular"] = 1.0
    np.apply_noun(noun_token)
    
    # Add a determiner
    det_token = VectorSpace(word="the")
    det_token["det"] = 1.0
    det_token["def"] = 1.0
    np.apply_determiner(det_token)
    
    print(f"Original NP: {np}")
    print(f"NP vector dimensions: {[(dim, np.vector[dim]) for dim in VECTOR_DIMENSIONS if np.vector[dim] != 0.0]}")
    
    # Create SceneObjectPhrase from NP
    so = SceneObjectPhrase.from_noun_phrase(np)
    print(f"SceneObjectPhrase: {so}")
    print(f"SO vector dimensions: {[(dim, so.vector[dim]) for dim in VECTOR_DIMENSIONS if so.vector[dim] != 0.0]}")
    
    # Verify SO marker was added
    assert so.vector["SO"] == 1.0, "SO dimension should be set to 1.0"
    print("✅ SO dimension correctly added")
    
    # Verify other attributes were copied
    assert so.noun == np.noun, "Noun should be copied"
    assert so.determiner == np.determiner, "Determiner should be copied"
    assert len(so.consumed_tokens) == len(np.consumed_tokens), "Consumed tokens should be copied"
    print("✅ All attributes correctly copied")
    
    # Test resolution to scene object
    scene_obj = SceneObject("test_cube", {})
    scene_obj.vector = VectorSpace()
    scene_obj.vector["noun"] = 1.0
    scene_obj.vector["singular"] = 1.0
    
    so.resolve_to_scene_object(scene_obj)
    assert so.is_resolved(), "SO should be resolved"
    assert so.get_resolved_object() == scene_obj, "Resolved object should match"
    print("✅ Scene object resolution works")
    
    print(f"Final resolved SO: {so}")
    print("✅ All tests passed!")


def test_np_vs_so_difference():
    """Test that NP and SO are different types."""
    print("\n=== Testing NP vs SO Type Difference ===")
    
    # Create NP
    np = NounPhrase()
    noun_token = VectorSpace(word="box")
    noun_token["noun"] = 1.0
    np.apply_noun(noun_token)
    
    # Create SO from NP
    so = SceneObjectPhrase.from_noun_phrase(np)
    
    print(f"NP type: {type(np).__name__}")
    print(f"SO type: {type(so).__name__}")
    print(f"NP has SO dimension: {np.vector['SO']}")
    print(f"SO has SO dimension: {so.vector['SO']}")
    
    # Verify NP doesn't have resolve methods
    assert not hasattr(np, 'resolve_to_scene_object'), "NP should not have resolve_to_scene_object method"
    print("✅ NP correctly lacks resolution methods")
    
    # Verify SO has resolve methods
    assert hasattr(so, 'resolve_to_scene_object'), "SO should have resolve_to_scene_object method"
    assert hasattr(so, 'is_resolved'), "SO should have is_resolved method"
    assert hasattr(so, 'get_resolved_object'), "SO should have get_resolved_object method"
    print("✅ SO correctly has resolution methods")
    
    print("✅ NP and SO type differences verified!")


if __name__ == "__main__":
    test_scene_object_phrase_creation()
    test_np_vs_so_difference()
    print("\n🎉 All SceneObjectPhrase tests passed!")
