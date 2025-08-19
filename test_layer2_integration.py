#!/usr/bin/env python3
"""
Test Layer 2 Grounding with SceneObjectPhrase Integration
"""

from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


def test_layer2_grounding_creates_scene_object_phrase():
    """Test that Layer 2 grounding creates SceneObjectPhrase instances."""
    print("=== Testing Layer 2 Grounding → SceneObjectPhrase ===")
    
    # Create a scene with a red cube
    scene = SceneModel()
    red_cube = SceneObject("red_cube", {})
    red_cube.vector = VectorSpace()
    red_cube.vector["noun"] = 1.0
    red_cube.vector["red"] = 1.0
    red_cube.vector["singular"] = 1.0
    scene.add_object(red_cube)
    
    # Create a NounPhrase for "red cube"
    np = NounPhrase()
    
    # Add adjective "red"
    red_token = VectorSpace(word="red")
    red_token["adj"] = 1.0
    red_token["red"] = 1.0
    np.apply_adjective(red_token)
    
    # Add noun "cube"
    cube_token = VectorSpace(word="cube")
    cube_token["noun"] = 1.0
    cube_token["singular"] = 1.0
    np.apply_noun(cube_token)
    
    print(f"Original NP: {np}")
    print(f"NP type: {type(np).__name__}")
    print(f"NP has resolved_object methods: {hasattr(np, 'resolve_to_scene_object')}")
    
    # Ground the NP using Layer 2
    grounder = Layer2SemanticGrounder(scene)
    result = grounder.ground(np)
    
    print(f"\nGrounding result:")
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence}")
    print(f"Resolved object: {result.resolved_object}")
    print(f"Scene object phrase: {result.scene_object_phrase}")
    
    # Verify results
    assert result.success, "Grounding should succeed"
    assert result.resolved_object == red_cube, "Should resolve to red cube"
    assert result.scene_object_phrase is not None, "Should create SceneObjectPhrase"
    assert isinstance(result.scene_object_phrase, SceneObjectPhrase), "Should be SceneObjectPhrase type"
    assert result.scene_object_phrase.is_resolved(), "SO should be resolved"
    assert result.scene_object_phrase.get_resolved_object() == red_cube, "SO should resolve to red cube"
    assert result.scene_object_phrase.vector["SO"] == 1.0, "SO should have SO dimension"
    
    print("\n✅ All Layer 2 grounding tests passed!")
    print(f"Final SceneObjectPhrase: {result.scene_object_phrase}")


if __name__ == "__main__":
    test_layer2_grounding_creates_scene_object_phrase()
    print("\n🎉 Layer 2 SceneObjectPhrase integration verified!")
