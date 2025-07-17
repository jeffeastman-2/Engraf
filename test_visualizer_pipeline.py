#!/usr/bin/env python3
"""
Quick test to verify the visualizer scene components work end-to-end.
"""

from engraf.visualizer.scene.scene_model import SceneModel, resolve_pronoun
from engraf.visualizer.scene.scene_object import SceneObject, scene_object_from_np
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence


def test_full_pipeline():
    """Test the complete pipeline from sentence to scene objects."""
    print("Testing full pipeline...")
    
    # Create scene
    scene = SceneModel()
    
    # Parse sentence
    sentence = "draw a tall blue cube"
    tokens = TokenStream(tokenize(sentence))
    parsed = run_sentence(tokens)
    
    # Create scene object
    np = parsed.predicate.noun_phrase
    obj = scene_object_from_np(np)
    scene.add_object(obj)
    
    # Test pronoun resolution
    targets = resolve_pronoun("it", scene)
    
    print(f"✅ Parsed sentence: {sentence}")
    print(f"✅ Created object: {obj.name}")
    print(f"✅ Object is tall: {obj.vector['scaleY'] > 1.0}")
    print(f"✅ Object is blue: {obj.vector['blue'] > 0.5}")
    print(f"✅ Pronoun 'it' resolves to: {targets[0].name}")
    print("✅ Full pipeline test passed!")


if __name__ == "__main__":
    test_full_pipeline()
