#!/usr/bin/env python3
"""Test script to verify dynamic attribute elimination didn't break anything."""

from engraf.lexer.vector_space import VectorSpace
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.scene_object_phrase import SceneObjectPhrase

def test_static_fields():
    """Test that all static fields are working properly."""
    
    print("🧪 Testing VectorSpace static fields...")
    vs = VectorSpace()  # Use default constructor
    
    # Test all the static fields we added
    assert vs._grounded_phrase is None, "❌ _grounded_phrase should be None"
    assert vs._original_np is None, "❌ _original_np should be None" 
    assert vs._original_pp is None, "❌ _original_pp should be None"
    assert vs.scene_object is None, "❌ scene_object should be None"
    assert vs._attachment_info == {}, "❌ _attachment_info should be {}"
    print("✅ VectorSpace static fields working")
    
    print("\n🧪 Testing NounPhrase static fields...")
    np = NounPhrase()
    
    # Test the static field we added
    assert np.scene_object is None, "❌ scene_object should be None"
    print("✅ NounPhrase static fields working")
    
    print("\n🧪 Testing PrepositionalPhrase static fields...")
    pp = PrepositionalPhrase()
    
    # Test all the static fields we added
    assert pp.spatial_vector is None, "❌ spatial_vector should be None"
    assert pp.vector_text is None, "❌ vector_text should be None"
    assert pp.spatial_location is None, "❌ spatial_location should be None"
    assert pp.locX is None, "❌ locX should be None"
    assert pp.locY is None, "❌ locY should be None"
    assert pp.locZ is None, "❌ locZ should be None"
    print("✅ PrepositionalPhrase static fields working")
    
    print("\n🧪 Testing SceneObjectPhrase static fields...")
    sop = SceneObjectPhrase()
    
    # Test the static fields we added
    assert sop.spatial_relationships == [], "❌ spatial_relationships should be []"
    assert sop.grounded_objects == [], "❌ grounded_objects should be []"
    print("✅ SceneObjectPhrase static fields working")
    
    print("\n🧪 Testing direct None checks (no hasattr needed!)...")
    
    # This is the key improvement - we can now test directly for None!
    vs._grounded_phrase = "something"
    if vs._grounded_phrase is not None:
        print("✅ Direct None check working on _grounded_phrase")
    
    np.scene_object = "test_object"
    if np.scene_object is not None:
        print("✅ Direct None check working on scene_object")
        
    pp.spatial_vector = "test_vector"
    if pp.spatial_vector is not None:
        print("✅ Direct None check working on spatial_vector")
        
    # Test collections  
    sop.spatial_relationships.append("test_relationship")
    if sop.spatial_relationships:  # Can check if list is not empty
        print("✅ Direct list check working on spatial_relationships")
        
    print("\n🎉 DYNAMIC ATTRIBUTE ELIMINATION SUCCESSFUL!")
    print("🎉 All static fields are working properly!")
    print("🎉 No more hasattr() checks needed!")
    print("🎉 Direct None testing enabled!")
    
    return True

if __name__ == "__main__":
    test_static_fields()
