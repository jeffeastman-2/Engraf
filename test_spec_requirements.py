#!/usr/bin/env python3
"""
Test script to verify the spec requirements for coordinated phrases in ENGRAF.

This tests the specific examples mentioned in the specification:
1. "The blue box and the green sphere are tall" - Subject coordination
2. "Draw a blue box and a green sphere" - Imperative with coordinated objects
"""

from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.verb_phrase import VerbPhrase

def test_subject_coordination():
    """Test parsing 'The blue box and the green sphere are tall'"""
    print("Testing subject coordination...")
    tokens = TokenStream(tokenize("The blue box and the green sphere are tall"))
    sentence = run_sentence(tokens)
    
    print(f"Parsed sentence: {sentence}")
    
    if sentence is None:
        print("❌ Failed to parse sentence")
        return False
    
    # Check if subject is a conjunction
    if not isinstance(sentence.subject, ConjunctionPhrase):
        print(f"❌ Subject should be ConjunctionPhrase, got {type(sentence.subject)}")
        return False
    
    # Check that the tobe is "are" (this is a copular construction)
    if sentence.tobe != "are":
        print(f"❌ Expected 'are' in tobe, got '{sentence.tobe}'")
        return False
    
    # Check that we have two noun phrases in the subject
    subjects = list(sentence.subject.flatten())
    if len(subjects) != 2:
        print(f"❌ Should have 2 subjects, got {len(subjects)}")
        return False
    
    # Check first NP: "the blue box"
    first_np = subjects[0]
    if not isinstance(first_np, NounPhrase):
        print(f"❌ First subject should be NounPhrase, got {type(first_np)}")
        return False
    
    if first_np.noun != "box":
        print(f"❌ First subject noun should be 'box', got '{first_np.noun}'")
        return False
    
    # Check second NP: "the green sphere"
    second_np = subjects[1]
    if not isinstance(second_np, NounPhrase):
        print(f"❌ Second subject should be NounPhrase, got {type(second_np)}")
        return False
    
    if second_np.noun != "sphere":
        print(f"❌ Second subject noun should be 'sphere', got '{second_np.noun}'")
        return False
    
    print("✅ Subject coordination test passed")
    return True

def test_imperative_coordinated_objects():
    """Test parsing 'Draw a blue box and a green sphere'"""
    print("\nTesting imperative with coordinated objects...")
    tokens = TokenStream(tokenize("Draw a blue box and a green sphere"))
    sentence = run_sentence(tokens)
    
    print(f"Parsed sentence: {sentence}")
    
    if sentence is None:
        print("❌ Failed to parse sentence")
        return False
    
    # Check this is an imperative (no subject)
    if sentence.subject is not None:
        print(f"❌ Imperative should have no subject, got {sentence.subject}")
        return False
    
    # Check the verb is "draw"
    if sentence.predicate.verb != "draw":
        print(f"❌ Verb should be 'draw', got '{sentence.predicate.verb}'")
        return False
    
    # Check that the object is a conjunction
    if not isinstance(sentence.predicate.noun_phrase, ConjunctionPhrase):
        print(f"❌ Object should be ConjunctionPhrase, got {type(sentence.predicate.noun_phrase)}")
        return False
    
    # Check that we have two noun phrases in the object
    objects = list(sentence.predicate.noun_phrase.flatten())
    if len(objects) != 2:
        print(f"❌ Should have 2 objects, got {len(objects)}")
        return False
    
    # Check first NP: "a blue box"
    first_obj = objects[0]
    if not isinstance(first_obj, NounPhrase):
        print(f"❌ First object should be NounPhrase, got {type(first_obj)}")
        return False
    
    if first_obj.noun != "box":
        print(f"❌ First object noun should be 'box', got '{first_obj.noun}'")
        return False
    
    # Check second NP: "a green sphere"
    second_obj = objects[1]
    if not isinstance(second_obj, NounPhrase):
        print(f"❌ Second object should be NounPhrase, got {type(second_obj)}")
        return False
    
    if second_obj.noun != "sphere":
        print(f"❌ Second object noun should be 'sphere', got '{second_obj.noun}'")
        return False
    
    print("✅ Imperative coordinated objects test passed")
    return True

def test_recursive_conjunctions():
    """Test parsing 'Draw a red cube and a blue sphere and a green pyramid'"""
    print("\nTesting recursive conjunctions...")
    tokens = TokenStream(tokenize("Draw a red cube and a blue sphere and a green pyramid"))
    sentence = run_sentence(tokens)
    
    print(f"Parsed sentence: {sentence}")
    
    if sentence is None:
        print("❌ Failed to parse sentence")
        return False
    
    # Check that the object is a conjunction
    if not isinstance(sentence.predicate.noun_phrase, ConjunctionPhrase):
        print(f"❌ Object should be ConjunctionPhrase, got {type(sentence.predicate.noun_phrase)}")
        return False
    
    # Check that we have three noun phrases in the object
    objects = list(sentence.predicate.noun_phrase.flatten())
    if len(objects) != 3:
        print(f"❌ Should have 3 objects, got {len(objects)}")
        return False
    
    # Check the nouns
    expected_nouns = ["cube", "sphere", "pyramid"]
    for i, obj in enumerate(objects):
        if obj.noun != expected_nouns[i]:
            print(f"❌ Object {i+1} should be '{expected_nouns[i]}', got '{obj.noun}'")
            return False
    
    print("✅ Recursive conjunctions test passed")
    return True

def main():
    print("=== Testing ENGRAF Coordinated Phrases Specification ===\n")
    
    tests = [
        test_subject_coordination,
        test_imperative_coordinated_objects,
        test_recursive_conjunctions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Results: {sum(results)}/{len(results)} tests passed ===")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Implementation needs work.")

if __name__ == "__main__":
    main()
