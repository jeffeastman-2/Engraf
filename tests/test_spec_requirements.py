#!/usr/bin/env python3
"""
Test script to verify the spec requirements for coordinated phrases in ENGRAF.

This tests the specific examples mentioned in the specification:
1. "The blue box and the green sphere are tall" - Subject coordination
2. "Draw a blue box and a green sphere" - Imperative with coordinated objects
"""

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.verb_phrase import VerbPhrase


def parse_sentence(text):
    """Helper to parse a sentence using LATNLayerExecutor and return the SentencePhrase."""
    executor = LATNLayerExecutor()
    result = executor.execute_layer5(text)
    if result.success and result.hypotheses:
        for tok in result.hypotheses[0].tokens:
            if hasattr(tok, 'phrase') and tok.phrase is not None:
                return tok.phrase
    return None


def test_subject_coordination():
    """Test parsing 'The blue box and the green sphere are tall'"""
    sentence = parse_sentence("The blue box and the green sphere are tall")
    
    # Check if sentence was parsed successfully
    assert sentence is not None, "Failed to parse sentence"
    
    # Check if subject is a conjunction
    assert isinstance(sentence.subject, ConjunctionPhrase), \
        f"Subject should be ConjunctionPhrase, got {type(sentence.subject)}"
    
    # Check that the verb is "are" (this is a copular construction)
    assert sentence.predicate.verb == "are", \
        f"Expected 'are' in predicate.verb, got '{sentence.predicate.verb}'"
    
    # Check that we have two noun phrases in the subject
    subjects = sentence.subject.phrases
    assert len(subjects) == 2, \
        f"Should have 2 subjects, got {len(subjects)}"
    
    # Check first NP: "the blue box"
    first_np = subjects[0]
    assert isinstance(first_np, NounPhrase), \
        f"First subject should be NounPhrase, got {type(first_np)}"
    
    assert first_np.noun == "box", \
        f"First subject noun should be 'box', got '{first_np.noun}'"
    
    # Check second NP: "the green sphere"
    second_np = subjects[1]
    assert isinstance(second_np, NounPhrase), \
        f"Second subject should be NounPhrase, got {type(second_np)}"
    
    assert second_np.noun == "sphere", \
        f"Second subject noun should be 'sphere', got '{second_np.noun}'"

def test_imperative_coordinated_objects():
    """Test parsing 'Draw a blue box and a green sphere'"""
    sentence = parse_sentence("Draw a blue box and a green sphere")
    
    # Check if sentence was parsed successfully
    assert sentence is not None, "Failed to parse sentence"
    
    # Check this is an imperative (no subject)
    assert sentence.subject is None, \
        f"Imperative should have no subject, got {sentence.subject}"
    
    # Check the verb is "draw" (case-insensitive)
    assert sentence.predicate.verb.lower() == "draw", \
        f"Verb should be 'draw', got '{sentence.predicate.verb}'"
    
    # Check that the object is a conjunction
    assert isinstance(sentence.predicate.noun_phrase, ConjunctionPhrase), \
        f"Object should be ConjunctionPhrase, got {type(sentence.predicate.noun_phrase)}"
    
    # Check that we have two noun phrases in the object
    objects = sentence.predicate.noun_phrase.phrases
    assert len(objects) == 2, \
        f"Should have 2 objects, got {len(objects)}"
    
    # Check first NP: "a blue box"
    first_obj = objects[0]
    assert isinstance(first_obj, NounPhrase), \
        f"First object should be NounPhrase, got {type(first_obj)}"
    
    assert first_obj.noun == "box", \
        f"First object noun should be 'box', got '{first_obj.noun}'"
    
    # Check second NP: "a green sphere"
    second_obj = objects[1]
    assert isinstance(second_obj, NounPhrase), \
        f"Second object should be NounPhrase, got {type(second_obj)}"
    
    assert second_obj.noun == "sphere", \
        f"Second object noun should be 'sphere', got '{second_obj.noun}'"

def test_recursive_conjunctions():
    """Test parsing 'Draw a red cube and a blue sphere and a green pyramid'"""
    sentence = parse_sentence("Draw a red cube and a blue sphere and a green pyramid")
    
    # Check if sentence was parsed successfully
    assert sentence is not None, "Failed to parse sentence"
    
    # Check that the object is a conjunction
    assert isinstance(sentence.predicate.noun_phrase, ConjunctionPhrase), \
        f"Object should be ConjunctionPhrase, got {type(sentence.predicate.noun_phrase)}"
    
    # Check that we have three noun phrases in the object
    objects = sentence.predicate.noun_phrase.phrases
    assert len(objects) == 3, \
        f"Should have 3 objects, got {len(objects)}"
    
    # Check the nouns
    expected_nouns = ["cube", "sphere", "pyramid"]
    for i, obj in enumerate(objects):
        assert obj.noun == expected_nouns[i], \
            f"Object {i+1} should be '{expected_nouns[i]}', got '{obj.noun}'"

def main():
    """Legacy main function for standalone execution"""
    print("=== Testing ENGRAF Coordinated Phrases Specification ===\n")
    
    tests = [
        test_subject_coordination,
        test_imperative_coordinated_objects,
        test_recursive_conjunctions
    ]
    
    results = []
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} passed")
            results.append(True)
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Results: {sum(results)}/{len(results)} tests passed ===")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Implementation needs work.")

if __name__ == "__main__":
    main()
