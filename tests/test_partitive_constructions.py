import pytest
from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.prepositional_phrase import PrepositionalPhrase


def test_partitive_constructions():
    """Test parsing partitive constructions like 'one of them', 'two of them', etc."""
    test_cases = [
        'draw one of them',
        'draw 1 of them',
        'draw two of them',
        'draw three of them',
        'draw four of them',
        'draw five of them'
    ]
    
    for phrase in test_cases:
        sentence = run_sentence(TokenStream(tokenize(phrase)))
        
        assert sentence is not None, f"Failed to parse: {phrase}"
        assert sentence.predicate.verb == "draw", f"Verb should be 'draw' for: {phrase}"
        
        # Check that the noun phrase has a prepositional phrase
        np = sentence.predicate.noun_phrase
        assert len(np.preps) == 1, f"Should have exactly 1 prepositional phrase for: {phrase}"
        
        # Check the prepositional phrase structure
        pp = np.preps[0]
        assert pp.preposition == "of", f"Preposition should be 'of' for: {phrase}"
        assert pp.noun_phrase.pronoun == "them", f"Object of preposition should be 'them' for: {phrase}"


def test_partitive_with_adjectives():
    """Test parsing partitive constructions with adjectives like 'draw one of the red cube'"""
    sentence = run_sentence(TokenStream(tokenize('draw one of the red cube')))
    
    assert sentence is not None, "Failed to parse 'draw one of the red cube'"
    assert sentence.predicate.verb == "draw", "Verb should be 'draw'"
    
    # Check that the noun phrase has a prepositional phrase
    np = sentence.predicate.noun_phrase
    assert len(np.preps) == 1, "Should have exactly 1 prepositional phrase"
    
    # Check the prepositional phrase structure
    pp = np.preps[0]
    assert pp.preposition == "of", "Preposition should be 'of'"
    assert pp.noun_phrase.noun == "cube", "Object noun should be 'cube'"
    assert pp.noun_phrase.determiner == "the", "Object determiner should be 'the'"
    assert pp.noun_phrase.vector["red"] == 1.0, "Object should be red"


def test_partitive_with_numbers():
    """Test parsing partitive constructions with explicit numbers"""
    test_cases = [
        ('draw 1 of them', None),  # "1" might not be recognized as a determiner
        ('draw 2 of them', None),
        ('draw 10 of them', None)
    ]
    
    for phrase, expected_determiner in test_cases:
        sentence = run_sentence(TokenStream(tokenize(phrase)))
        
        assert sentence is not None, f"Failed to parse: {phrase}"
        assert sentence.predicate.verb == "draw", f"Verb should be 'draw' for: {phrase}"
        
        # Check that the noun phrase has a prepositional phrase
        np = sentence.predicate.noun_phrase
        assert len(np.preps) == 1, f"Should have exactly 1 prepositional phrase for: {phrase}"
        
        # Check the prepositional phrase structure
        pp = np.preps[0]
        assert pp.preposition == "of", f"Preposition should be 'of' for: {phrase}"
        assert pp.noun_phrase.pronoun == "them", f"Object of preposition should be 'them' for: {phrase}"


def test_partitive_nested():
    """Test parsing nested partitive constructions"""
    sentence = run_sentence(TokenStream(tokenize('draw one of the cube on the table')))
    
    assert sentence is not None, "Failed to parse nested partitive construction"
    assert sentence.predicate.verb == "draw", "Verb should be 'draw'"
    
    # Check that the main noun phrase has a prepositional phrase
    np = sentence.predicate.noun_phrase
    assert len(np.preps) == 1, "Should have exactly 1 prepositional phrase"
    
    # Check the first prepositional phrase structure
    pp = np.preps[0]
    assert pp.preposition == "of", "First preposition should be 'of'"
    assert pp.noun_phrase.noun == "cube", "Object noun should be 'cube'"
    assert pp.noun_phrase.determiner == "the", "Object determiner should be 'the'"
    
    # Check that the nested noun phrase also has a prepositional phrase
    nested_np = pp.noun_phrase
    assert len(nested_np.preps) == 1, "Nested noun phrase should have exactly 1 prepositional phrase"
    
    nested_pp = nested_np.preps[0]
    assert nested_pp.preposition == "on", "Nested preposition should be 'on'"
    assert nested_pp.noun_phrase.noun == "table", "Final object noun should be 'table'"
    assert nested_pp.noun_phrase.determiner == "the", "Final object determiner should be 'the'"
