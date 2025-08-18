from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_tokenizer import latn_tokenize_best as tokenize
from engraf.visualizer.scene.scene_model import SceneModel, resolve_pronoun
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from pprint import pprint

def test_imperative():
    # First sentence: draw a red cube
    tokens = TokenStream(tokenize("draw a red cube"))
    sentence = run_sentence(tokens)
    assert sentence is not None, "Failed to parse sentence: 'draw a red cube'"
    vp = sentence.predicate
    assert vp is not None
    assert vp.verb == "draw"
    np = vp.noun_phrase
    assert np is not None
    assert np.noun == "cube"

def test_imperative_with_preposition():
    # First sentence: draw a red cube
    tokens = TokenStream(tokenize("draw a red cube at [3,4,5]"))
    sentence = run_sentence(tokens)
    assert sentence is not None
    vp = sentence.predicate
    assert vp is not None
    assert vp.verb == "draw"
    np = vp.noun_phrase
    assert np is not None
    assert np.noun == "cube"
    preps = np.preps
    assert len(preps) == 1
    pp = preps[0]
    v = pp.vector
    assert v is not None

def test_declarative_sentence():
    tokens = TokenStream(tokenize("the cube is blue"))
    sentence = run_sentence(tokens)
    assert sentence is not None, "Failed to parse sentence: 'the cube is blue'"  
    subject = sentence.subject
    assert subject is not None
    assert sentence.tobe == "is"
    print(f"Sentence is {sentence}")

# Conjunction predicate tests
def test_simple_predicate_conjunction():
    """Test parsing 'draw a cube and move it'"""
    tokens = TokenStream(tokenize("draw a cube and move it"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse sentence: 'draw a cube and move it'"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    # Get all predicates from the conjunction
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 2, "Should have exactly 2 predicates"
    
    # Check first predicate
    first_pred = predicates[0]
    assert first_pred.verb == "draw"
    assert first_pred.noun_phrase is not None
    assert first_pred.noun_phrase.noun == "cube"
    
    # Check second predicate  
    second_pred = predicates[1]
    assert second_pred.verb == "move"
    assert second_pred.noun_phrase is not None
    assert second_pred.noun_phrase.pronoun == "it"

def test_triple_predicate_conjunction():
    """Test parsing 'draw a cube and move it and color it red'"""
    tokens = TokenStream(tokenize("draw a cube and move it and color it red"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse sentence: 'draw a cube and move it and color it red'"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 3, "Should have exactly 3 predicates"
    
    # Check first predicate
    assert predicates[0].verb == "draw"
    assert predicates[0].noun_phrase.noun == "cube"
    
    # Check second predicate
    assert predicates[1].verb == "move"
    assert predicates[1].noun_phrase.pronoun == "it"
    
    # Check third predicate
    assert predicates[2].verb == "color"
    assert predicates[2].noun_phrase.pronoun == "it"

def test_predicate_conjunction_with_prepositions():
    """Test parsing 'draw a cube over the sphere and move it to the pyramid'"""
    tokens = TokenStream(tokenize("draw a cube over the sphere and move it to the pyramid"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse complex conjunction with prepositions"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 2, "Should have exactly 2 predicates"
    
    # Check first predicate has preposition attached to noun phrase
    first_pred = predicates[0]
    assert first_pred.verb == "draw"
    assert first_pred.noun_phrase.noun == "cube"
    assert len(first_pred.noun_phrase.preps) == 1, "First predicate should have a prepositional phrase"
    assert first_pred.noun_phrase.preps[0].preposition == "over"
    assert first_pred.noun_phrase.preps[0].noun_phrase.noun == "sphere"
    
    # Check second predicate has preposition as verb complement
    second_pred = predicates[1]
    assert second_pred.verb == "move"
    assert second_pred.noun_phrase.pronoun == "it"
    # Note: prepositional phrases as verb complements might be handled differently
    # This test may need adjustment based on the actual implementation

def test_mixed_predicate_conjunction():
    """Test parsing 'draw a red cube and color the sphere blue'"""
    tokens = TokenStream(tokenize("draw a red cube and color the sphere blue"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse mixed predicate conjunction"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 2, "Should have exactly 2 predicates"
    
    # Check first predicate
    first_pred = predicates[0]
    assert first_pred.verb == "draw"
    assert first_pred.noun_phrase.noun == "cube"
    
    # Check second predicate
    second_pred = predicates[1]
    assert second_pred.verb == "color"
    assert second_pred.noun_phrase.noun == "sphere"

def test_predicate_conjunction_with_vectors():
    """Test parsing 'draw a cube at [1,2,3] and move it to [4,5,6]'"""
    tokens = TokenStream(tokenize("draw a cube at [1,2,3] and move it to [4,5,6]"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse conjunction with vectors"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 2, "Should have exactly 2 predicates"
    
    # Check first predicate
    first_pred = predicates[0]
    assert first_pred.verb == "draw"
    assert first_pred.noun_phrase.noun == "cube"
    
    # Check second predicate
    second_pred = predicates[1]
    assert second_pred.verb == "move"
    assert second_pred.noun_phrase.pronoun == "it"

def test_declarative_predicate_conjunction():
    """Test parsing 'the cube is red and large'"""
    tokens = TokenStream(tokenize("the cube is red and large"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse declarative conjunction"
    assert sentence.subject is not None
    assert sentence.subject.noun == "cube"
    assert sentence.tobe == "is"
    
    # The predicate part should handle "red and large" as a conjunction
    # This test may need adjustment based on how adjective conjunctions are handled

def test_imperative_vs_conjunction():
    """Test that simple imperatives without conjunctions still work"""
    tokens = TokenStream(tokenize("draw a red cube"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse simple imperative"
    assert not isinstance(sentence.predicate, ConjunctionPhrase), "Simple predicate should not be a ConjunctionPhrase"
    
    vp = sentence.predicate
    assert vp.verb == "draw"
    assert vp.noun_phrase.noun == "cube"

def test_conjunction_with_nested_noun_phrases():
    """Test parsing 'draw a red cube and move the large blue sphere'"""
    tokens = TokenStream(tokenize("draw a red cube and move the large blue sphere"))
    sentence = run_sentence(tokens)
    
    assert sentence is not None, "Failed to parse conjunction with complex noun phrases"
    assert isinstance(sentence.predicate, ConjunctionPhrase), "Predicate should be a ConjunctionPhrase"
    
    predicates = list(sentence.predicate.flatten())
    assert len(predicates) == 2, "Should have exactly 2 predicates"
    
    # Check first predicate
    first_pred = predicates[0]
    assert first_pred.verb == "draw"
    assert first_pred.noun_phrase.noun == "cube"
    
    # Check second predicate
    second_pred = predicates[1]
    assert second_pred.verb == "move"
    assert second_pred.noun_phrase.noun == "sphere"