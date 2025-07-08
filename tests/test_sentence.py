from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.scenes.scene_model import SceneModel, resolve_pronoun
from engraf.scenes.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace, vector_from_features, is_verb, is_tobe, is_determiner, \
    is_pronoun, any_of
from engraf.atn.subnet_sentence import run_sentence
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
    assert sentence is not None, "Failed to parse second sentence: 'the cube is blue'"  
    predicate = sentence.predicate
    assert predicate is not None
    assert predicate.verb == "is"
    predicate_np = predicate.noun_phrase
    assert predicate_np is not None
    assert predicate_np.noun == "cube"
