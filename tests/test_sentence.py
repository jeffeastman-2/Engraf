from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.scenes.scene_model import SceneModel, resolve_pronoun
from engraf.scenes.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace, vector_from_features, is_verb, is_tobe, is_determiner, \
    is_pronoun, any_of
from engraf.atn.subnet_sentence import run_sentence
from pprint import pprint


def test_draw_and_color():
    scene = SceneModel()

    # First sentence: draw a red cube
    tokens1 = TokenStream(tokenize("draw a red cube"))
    sentence1 = run_sentence(tokens1)
    assert sentence1 is not None, "Failed to parse first sentence: 'draw a red cube'"
    predicate1 = sentence1.predicate
    assert predicate1 is not None
    assert predicate1.verb == "draw"
    predicate1_np = predicate1.noun_phrase
    assert predicate1_np is not None
    assert predicate1_np.noun == "cube"

    # Add object to scene
    obj1 = SceneObject(name=predicate_noun, vector=predicate_np.vector)
    scene.add_object(obj1)

    # Verify initial color is red
    assert obj1.vector["red"] == 1.0
    assert obj1.vector["green"] == 0.0
    assert obj1.vector["blue"] == 0.0

    # Second sentence: color it green
    tokens2 = TokenStream(tokenize("color it green"))
    sentence2 = run_sentence(tokens2)
    assert sentence2 is not None, "Failed to parse second sentence: 'color it green'"
    predicate2 = sentence2.predicate
    assert predicate2 is not None
    assert predicate2.verb == "color"
    predicate2_noun_phrase = predicate2.noun_phrase
    assert predicate2_noun_phrase is not None
    pronoun = predicate2_noun_phrase.pronoun
    targets = resolve_pronoun(pronoun, scene)
    assert len(targets) == 1

    # Apply color from parsed NP
    new_color_vec = sentence2["noun_phrase"]["vector"]
    for channel in ("red", "green", "blue"):
        targets[0].vector[channel] = new_color_vec[channel]

    # Verify updated color
    assert targets[0].vector["red"] == 0.0
    assert targets[0].vector["green"] == 1.0
    assert targets[0].vector["blue"] == 0.0


def test_draw_and_color_multiple_objects():
    scene = SceneModel()

    # First sentence: draw a red cube
    tokens1 = TokenStream(tokenize("draw a red cube"))
    sentence1 = run_sentence(tokens1)
    assert sentence1 is not None, "Failed to parse first sentence: 'draw a red cube'"
    predicate1 = sentence1.predicate
    assert predicate1 is not None
    assert predicate1.verb == "draw"
    predicate1_np = predicate1.noun_phrase
    assert predicate1_np is not None
    assert predicate1_np.noun == "cube"
    # Add object to scene
    obj1 = SceneObject(name=np.noun, vector=np.vector)
    scene.add_object(obj1)

    # Verify initial color is red
    assert obj1.vector["red"] == 1.0
    assert obj1.vector["green"] == 0.0
    assert obj1.vector["blue"] == 0.0

    # Second sentence: draw a blue sphere
    tokens2 = TokenStream(tokenize("draw a blue sphere"))
    sentence2 = run_sentence(tokens2)
    assert sentence2 is not None, "Failed to parse second sentence: 'draw a blue sphere'"
    predicate2 = sentence2.predicate
    assert predicate2 is not None
    assert predicate2.verb == "draw"
    predicate2_np = predicate2.noun_phrase
    assert predicate2_np is not None
 
    # Add second object to scene
    obj2 = SceneObject(name=predicate2_np.noun, vector=predicate2_np.vector)
    scene.add_object(obj2)

    # Verify initial color is blue
    assert obj2.vector["red"] == 0.0
    assert obj2.vector["green"] == 0.0
    assert obj2.vector["blue"] == 1.0

    # Third sentence: color them green
    tokens3 = TokenStream(tokenize("color them green"))
    sentence3 = run_sentence(tokens3)
    assert sentence3 is not None, "Failed to parse third sentence: 'color them green'"
    predicate3 = sentence3.predicate
    assert predicate3.verb == "color"
    predicate3_np = predicate3.noun_phrase
    assert predicate3_np is not None
    pronoun = predicate3_np.pronoun
   
    targets = resolve_pronoun(pronoun, scene)
    assert len(targets) == 2

    # Apply color from parsed NP to both objects
    new_color_vec = predicate3_np.vector
    for target in targets:
        for channel in ("red", "green", "blue"):
            target.vector[channel] = new_color_vec[channel]

    # Verify updated colors
    for target in targets:
        assert target.vector["red"] == 0.0
        assert target.vector["green"] == 1.0
        assert target.vector["blue"] == 0.0


def test_declarative_sentence():
    scene = SceneModel()

    # First sentence: draw a red cube
    tokens1 = TokenStream(tokenize("draw a red cube"))
    sentence1 = run_sentence(tokens1)
    assert sentence1 is not None
    predicate1 = sentence1.predicate
    assert predicate1.verb == "draw"
    predicate1_np = predicate1.noun_phrase
    assert predicate1_np is not None


    # Add object to scene
    obj1 = SceneObject(name=predicate1_np.noun, vector=predicate1_np.vector)
    scene.add_object(obj1)

    # Verify initial color is red
    assert obj1.vector["red"] == 1.0
    assert obj1.vector["green"] == 0.0
    assert obj1.vector["blue"] == 0.0

    # Second sentence: the cube is blue
    tokens2 = TokenStream(tokenize("the cube is blue"))
    sentence2 = run_sentence(tokens2)
    assert sentence2 is not None, "Failed to parse second sentence: 'the cube is blue'"  
    predicate2 = sentence2.predicate
    assert predicate2 is not None
    assert predicate2.verb == "is"
    predicate2_np = predicate2.noun_phrase
    assert predicate2_np is not None
    assert predicate2_np.noun == "cube"

    # Find the target object in the scene   
    targets = scene.find_noun_phrase(predicate2_np)
    assert targets is not None, "Failed to find noun phrase in scene"