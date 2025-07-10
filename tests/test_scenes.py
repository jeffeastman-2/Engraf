from engraf.scenes.scene_object import SceneObject, scene_object_from_np 
from engraf.scenes.scene_model import SceneModel, resolve_pronoun
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_sentence import run_sentence
from engraf.lexer.token_stream import tokenize

def test_scene_from_simple_sentence():
    tokens = TokenStream(tokenize("draw a tall blue cube over the green sphere"))
    sentence = run_sentence(tokens)
    assert sentence is not None
    assert sentence.subject is None
    assert sentence.predicate is not None
    np = sentence.predicate.noun_phrase
    assert np is not None
    scene = scene_object_from_np(np)

    assert isinstance(scene, SceneObject)
    assert scene.name == "cube"
    assert isinstance(scene.vector, VectorSpace)

    # Ensure the object has one modifier
    assert scene.modifiers is not None
    assert len(scene.modifiers) == 1

    modifier = scene.modifiers[0]
    assert isinstance(modifier, SceneObject)
    assert modifier.name == "sphere"
    assert isinstance(modifier.vector, VectorSpace)
    assert modifier.vector["green"] > 0.5

    # Example: check that the cube is tall and blue
    assert scene.vector["scaleY"] > 1.0
    assert scene.vector["blue"] > 0.5

def test_scene_from_sentence_with_chained_pps():
    tokens = TokenStream(tokenize("draw a tall blue cube over the green sphere by the very tall arch"))
    sentence = run_sentence(tokens)
    assert sentence is not None
    assert sentence.subject is None
    assert sentence.predicate is not None
    np = sentence.predicate.noun_phrase
    assert np is not None
    scene = scene_object_from_np(np)

    assert isinstance(scene, SceneObject)
    assert scene.name == "cube"
    assert isinstance(scene.vector, VectorSpace)

    # Ensure the object has one modifier
    assert scene.modifiers is not None
    assert len(scene.modifiers) == 2

    modifier = scene.modifiers[0]
    assert isinstance(modifier, SceneObject)
    assert modifier.name == "sphere"
    assert isinstance(modifier.vector, VectorSpace)
    assert modifier.vector["green"] > 0.5

    modifier = scene.modifiers[1]
    assert isinstance(modifier, SceneObject)
    assert modifier.name == "arch"
    assert isinstance(modifier.vector, VectorSpace)
    assert modifier.vector["scaleY"] > 2.0

    # Example: check that the cube is tall and blue
    assert scene.vector["scaleY"] > 1.0
    assert scene.vector["blue"] > 0.5

def test_scene_pronoun_resolution():
    from engraf.scenes.scene_model import SceneModel, resolve_pronoun
    from engraf.lexer.vector_space import VectorSpace

    scene = SceneModel()
    scene.add_object(SceneObject(name="cube", vector=VectorSpace()))
    scene.add_object(SceneObject(name="sphere", vector=VectorSpace()))

    assert len(scene.objects) == 2
    # Test singular pronoun resolution
    result = resolve_pronoun("it", scene)
    assert len(result) == 1
    assert result[0].name == "sphere"  # Last added object should be resolved

    # Test plural pronoun resolution
    result = resolve_pronoun("they", scene)
    assert len(result) == 2
    assert {obj.name for obj in result} == {"cube", "sphere"}  # Both objects should be resolved

def test_scene_draw_and_color():
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
    obj1 = SceneObject(name=predicate1_np.noun, vector=predicate1_np.vector)
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
    predicate2_np = predicate2.noun_phrase
    assert predicate2_np is not None
    pronoun = predicate2_np.pronoun
    targets = resolve_pronoun(pronoun, scene)
    assert len(targets) == 1

    # Apply color from parsed NP
    new_color_vec = predicate2_np.vector
    for channel in ("red", "green", "blue"):
        targets[0].vector[channel] = new_color_vec[channel]

    # Verify updated color
    assert targets[0].vector["red"] == 0.0
    assert targets[0].vector["green"] == 1.0
    assert targets[0].vector["blue"] == 0.0


def test_scene_draw_and_color_multiple_objects():
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
    obj1 = SceneObject(name=predicate1_np.noun, vector=predicate1_np.vector)
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


def test_scene_declarative_sentence():
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
    print(f"Sentence2 = {sentence2}")
    subject = sentence2.subject
    assert subject is not None
    assert sentence2.tobe == "is"

    # Find the target object in the scene   
    targets = scene.find_noun_phrase(subject)
    assert targets is not None, "Failed to find noun phrase in scene"