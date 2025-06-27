from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_vp import run_vp
from engraf.scenes.scene_model import SceneModel, resolve_pronoun
from engraf.scenes.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from pprint import pprint


def test_draw_and_color():
    scene = SceneModel()

    # First sentence: draw a red cube
    tokens1 = TokenStream(tokenize("draw a red cube"))
    result1 = run_vp(tokens1)
    assert result1["verb"] == "draw"

    # Add object to scene
    obj1 = SceneObject(name=result1["object"], vector=result1["vector"])
    scene.add_object(obj1)

    # Verify initial color is red
    assert obj1.vector["red"] == 1.0
    assert obj1.vector["green"] == 0.0
    assert obj1.vector["blue"] == 0.0

    # Second sentence: color it green
    tokens2 = TokenStream(tokenize("color it green"))
    result2 = run_vp(tokens2)
    assert result2 is not None, "Failed to parse second sentence: 'color it green'"
    assert result2["verb"] == "color"
    
    pronoun = result2["noun_phrase"]["pronoun"]
    targets = resolve_pronoun(pronoun, scene)
    assert len(targets) == 1

    # Apply color from parsed NP
    new_color_vec = result2["noun_phrase"]["vector"]
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
    result1 = run_vp(tokens1)
    assert result1["verb"] == "draw"

    # Add object to scene
    obj1 = SceneObject(name=result1["object"], vector=result1["vector"])
    scene.add_object(obj1)

    # Verify initial color is red
    assert obj1.vector["red"] == 1.0
    assert obj1.vector["green"] == 0.0
    assert obj1.vector["blue"] == 0.0

    # Second sentence: draw a blue sphere
    tokens2 = TokenStream(tokenize("draw a blue sphere"))
    result2 = run_vp(tokens2)
    assert result2["verb"] == "draw"

    # Add second object to scene
    obj2 = SceneObject(name=result2["object"], vector=result2["vector"])
    scene.add_object(obj2)

    # Verify initial color is blue
    assert obj2.vector["red"] == 0.0
    assert obj2.vector["green"] == 0.0
    assert obj2.vector["blue"] == 1.0

    # Third sentence: color them green
    tokens3 = TokenStream(tokenize("color them green"))
    result3 = run_vp(tokens3)
    assert result3 is not None, "Failed to parse third sentence: 'color them green'"
    assert result3["verb"] == "color"
    pronoun = result3["noun_phrase"]["pronoun"]
   
    targets = resolve_pronoun(pronoun, scene)
    assert len(targets) == 2

    # Apply color from parsed NP to both objects
    new_color_vec = result3["noun_phrase"]["vector"]
    for target in targets:
        for channel in ("red", "green", "blue"):
            target.vector[channel] = new_color_vec[channel]

    # Verify updated colors
    for target in targets:
        assert target.vector["red"] == 0.0
        assert target.vector["green"] == 1.0
        assert target.vector["blue"] == 0.0