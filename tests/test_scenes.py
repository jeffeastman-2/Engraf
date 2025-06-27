from engraf.scenes.scene_object import SceneObject 
from engraf.scenes.scene_model import SceneModel
from engraf.scenes.scene_model import scene_from_parse
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.token_stream import TokenStream
from engraf.atn.subnet_vp import run_vp
from engraf.lexer.token_stream import tokenize
from pprint import pprint

def test_scene_from_simple_vp():
    tokens = TokenStream(tokenize("draw a tall blue cube over the green sphere"))
    parse_tree = run_vp(tokens)
    scene = scene_from_parse(parse_tree)

    # Check the scene has one object
    assert isinstance(scene, SceneModel)
    assert len(scene.objects) == 1

    obj = scene.objects[0]
    assert isinstance(obj, SceneObject)
    assert obj.name == "cube"
    assert isinstance(obj.vector, VectorSpace)

    # Ensure the object has one modifier
    assert obj.modifiers is not None
    assert len(obj.modifiers) == 1

    mod = obj.modifiers[0]
    assert isinstance(mod, SceneObject)
    assert mod.name == "sphere"
    assert isinstance(mod.vector, VectorSpace)

    # Example: check that the sphere is green
    assert mod.vector["green"] > 0.5
    # Example: check that the cube is tall and blue
    assert obj.vector["scaleY"] > 1.0
    assert obj.vector["blue"] > 0.5
