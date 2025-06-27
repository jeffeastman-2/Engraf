from engraf.scenes.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from typing import List


class SceneModel:
    def __init__(self):
        self.objects = []  # List[SceneObject]

    def add_object(self, obj):
        self.objects.append(obj)

    def __repr__(self):
        return "\n".join(repr(obj) for obj in self.objects)

def scene_from_parse(parse_tree):
    scene = SceneModel()
    
    if parse_tree["verb"] == "draw":
        noun_data = parse_tree["noun_phrase"]
        obj = SceneObject(
            name=noun_data["noun"],
            vector=noun_data["vector"],
            modifiers=[
                SceneObject(
                    name=mod["object"],
                    vector=mod.get("vector", VectorSpace()),
                    modifiers=mod.get("modifiers", [])
                ) for mod in noun_data.get("modifiers", [])
            ]
        )
        scene.add_object(obj)
    return scene
