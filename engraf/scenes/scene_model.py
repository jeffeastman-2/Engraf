from engraf.scenes.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from typing import List


class SceneModel:
    def __init__(self):
        self.objects = []
        self.recent = []

    def add_object(self, obj):
        self.objects.append(obj)

    def __repr__(self):
        return "\n".join(repr(obj) for obj in self.objects)

    def add_object(self, obj):
        self.objects.append(obj)
        self.recent = [obj]  # or append for plural

    def get_recent_objects(self, count=None):
        return self.recent if count is None else self.recent[-count:]
        

def resolve_pronoun(word, scene: SceneModel):
    word = word.lower()
    if word == "it":
        if scene.objects:
            return [scene.objects[-1]]  # Return the most recently added object
    elif word in ("they", "them"):
        return scene.objects  # Return all known objects in the scene
    else:
        raise ValueError(f"Unrecognized pronoun: {word}")

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
