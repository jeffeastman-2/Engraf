from pprint import pprint

class SceneObject:
    def __init__(self, name, vector, modifiers=None):
        self.name = name                  # e.g., 'cube'
        self.vector = vector              # VectorSpace instance
        self.modifiers = modifiers or []  # nested SceneObjects from PPs

    def __repr__(self):
        return f"<{self.name} {self.vector} modifiers={self.modifiers}>"

def scene_object_from_np(noun_phrase):
    pprint(f"🟢 scene from NP = {noun_phrase}")
    preps = noun_phrase.preps
    obj = SceneObject(
        name=noun_phrase.noun,
        vector=noun_phrase.vector,
        modifiers=[
            SceneObject(
                name=pp.noun_phrase.noun,
                vector=pp.noun_phrase.vector,
                modifiers=pp.noun_phrase.preps)
            for pp in preps
        ]
    )
    return obj