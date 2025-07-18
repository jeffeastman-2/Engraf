from pprint import pprint

class SceneObject:
    def __init__(self, name, vector, modifiers=None, object_id=None):
        self.name = name                  # e.g., 'cube' (the base noun)
        self.object_id = object_id or name  # e.g., 'red_cube_1' (unique identifier)
        self.vector = vector              # VectorSpace instance
        self.modifiers = modifiers or []  # nested SceneObjects from PPs
        self.metadata = {}                # Store additional metadata for matching

    def __repr__(self):
        return f"<{self.name} ({self.object_id}) {self.vector} modifiers={self.modifiers}>"

def scene_object_from_np(noun_phrase):
    from pprint import pprint
    pprint(f"🟢 scene from NP = {noun_phrase}")

    def flatten_modifiers(np):
        """Recursively extract all PPs from a noun phrase and return flat list of SceneObjects."""
        modifiers = []
        for pp in np.preps:
            mod_np = pp.noun_phrase
            if mod_np.noun is not None:
                modifiers.append(SceneObject(
                    name=mod_np.noun,
                    vector=mod_np.vector,
                    modifiers=[]  # We will flatten everything at this level
                ))
                # Recurse to grab their own PPs too
                modifiers.extend(flatten_modifiers(mod_np))
        return modifiers

    obj = SceneObject(
        name=noun_phrase.noun,
        vector=noun_phrase.vector,
        modifiers=flatten_modifiers(noun_phrase)
    )
    return obj
