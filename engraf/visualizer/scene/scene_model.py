from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from typing import List


class SceneModel:
    def __init__(self):
        self.objects = []
        self.recent = []

    def add_object(self, obj):
        self.objects.append(obj)
        self.recent = [obj]  # or append for plural

    def __repr__(self):
        return "\n".join(repr(obj) for obj in self.objects)

    def get_recent_objects(self, count=None):
        return self.recent if count is None else self.recent[-count:]
        
    def find_noun_phrase(self, np):
        """
        Given a noun phrase context, try to find the most relevant SceneObject in the scene.
        ctx should contain at least 'noun' and optionally a 'vector'.
        """
        noun = np.noun
        vector = np.vector

        candidates = []

        for obj in self.objects:
            if noun and obj.name != noun:
                continue  # Filter by object name

            # If a vector is provided, compute similarity
            if vector:
                similarity = obj.vector.cosine_similarity(vector)
                candidates.append((similarity, obj))
            else:
                candidates.append((1.0, obj))  # perfect match by name

        if not candidates:
            return None

        # Return the best match by similarity
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]
    

def resolve_pronoun(word, scene: SceneModel):
    word = word.lower()
    if word == "it":
        if scene.objects:
            return [scene.objects[-1]]  # Return the most recently added object
        else:
            return []  # Return empty list if no objects
    elif word in ("they", "them"):
        return scene.objects  # Return all known objects in the scene
    else:
        raise ValueError(f"Unrecognized pronoun: {word}")
