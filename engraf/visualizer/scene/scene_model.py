from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from typing import List
import copy


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
    
    def copy(self):
        """
        Create a deep copy of the scene model.
        
        Returns:
            SceneModel: A new SceneModel instance with deep copies of all objects
        """
        new_scene = SceneModel()
        
        # Deep copy all objects
        new_scene.objects = [copy.deepcopy(obj) for obj in self.objects]
        
        # Deep copy recent objects list
        # We need to map the old objects to new objects to maintain references
        if self.recent:
            obj_mapping = {id(old_obj): new_obj for old_obj, new_obj in zip(self.objects, new_scene.objects)}
            new_scene.recent = [obj_mapping.get(id(obj), copy.deepcopy(obj)) for obj in self.recent]
        
        return new_scene
    

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
