from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.scene.scene_assembly import SceneAssembly
from engraf.lexer.vector_space import VectorSpace
from typing import List, Optional, Union
import copy


class SceneModel:
    def __init__(self):
        self.objects = []        # Individual SceneObjects not in assemblies
        self.assemblies = []     # SceneAssembly instances (contain their own objects)
        self.recent = []         # Recent objects or assemblies

    def add_object(self, obj):
        """Add a standalone SceneObject to the scene."""
        self.objects.append(obj)
        self.recent = [obj]

    def add_assembly(self, assembly: SceneAssembly):
        """Add a SceneAssembly to the scene."""
        self.assemblies.append(assembly)
        self.recent = [assembly]

    def __repr__(self):
        """String representation showing both objects and assemblies."""
        lines = []
        if self.objects:
            lines.append("Objects:")
            lines.extend("  " + repr(obj) for obj in self.objects)
        if self.assemblies:
            lines.append("Assemblies:")
            lines.extend("  " + repr(assembly) for assembly in self.assemblies)
        return "\n".join(lines) if lines else "Empty scene"

    def get_recent_objects(self, count=None):
        """Get recent objects/assemblies."""
        return self.recent if count is None else self.recent[-count:]

    def get_all_scene_objects(self) -> List[SceneObject]:
        """Get all SceneObjects, both standalone and in assemblies."""
        all_objects = self.objects.copy()
        for assembly in self.assemblies:
            all_objects.extend(assembly.objects)
        return all_objects

    def find_object_by_id(self, object_id: str) -> Optional[SceneObject]:
        """Find a SceneObject by ID, searching both standalone and assembly objects."""
        # Search standalone objects
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        
        # Search objects within assemblies
        for assembly in self.assemblies:
            for obj in assembly.objects:
                if obj.object_id == object_id:
                    return obj
        
        return None

    def find_assembly_by_id(self, assembly_id: str) -> Optional[SceneAssembly]:
        """Find a SceneAssembly by ID."""
        for assembly in self.assemblies:
            if assembly.assembly_id == assembly_id:
                return assembly
        return None

    def find_assembly_by_name(self, name: str) -> Optional[SceneAssembly]:
        """Find a SceneAssembly by name."""
        for assembly in self.assemblies:
            if assembly.name == name:
                return assembly
        return None

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene (handles both standalone and assembly objects)."""
        # Try to remove from standalone objects
        for obj in self.objects:
            if obj.object_id == object_id:
                self.objects.remove(obj)
                return True
        
        # Try to remove from assemblies
        for assembly in self.assemblies:
            for obj in assembly.objects:
                if obj.object_id == object_id:
                    assembly.remove_object(obj)
                    return True
        
        return False

    def remove_assembly(self, assembly_id: str) -> bool:
        """Remove an entire assembly from the scene."""
        for assembly in self.assemblies:
            if assembly.assembly_id == assembly_id:
                self.assemblies.remove(assembly)
                return True
        return False

    def move_object_to_assembly(self, object_id: str, assembly_id: str) -> bool:
        """Move a standalone object into an assembly."""
        obj = self.find_object_by_id(object_id)
        assembly = self.find_assembly_by_id(assembly_id)
        
        if obj and assembly and obj in self.objects:
            self.objects.remove(obj)
            assembly.add_object(obj)
            return True
        
        return False

    def extract_object_from_assembly(self, object_id: str) -> bool:
        """Extract an object from its assembly and make it standalone."""
        for assembly in self.assemblies:
            for obj in assembly.objects:
                if obj.object_id == object_id:
                    assembly.remove_object(obj)
                    self.objects.append(obj)
                    return True
        
        return False
        
    def find_noun_phrase(self, np):
        """
        Given a noun phrase context, try to find the most relevant SceneObject or SceneAssembly.
        First searches assemblies, then individual objects.
        """
        noun = np.noun
        vector = np.vector

        candidates = []

        # Search assemblies first (they have precedence as compound nouns)
        for assembly in self.assemblies:
            if noun and assembly.name != noun:
                continue  # Filter by assembly name

            # If a vector is provided, compute similarity
            if vector:
                similarity = assembly.vector.cosine_similarity(vector)
                candidates.append((similarity, assembly))
            else:
                candidates.append((1.0, assembly))  # perfect match by name

        # Search individual objects
        for obj in self.objects:
            if noun and obj.name != noun:
                continue  # Filter by object name

            # If a vector is provided, compute similarity
            if vector:
                similarity = obj.vector.cosine_similarity(vector)
                candidates.append((similarity, obj))
            else:
                candidates.append((1.0, obj))  # perfect match by name

        # Also search objects within assemblies
        for assembly in self.assemblies:
            for obj in assembly.objects:
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
            SceneModel: A new SceneModel instance with deep copies of all objects and assemblies
        """
        new_scene = SceneModel()
        
        # Deep copy all standalone objects
        new_scene.objects = [copy.deepcopy(obj) for obj in self.objects]
        
        # Deep copy all assemblies
        new_scene.assemblies = [copy.deepcopy(assembly) for assembly in self.assemblies]
        
        # Deep copy recent objects/assemblies list
        if self.recent:
            # Create mapping from old to new objects
            obj_mapping = {id(old_obj): new_obj for old_obj, new_obj in zip(self.objects, new_scene.objects)}
            assembly_mapping = {id(old_assembly): new_assembly for old_assembly, new_assembly in zip(self.assemblies, new_scene.assemblies)}
            
            new_recent = []
            for item in self.recent:
                if isinstance(item, SceneAssembly):
                    new_recent.append(assembly_mapping.get(id(item), copy.deepcopy(item)))
                else:
                    new_recent.append(obj_mapping.get(id(item), copy.deepcopy(item)))
            new_scene.recent = new_recent
        
        return new_scene
    

def resolve_pronoun(word, scene: SceneModel):
    """Resolve pronouns to objects or assemblies in the scene."""
    word = word.lower()
    if word == "it":
        if scene.recent:
            return [scene.recent[-1]]  # Return the most recently added object/assembly
        else:
            return []  # Return empty list if no objects
    elif word in ("they", "them"):
        # Return all known objects and assemblies in the scene
        all_items = scene.objects + scene.assemblies
        return all_items
    else:
        raise ValueError(f"Unrecognized pronoun: {word}")
