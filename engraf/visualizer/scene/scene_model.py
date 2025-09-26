from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.scene.scene_assembly import SceneAssembly
from engraf.visualizer.scene.scene_entity import SceneEntity
from engraf.lexer.vector_space import VectorSpace
from typing import List, Optional, Union
import copy


class SceneModel:
    def __init__(self):
        self.entities = []       # Unified list of SceneEntity objects (both objects and assemblies)
        self.recent = []         # Recent entities
        
        # Deprecated - kept for backward compatibility during transition
        self._objects = []       # Will be removed after refactoring
        self._assemblies = []    # Will be removed after refactoring
    
    @property
    def objects(self) -> List[SceneObject]:
        """Get all SceneObject entities (backward compatibility)."""
        return [entity for entity in self.entities if isinstance(entity, SceneObject)]
    
    @property
    def assemblies(self) -> List[SceneAssembly]:
        """Get all SceneAssembly entities (backward compatibility)."""
        return [entity for entity in self.entities if isinstance(entity, SceneAssembly)]

    def add_object(self, obj: SceneObject):
        """Add a SceneObject to the scene."""
        self.entities.append(obj)
        self.recent = [obj]

    def add_assembly(self, assembly: SceneAssembly):
        """Add a SceneAssembly to the scene."""
        self.entities.append(assembly)
        self.recent = [assembly]
    
    def add_entity(self, entity: SceneEntity):
        """Add any SceneEntity to the scene."""
        self.entities.append(entity)
        self.recent = [entity]

    def __repr__(self):
        """String representation showing all entities."""
        lines = []
        objects = self.objects
        assemblies = self.assemblies
        
        if objects:
            lines.append("Objects:")
            lines.extend("  " + repr(obj) for obj in objects)
        if assemblies:
            lines.append("Assemblies:")
            lines.extend("  " + repr(assembly) for assembly in assemblies)
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
    
    def find_entity_by_id(self, entity_id: str) -> Optional[SceneEntity]:
        """Find any SceneEntity by ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def find_object_by_id(self, object_id: str) -> Optional[SceneObject]:
        """Find a SceneObject by ID, searching both standalone and assembly objects."""
        # Search standalone objects
        for entity in self.entities:
            if isinstance(entity, SceneObject) and entity.object_id == object_id:
                return entity
        
        # Search objects within assemblies
        for entity in self.entities:
            if isinstance(entity, SceneAssembly):
                for obj in entity.objects:
                    if obj.object_id == object_id:
                        return obj
        
        return None

    def find_assembly_by_id(self, assembly_id: str) -> Optional[SceneAssembly]:
        """Find a SceneAssembly by ID."""
        for entity in self.entities:
            if isinstance(entity, SceneAssembly) and entity.assembly_id == assembly_id:
                return entity
        return None

    def find_assembly_by_name(self, name: str) -> Optional[SceneAssembly]:
        """Find a SceneAssembly by name."""
        for entity in self.entities:
            if isinstance(entity, SceneAssembly) and entity.name == name:
                return entity
        return None

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene (handles both standalone and assembly objects)."""
        # Try to remove from standalone objects
        for entity in self.entities:
            if isinstance(entity, SceneObject) and entity.object_id == object_id:
                self.entities.remove(entity)
                return True
        
        # Try to remove from assemblies
        for entity in self.entities:
            if isinstance(entity, SceneAssembly):
                for obj in entity.objects:
                    if obj.object_id == object_id:
                        entity.remove_object(obj)
                        return True
        
        return False

    def remove_assembly(self, assembly_id: str) -> bool:
        """Remove an entire assembly from the scene."""
        for entity in self.entities:
            if isinstance(entity, SceneAssembly) and entity.assembly_id == assembly_id:
                self.entities.remove(entity)
                return True
        return False
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove any entity from the scene by ID."""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                self.entities.remove(entity)
                return True
        return False

    def move_object_to_assembly(self, object_id: str, assembly_id: str) -> bool:
        """Move a standalone object into an assembly."""
        obj = self.find_object_by_id(object_id)
        assembly = self.find_assembly_by_id(assembly_id)
        
        if obj and assembly and obj in self.entities:
            self.entities.remove(obj)
            assembly.add_object(obj)
            return True
        
        return False

    def extract_object_from_assembly(self, object_id: str) -> bool:
        """Extract an object from its assembly and make it standalone."""
        for assembly in self.assemblies:
            for obj in assembly.objects:
                if obj.object_id == object_id:
                    assembly.remove_object(obj)
                    self.entities.append(obj)
                    return True
        
        return False
        
    def clear(self):
        """Clear all entities and recent items from the scene."""
        self.entities.clear()
        self.recent.clear()
        
    def find_noun_phrase(self, np, return_all_matches=True):
        """
        Given a noun phrase context, try to find the most relevant SceneObject or SceneAssembly.
        First searches assemblies, then individual objects.
        
        IMPORTANT: Requires EXACT noun name match first (e.g., "sphere" only matches spheres),
        then uses semantic similarity on attributes within objects of the same type.
        This prevents cross-type matching like "red sphere" matching "red box".
        
        Args:
            np: The NounPhrase to resolve
            return_all_matches: If True, returns all matching objects for LATN hypothesis generation.
                               If False, returns only the best match (legacy behavior).
        
        Returns:
            If return_all_matches=False: Single best SceneObject/Assembly or None
            If return_all_matches=True: List of (similarity, object) tuples sorted by similarity
        """
        noun = np.noun
        vector = np.vector

        candidates = []

        # Search assemblies first (they have precedence as compound nouns)
        for assembly in self.assemblies:
            if noun and assembly.name != noun:
                continue  # Filter by assembly name

            # If a vector is provided, compute semantic similarity
            if vector:
                similarity = vector.semantic_similarity(assembly.vector)
                candidates.append((similarity, assembly))
            else:
                candidates.append((1.0, assembly))  # perfect match by name

        # Search individual objects
        for obj in self.objects:
            # Handle "object" as universal shape matcher - matches any object
            if noun and noun != "object" and obj.name != noun:
                continue  # Filter by object name

            # If a vector is provided, compute semantic similarity
            if vector:
                similarity = vector.semantic_similarity(obj.vector)
                candidates.append((similarity, obj))
            else:
                candidates.append((1.0, obj))  # perfect match by name

        # Also search objects within assemblies
        for assembly in self.assemblies:
            for obj in assembly.objects:
                # Handle "object" as universal shape matcher - matches any object
                if noun and noun != "object" and obj.name != noun:
                    continue  # Filter by object name

                # If a vector is provided, compute semantic similarity
                if vector:
                    similarity = vector.semantic_similarity(obj.vector)
                    candidates.append((similarity, obj))
                else:
                    candidates.append((1.0, obj))  # perfect match by name

        if not candidates:
            return [] if return_all_matches else None

        # Sort by similarity (highest first)
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        
        if return_all_matches:
            # For LATN: return all candidates that have meaningful similarity (> 0)
            # or if no semantic constraints, return all name matches
            if vector:
                # Only return matches with positive semantic similarity
                return [(sim, obj) for sim, obj in candidates if sim > 0]
            else:
                # No semantic constraints - return all name matches
                return candidates
        else:
            # Legacy behavior: return single best match
            return candidates[0][1]
    
    def copy(self):
        """
        Create a deep copy of the scene model.
        
        Returns:
            SceneModel: A new SceneModel instance with deep copies of all objects and assemblies
        """
        new_scene = SceneModel()
        
        # Deep copy all entities directly to the unified list
        new_scene.entities = [copy.deepcopy(entity) for entity in self.entities]
        
        # Deep copy recent objects/assemblies list
        if self.recent:
            # Create mapping from old to new entities
            entity_mapping = {id(old_entity): new_entity for old_entity, new_entity in zip(self.entities, new_scene.entities)}
            
            new_recent = []
            for item in self.recent:
                new_recent.append(entity_mapping.get(id(item), copy.deepcopy(item)))
            new_scene.recent = new_recent
        
        return new_scene
    

def resolve_pronoun(word, scene: SceneModel):
    """Resolve pronouns to entities in the scene."""
    word = word.lower()
    if word == "it":
        if scene.recent:
            return [scene.recent[-1]]  # Return the most recently added entity
        else:
            return []  # Return empty list if no entities
    elif word in ("they", "them"):
        # Return all known entities in the scene
        return scene.entities.copy()
    else:
        raise ValueError(f"Unrecognized pronoun: {word}")
