"""
Assembly Creator Handler

This module handles the creation of SceneAssemblies from existing scene objects
through grouping operations like "group them as an 'arch'".
"""

from typing import List, Dict, Any, Optional
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_assembly import SceneAssembly
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS


class AssemblyCreator:
    """
    Handles the creation of assemblies from existing scene objects.
    """
    
    def __init__(self, scene, assembly_counter_ref, object_resolver):
        """
        Initialize the assembly creator.
        
        Args:
            scene: The scene model to add assemblies to
            assembly_counter_ref: Reference to the assembly counter (list with single int)
            object_resolver: The object resolver for finding target objects
        """
        self.scene = scene
        self.assembly_counter_ref = assembly_counter_ref
        self.object_resolver = object_resolver
    
    def create_assembly_from_verb_phrase(self, vp: VerbPhrase) -> Optional[str]:
        """
        Create an assembly from a grouping verb phrase.
        
        Examples:
        - "group them as an 'arch'"
        - "group the red cube and blue sphere as a 'tower'"
        - "group them"
        
        Returns:
            Assembly ID if successful, None otherwise
        """
        try:
            # Step 1: Resolve target objects to group
            target_objects = self._resolve_grouping_targets(vp)
            if not target_objects:
                print("⚠️  No objects found to group")
                return None
            
            # Step 2: Extract assembly name from verb phrase
            assembly_name = self._extract_assembly_name(vp)
            if not assembly_name:
                assembly_name = "assembly"  # Default name
            
            # Step 3: Create the assembly
            assembly_id = self._create_assembly(assembly_name, target_objects)
            
            # Step 4: Move objects from scene to assembly
            self._move_objects_to_assembly(target_objects, assembly_id)
            
            return assembly_id
            
        except Exception as e:
            print(f"❌ Error creating assembly: {e}")
            return None
    
    def _resolve_grouping_targets(self, vp: VerbPhrase) -> List[SceneObject]:
        """Resolve which objects to group based on the verb phrase."""
        target_objects = []
        
        # Use the existing object resolver to handle pronouns and object references
        target_object_ids = self.object_resolver.resolve_target_objects(vp)
        
        # Convert object IDs to SceneObject instances
        for obj_id in target_object_ids:
            scene_obj = self.scene.find_object_by_id(obj_id)
            if scene_obj:
                target_objects.append(scene_obj)
            
        # If no specific targets found, fall back to recent objects
        if not target_objects:
            # As fallback, get recent objects
            recent_objects = self.scene.get_recent_objects(count=None)  # Get all recent
            target_objects = recent_objects[:5]  # Limit to avoid accidentally grouping everything
        
        return target_objects
    
    def _extract_assembly_name(self, vp: VerbPhrase) -> Optional[str]:
        """Extract the assembly name from phrases like 'as an arch' or 'as a tower'."""
        # Look for prepositional phrases with "as"
        if hasattr(vp, 'prep_phrases') and vp.prep_phrases:
            for prep_phrase in vp.prep_phrases:
                if hasattr(prep_phrase, 'preposition') and prep_phrase.preposition == 'as':
                    if hasattr(prep_phrase, 'noun_phrase') and prep_phrase.noun_phrase:
                        if hasattr(prep_phrase.noun_phrase, 'noun'):
                            return prep_phrase.noun_phrase.noun
        
        # Look for quoted strings in the original sentence (would need access to raw text)
        # This is a simplified approach - could be enhanced
        
        return None
    
    def _create_assembly(self, name: str, objects: List[SceneObject]) -> str:
        """Create a new SceneAssembly with the given objects."""
        self.assembly_counter_ref[0] += 1
        assembly_id = f"{name}_{self.assembly_counter_ref[0]}"
        
        # Create the assembly (SceneAssembly doesn't accept vector parameter)
        assembly = SceneAssembly(
            name=name,
            assembly_id=assembly_id,
            objects=objects.copy()  # Start with copies of the objects
        )
        
        # Add assembly to scene
        self.scene.add_assembly(assembly)
        
        print(f"📦 Created assembly '{assembly_id}' with {len(objects)} objects")
        return assembly_id
    
    def _move_objects_to_assembly(self, objects: List[SceneObject], assembly_id: str):
        """Move objects from standalone scene to the assembly."""
        for obj in objects:
            try:
                # Use scene's move_object_to_assembly method
                success = self.scene.move_object_to_assembly(obj.object_id, assembly_id)
                if success:
                    print(f"  ↳ Moved {obj.object_id} to assembly")
                else:
                    print(f"  ⚠️  Failed to move {obj.object_id} to assembly")
            except Exception as e:
                print(f"  ❌ Error moving {obj.object_id}: {e}")
