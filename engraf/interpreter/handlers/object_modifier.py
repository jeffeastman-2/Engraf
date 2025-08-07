"""
Object Modifier Handler

This module handles modifications to existing scene objects,
including transformations like movement, rotation, and scaling.
"""

from typing import List
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import VectorSpace


class ObjectModifier:
    """
    Handles modifications to existing scene objects.
    """
    
    def __init__(self, scene, renderer, object_resolver):
        """
        Initialize the object modifier.
        
        Args:
            scene: The scene model containing objects
            renderer: The renderer for visual updates
            object_resolver: The object resolver for finding targets
        """
        self.scene = scene
        self.renderer = renderer
        self.object_resolver = object_resolver
    
    def modify_scene_object(self, obj_id: str, vp: VerbPhrase) -> bool:
        """Modify a scene object based on a verb phrase."""
        print(f"🔧 _modify_scene_object called with obj_id: {obj_id}, verb: {vp.verb}")
        try:
            # Find the scene object
            scene_obj = None
            for obj in self.scene.objects:
                if obj.object_id == obj_id:
                    scene_obj = obj
                    break
            
            if not scene_obj:
                print(f"🔧 Scene object not found: {obj_id}")
                return False
            
            verb = vp.verb
            print(f"🔧 Processing verb: {verb}")
            
            # Check if verb phrase has vector space information
            if hasattr(vp, 'vector') and vp.vector:
                # Handle style verbs (color, texture, etc.) using vector space
                if vp.vector['style'] > 0.0 and hasattr(vp, 'adjective_complement'):
                    # Style the object - adjectives are already applied during ATN parsing
                    # so we just update the visual representation
                    pass
                
                # Handle transform verbs (move, rotate, scale) using vector space
                elif (vp.vector['move'] > 0.0 or vp.vector['rotate'] > 0.0 or vp.vector['scale'] > 0.0) and vp.noun_phrase:
                    if vp.noun_phrase.preps:
                        # Process prepositional phrases using semantic dimensions
                        for pp in vp.noun_phrase.preps:
                            # Check for movement using directional_target dimension OR spatial relationships
                            if hasattr(pp, 'vector') and (pp.vector['directional_target'] > 0.5 or abs(pp.vector['spatial_vertical']) > 0.5):
                                self._apply_movement(scene_obj, pp)
                            # Check for rotation/scaling using directional_agency dimension
                            elif hasattr(pp, 'vector') and pp.vector['directional_agency'] > 0.5:
                                if hasattr(pp.noun_phrase, 'vector'):
                                    vector = pp.noun_phrase.vector
                                    
                                    # Check if this is a rotation verb context
                                    if vp.verb in ['rotate', 'xrotate', 'yrotate', 'zrotate'] or (hasattr(vp, 'vector') and vp.vector and (vp.vector['rotX'] > 0.5 or vp.vector['rotY'] > 0.5 or vp.vector['rotZ'] > 0.5)):
                                        print(f"🔧 Calling _apply_rotation for {vp.verb}")
                                        self._apply_rotation(scene_obj, vp, vp.verb)
                                    # If the vector has a 'number' field, it's likely scaling
                                    elif 'number' in vector and vector['number'] != 0.0:
                                        # Scaling with numeric factors
                                        print(f"🔧 Calling _apply_scaling for {vp.verb}")
                                        self._apply_scaling(scene_obj, vp)
                                    else:
                                        # Default to scaling with vector components
                                        print(f"🔧 Calling _apply_scaling for {vp.verb}")
                                        self._apply_scaling(scene_obj, vp)
                    else:
                        # Transform verb without prepositional phrases - could be basic transform
                        print(f"🔧 Transform verb {verb} without prepositions")
                
                else:
                    print(f"⚠️  Unsupported verb intent for modification: {verb}")
            
            else:
                print(f"⚠️  No vector space information for verb: {verb}")
            
            # Update the visual representation
            self.renderer.update_object(scene_obj)
            
            print(f"✅ Modified object: {obj_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to modify object {obj_id}: {e}")
            return False
    
    def _apply_movement(self, scene_obj: SceneObject, preposition):
        """Apply movement to an object based on prepositional phrase."""
        print(f"🔧 _apply_movement called for {scene_obj.name}")
        print(f"🔧 Preposition: {preposition.preposition}")
        
        # Handle spatial relationships like "above the cube"
        if hasattr(preposition, 'vector') and abs(preposition.vector['spatial_vertical']) > 0.5:
            print(f"🔧 Processing spatial relationship: {preposition.preposition}")
            
            # Find the reference object (e.g., "the cube" in "above the cube")
            if hasattr(preposition, 'noun_phrase') and preposition.noun_phrase:
                ref_description = preposition.noun_phrase
                print(f"🔧 Looking for reference object: {ref_description.noun}")
                
                # Find the reference object in the scene
                ref_object_ids = self.object_resolver.find_objects_by_description(ref_description)
                if ref_object_ids:
                    ref_obj_id = ref_object_ids[0]  # Use the first match
                    
                    # Get the actual SceneObject from the scene
                    ref_obj = None
                    for obj in self.scene.objects:
                        if obj.object_id == ref_obj_id:
                            ref_obj = obj
                            break
                    
                    if ref_obj:
                        print(f"🔧 Found reference object: {ref_obj.name}")
                        
                        # Calculate the new position based on spatial relationship
                        new_x, new_y, new_z = self._calculate_spatial_position(
                            scene_obj, ref_obj, preposition.preposition, preposition.vector['spatial_vertical']
                        )
                        
                        # Update the object's position
                        scene_obj.vector['locX'] = new_x
                        scene_obj.vector['locY'] = new_y
                        scene_obj.vector['locZ'] = new_z
                        
                        print(f"🔧 Moved {scene_obj.name} to [{new_x}, {new_y}, {new_z}]")
                        return
                    else:
                        print(f"🔧 Reference object {ref_obj_id} not found in scene")
                else:
                    print(f"🔧 Reference object not found for spatial relationship")        # Handle direct coordinate movement (original logic)
        if hasattr(preposition, 'noun_phrase') and hasattr(preposition.noun_phrase, 'vector'):
            vector = preposition.noun_phrase.vector
            print(f"🔧 Direct coordinate movement: [{vector['locX']}, {vector['locY']}, {vector['locZ']}]")
            if vector['locX'] != 0.0:
                scene_obj.vector['locX'] = vector['locX']
            if vector['locY'] != 0.0:
                scene_obj.vector['locY'] = vector['locY']
            if vector['locZ'] != 0.0:
                scene_obj.vector['locZ'] = vector['locZ']
    
    def _calculate_spatial_position(self, moving_obj: SceneObject, ref_obj: SceneObject, preposition: str, vertical_factor: float):
        """Calculate the position for spatial relationships like 'above', 'below', etc."""
        # Get reference object's position and size
        ref_x = ref_obj.vector['locX']
        ref_y = ref_obj.vector['locY'] 
        ref_z = ref_obj.vector['locZ']
        
        # Calculate reference object's effective size based on its type
        ref_half_height = self._get_object_half_height(ref_obj)
        moving_half_height = self._get_object_half_height(moving_obj)
        
        print(f"🔧 Reference {ref_obj.name}: center=[{ref_x}, {ref_y}, {ref_z}], half_height={ref_half_height}")
        print(f"🔧 Moving {moving_obj.name}: half_height={moving_half_height}")
        
        # Start with reference object's X and Z coordinates (side-by-side)
        new_x = ref_x
        new_z = ref_z
        
        # Calculate Y position based on spatial relationship using semantic vertical_factor
        if vertical_factor > 0:
            # Place object above reference object (positive vertical factor)
            # Reference object top: ref_y + ref_half_height
            # Moving object bottom should be at reference object top
            # Moving object center should be at: ref_obj_top + moving_half_height
            new_y = ref_y + ref_half_height + moving_half_height
        elif vertical_factor < 0:
            # Place object below reference object (negative vertical factor)
            new_y = ref_y - ref_half_height - moving_half_height
        else:
            # Default to same level (zero vertical factor)
            new_y = ref_y
        
        print(f"🔧 Calculated position for vertical_factor={vertical_factor}: [{new_x}, {new_y}, {new_z}]")
        return new_x, new_y, new_z
    
    def _get_object_half_height(self, obj: SceneObject):
        """Get the half-height of an object based on its type and size."""
        # Object centers are their locations
        # Cube size is the edge length whereas sphere size is only the radius
        
        if 'cube' in obj.name.lower():
            # For cubes, scaleY represents the edge length, so half-height is scaleY/2
            return obj.vector['scaleY'] / 2.0
        elif 'sphere' in obj.name.lower():
            # For spheres, scaleY represents the radius, so half-height equals the radius
            return obj.vector['scaleY']
        else:
            # Default: assume scaleY is half-height
            return obj.vector['scaleY']
    
    def _apply_scaling(self, scene_obj: SceneObject, vp: VerbPhrase):
        """Apply scaling to an object based on verb phrase."""
        print(f"🔧 _apply_scaling called with scene_obj: {scene_obj.name}")
        print(f"🔧 vp.noun_phrase: {vp.noun_phrase}")
        
        # The prepositional phrases are correctly attached to the noun phrase in the VerbPhrase
        # Access them directly without hasattr check
        if vp.noun_phrase and vp.noun_phrase.preps:
            print(f"🔧 Found {len(vp.noun_phrase.preps)} prepositional phrases")
            for pp in vp.noun_phrase.preps:
                print(f"🔧 Processing PP with vector dimensions")
                # Use semantic dimensions instead of hardcoded preposition strings
                if hasattr(pp, 'vector') and pp.vector['directional_agency'] > 0.5 and hasattr(pp.noun_phrase, 'vector'):
                    vector = pp.noun_phrase.vector
                    print(f"🔧 Vector: locX={vector['locX']}, locY={vector['locY']}, locZ={vector['locZ']}")
                    print(f"🔧 Before scaling: scaleX={scene_obj.vector['scaleX']}, scaleY={scene_obj.vector['scaleY']}, scaleZ={scene_obj.vector['scaleZ']}")
                    
                    # Scale values come from the location vector components
                    if vector['locX'] != 0.0:
                        scene_obj.vector['scaleX'] = vector['locX']
                    if vector['locY'] != 0.0:
                        scene_obj.vector['scaleY'] = vector['locY']
                    if vector['locZ'] != 0.0:
                        scene_obj.vector['scaleZ'] = vector['locZ']
                    
                    print(f"🔧 After scaling: scaleX={scene_obj.vector['scaleX']}, scaleY={scene_obj.vector['scaleY']}, scaleZ={scene_obj.vector['scaleZ']}")
        else:
            print(f"🔧 No prepositional phrases found in noun phrase")
        
        # Note: Adjectives are already applied during ATN parsing via NounPhrase.apply_adjective()
        # No need to manually apply them again here
    
    def _apply_rotation(self, scene_obj: SceneObject, vp: VerbPhrase, verb: str):
        """Apply rotation to an object based on verb phrase and rotation verb."""
        print(f"🔧 _apply_rotation called with scene_obj: {scene_obj.name}, verb: {verb}")
        print(f"🔧 vp.noun_phrase: {vp.noun_phrase}")
        
        # The prepositional phrases are correctly attached to the noun phrase in the VerbPhrase
        # Access them directly without hasattr check
        if vp.noun_phrase and vp.noun_phrase.preps:
            print(f"🔧 Found {len(vp.noun_phrase.preps)} prepositional phrases")
            for pp in vp.noun_phrase.preps:
                print(f"🔧 Processing PP with vector dimensions")
                # Use semantic dimensions instead of hardcoded preposition strings
                if hasattr(pp, 'vector') and pp.vector['directional_agency'] > 0.5 and hasattr(pp.noun_phrase, 'vector'):
                    vector = pp.noun_phrase.vector
                    print(f"🔧 Vector: locX={vector['locX']}, locY={vector['locY']}, locZ={vector['locZ']}")
                    print(f"🔧 Before rotation: rotX={scene_obj.vector['rotX']}, rotY={scene_obj.vector['rotY']}, rotZ={scene_obj.vector['rotZ']}")
                    
                    # Check if we have a vector literal with X,Y,Z coordinates
                    if vector['vector'] > 0.5 and (vector['locX'] != 0.0 or vector['locY'] != 0.0 or vector['locZ'] != 0.0):
                        # Multi-axis rotation from vector coordinates [x,y,z]
                        scene_obj.vector['rotX'] = vector['locX']  # X rotation from locX
                        scene_obj.vector['rotY'] = vector['locY']  # Y rotation from locY
                        scene_obj.vector['rotZ'] = vector['locZ']  # Z rotation from locZ
                        print(f"🔧 Applied multi-axis rotation from vector [{vector['locX']}, {vector['locY']}, {vector['locZ']}]")
                    else:
                        # Single-axis rotation - check for single angle value
                        angle = vector.get('number', 0.0) if hasattr(vector, 'get') else vector['number']
                        print(f"🔧 Extracted single angle: {angle}")
                        
                        # Use semantic rotation axis dimensions instead of hardcoded verb strings
                        if hasattr(vp, 'vector') and vp.vector:
                            if vp.vector['rotX'] > 0.5:
                                scene_obj.vector['rotX'] = angle
                            elif vp.vector['rotY'] > 0.5:
                                scene_obj.vector['rotY'] = angle
                            elif vp.vector['rotZ'] > 0.5:
                                scene_obj.vector['rotZ'] = angle
                            else:
                                # Default to Z-axis rotation for generic 'rotate' verb
                                scene_obj.vector['rotZ'] = angle
                        else:
                            # Fallback to Z-axis rotation if no vector information
                            scene_obj.vector['rotZ'] = angle
                    
                    print(f"🔧 After rotation: rotX={scene_obj.vector['rotX']}, rotY={scene_obj.vector['rotY']}, rotZ={scene_obj.vector['rotZ']}")
        else:
            print(f"🔧 No prepositional phrases found in noun phrase")
