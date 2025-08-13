"""
Object Modifier Handler

This module handles modifications to existing scene objects,
including transformations like movement, rotation, and scaling.
"""

from typing import List, Union
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.visualizer.scene.scene_assembly import SceneAssembly
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
    
    def modify_scene_object(self, entity_id: str, vp: VerbPhrase) -> bool:
        """Modify a scene entity (object or assembly) based on a verb phrase."""
        print(f"🔧 _modify_scene_object called with entity_id: {entity_id}, verb: {vp.verb}")
        try:
            # Find the scene entity (could be object or assembly)
            scene_entity = self.scene.find_entity_by_id(entity_id)
            
            if not scene_entity:
                print(f"🔧 Scene entity not found: {entity_id}")
                return False
            
            verb = vp.verb
            print(f"🔧 Processing verb: {verb}")
            print(f"🔧 Entity type: {type(scene_entity).__name__}")
            
            # Check if verb phrase has vector space information
            if hasattr(vp, 'vector') and vp.vector:
                print(f"🔧 VerbPhrase has vector: {vp.vector}")
                print(f"🔧 vp.vector.isa('transform'): {vp.vector.isa('transform')}")
                print(f"🔧 hasattr(vp, 'adjective_complement'): {hasattr(vp, 'adjective_complement')}")
                if hasattr(vp, 'adjective_complement'):
                    print(f"🔧 vp.adjective_complement: {vp.adjective_complement}")
                    print(f"🔧 bool(vp.adjective_complement): {bool(vp.adjective_complement)}")
                
                # Handle transform verbs (move, rotate, scale, ...) using vector space
                if vp.vector.isa('transform') and vp.noun_phrase:
                    print(f"🔧 Taking transform verb path with noun_phrase")
                    if vp.noun_phrase.preps:
                        print(f"🔧 Processing prepositional phrases")
                        # Process prepositional phrases using semantic dimensions
                        for pp in vp.noun_phrase.preps:
                            # Check for movement using directional_target dimension OR spatial relationships
                            if hasattr(pp, 'vector') and (pp.vector.isa('directional_target') or pp.vector.isa('spatial_location')):
                                self._apply_movement(scene_entity, pp)
                            # Check for rotation/scaling using directional_agency dimension
                            elif hasattr(pp, 'vector') and pp.vector.isa('directional_agency'):
                                if hasattr(pp.noun_phrase, 'vector'):
                                    vector = pp.noun_phrase.vector
                                    
                                    # Check if this is a rotation verb context
                                    if vp.verb in ['rotate', 'xrotate', 'yrotate', 'zrotate'] or (hasattr(vp, 'vector') and vp.vector and (vp.vector.isa('rotX') or vp.vector.isa('rotY') or vp.vector.isa('rotZ'))):
                                        print(f"🔧 Calling _apply_rotation for {vp.verb}")
                                        self._apply_rotation(scene_entity, vp, vp.verb)
                                    # If the vector has a 'number' field, it's likely scaling
                                    elif vp.vector.isa('number'):
                                        # Scaling with numeric factors
                                        print(f"🔧 Calling _apply_scaling for {vp.verb}")
                                        self._apply_scaling(scene_entity, vp)
                                    else:
                                        # Default to scaling with vector components
                                        print(f"🔧 Calling _apply_scaling for {vp.verb}")
                                        self._apply_scaling(scene_entity, vp)
                    else:
                        # Transform verb without prepositional phrases - could be basic transform
                        print(f"🔧 Transform verb {verb} without prepositions")
                        
                        # Check for adjective complements that might indicate scaling
                        # Generic transform verbs like 'make' can use adjective complements for scaling
                        if hasattr(vp, 'adjective_complement') and vp.adjective_complement:
                            print(f"🔧 Found adjective complements for transformation: {vp.adjective_complement}")
                            self._apply_adjective_scaling(scene_entity, vp)
                        else:
                            print(f"🔧 No adjective complements for transformation")
                            print(f"    hasattr(vp, 'adjective_complement'): {hasattr(vp, 'adjective_complement')}")
                            print(f"    vp.adjective_complement: {vp.adjective_complement if hasattr(vp, 'adjective_complement') else 'N/A'}")
                
                # Handle style verbs (color, texture, etc.) using vector space  
                elif vp.vector.isa('style') and hasattr(vp, 'adjective_complement'):
                    print(f"🔧 Taking style verb path")
                    # Style the object - adjectives are already applied during ATN parsing
                    # so we just update the visual representation
                    pass
                
                else:
                    print(f"🔧 Not taking any verb transformation path")
                    print(f"    Transform condition: {vp.vector.isa('transform') }")
                    print(f"    Has noun_phrase: {vp.noun_phrase is not None}")
                    if vp.noun_phrase:
                        print(f"    noun_phrase.preps: {vp.noun_phrase.preps}")
                    print(f"⚠️  Unsupported verb intent for modification: {verb}")
            
            else:
                print(f"⚠️  No vector space information for verb: {verb}")
            
            # Update the visual representation
            if isinstance(scene_entity, SceneAssembly):
                # For assemblies, update each constituent object
                for obj in scene_entity.objects:
                    self.renderer.update_object(obj)
            else:
                # For individual objects, update directly
                self.renderer.update_object(scene_entity)
            
            print(f"✅ Modified entity: {entity_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to modify entity {entity_id}: {e}")
            return False
    
    def _apply_movement(self, scene_entity: Union[SceneObject, SceneAssembly], preposition):
        """Apply movement to an object or assembly based on prepositional phrase."""
        print(f"🔧 _apply_movement called for {scene_entity.name}")
        print(f"🔧 Preposition: {preposition.preposition}")
        
        # First, check for spatial relationships like "above the cube"
        if hasattr(preposition, 'vector') and preposition.vector.isa('spatial_location'):
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
                        
                        # For assemblies, we need to handle spatial relationships differently
                        if isinstance(scene_entity, SceneAssembly):
                            print(f"🔧 Warning: Spatial relationships for assemblies not yet implemented")
                            return
                        
                        # Calculate the new position based on spatial relationship
                        new_x, new_y, new_z = self._calculate_spatial_position(
                            scene_entity, ref_obj, preposition.preposition, preposition.vector
                        )
                        
                        # Update the object's position
                        scene_entity.vector['locX'] = new_x
                        scene_entity.vector['locY'] = new_y
                        scene_entity.vector['locZ'] = new_z
                        
                        # Update transformation properties from vector (critical for renderer)
                        scene_entity.update_transformations()
                        
                        print(f"🔧 Moved {scene_entity.name} to [{new_x}, {new_y}, {new_z}]")
                        print(f"🔧 Updated transformation properties: position={scene_entity.position}")
                        return
                    else:
                        print(f"🔧 Reference object {ref_obj_id} not found in scene")
                else:
                    print(f"🔧 Reference object not found for spatial relationship")
            return
        
        # Handle direct coordinate movement (fallback for coordinates like [5,5,5])
        if hasattr(preposition, 'noun_phrase') and hasattr(preposition.noun_phrase, 'vector'):
            vector = preposition.noun_phrase.vector
            print(f"🔧 Direct coordinate movement: [{vector['locX']}, {vector['locY']}, {vector['locZ']}]")
            
            # Check if this is an assembly
            if isinstance(scene_entity, SceneAssembly):
                # Use assembly's move_to method which updates all constituent objects
                scene_entity.move_to(vector['locX'], vector['locY'], vector['locZ'])
                print(f"🔧 Updated assembly transformation properties: position={scene_entity.position}")
            else:
                # Original SceneObject logic
                if vector['locX'] != 0.0:
                    scene_entity.vector['locX'] = vector['locX']
                if vector['locY'] != 0.0:
                    scene_entity.vector['locY'] = vector['locY']
                if vector['locZ'] != 0.0:
                    scene_entity.vector['locZ'] = vector['locZ']
                
                # Update transformation properties from vector (critical for renderer)
                scene_entity.update_transformations()
                print(f"🔧 Updated transformation properties: position={scene_entity.position}")
            return

    def _calculate_spatial_position(self, moving_obj: SceneObject, ref_obj: SceneObject, preposition: str, preposition_vector):
        """Calculate the position for spatial relationships like 'above', 'below', 'beside', etc.
        
        Uses the simplified approach where prepositions directly specify their spatial effects
        through locX, locY, and locZ values rather than separate spatial dimension flags.
        """
        # Get direction factors directly from the preposition vector
        # Use __contains__ to check if dimension exists, then get the actual value (positive or negative)
        x_factor = preposition_vector['locX'] if 'locX' in preposition_vector and preposition_vector['locX'] != 0.0 else 0.0
        y_factor = preposition_vector['locY'] if 'locY' in preposition_vector and preposition_vector['locY'] != 0.0 else 0.0
        z_factor = preposition_vector['locZ'] if 'locZ' in preposition_vector and preposition_vector['locZ'] != 0.0 else 0.0

        # Get reference object's position and size
        ref_x = ref_obj.vector['locX']
        ref_y = ref_obj.vector['locY'] 
        ref_z = ref_obj.vector['locZ']
        
        # Calculate object dimensions for proper spacing
        refHeight, refBreadth, refDepth = self._get_object_half_scale(ref_obj)
        movingHeight, movingBreadth, movingDepth = self._get_object_half_scale(moving_obj)

        print(f"🔧 Reference {ref_obj.name}: center=[{ref_x}, {ref_y}, {ref_z}], half_size=[{refBreadth}, {refHeight}, {refDepth}]")
        print(f"🔧 Moving {moving_obj.name}: half_size=[{movingBreadth}, {movingHeight}, {movingDepth}]")
        
        # Start with reference object's position as base
        new_x = ref_x
        new_y = ref_y
        new_z = ref_z

        # Calculate X position based on directional factor
        if x_factor > 0:
            # Place object to the positive X direction (right/beside)
            new_x = ref_x + refBreadth + movingBreadth + abs(x_factor)
        elif x_factor < 0:
            # Place object to the negative X direction (left)
            new_x = ref_x - refBreadth - movingBreadth - abs(x_factor)

        # Calculate Y position based on directional factor
        if y_factor > 0:
            # Place object in positive Y direction (above)
            new_y = ref_y + refHeight + movingHeight + abs(y_factor)
        elif y_factor < 0:
            # Place object in negative Y direction (below)
            new_y = ref_y - refHeight - movingHeight - abs(y_factor)

        # Calculate Z position based on directional factor
        if z_factor > 0:
            # Place object in positive Z direction (behind)
            new_z = ref_z + refDepth + movingDepth + abs(z_factor)
        elif z_factor < 0:
            # Place object in negative Z direction (in front)
            new_z = ref_z - refDepth - movingDepth - abs(z_factor)

        print(f"🔧 Calculated position for preposition={preposition} (x:{x_factor}, y:{y_factor}, z:{z_factor}): [{new_x}, {new_y}, {new_z}]")
        return new_x, new_y, new_z
    
    def _get_object_half_scale(self, obj: SceneObject):
        """Get the half-scale of an object based on its type and size.
        
        Returns:
            tuple: (half_height, half_breadth, half_depth) representing the object's half-dimensions
        """
        if 'cube' in obj.name.lower():
            # For cubes, all scales represent edge lengths, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0
        elif 'sphere' in obj.name.lower():
            # For spheres, all scales represent radius, so half-size equals the radius
            radius = max(obj.vector['scaleX'], obj.vector['scaleY'], obj.vector['scaleZ'])
            half_height = radius
            half_breadth = radius
            half_depth = radius
        else:
            # Default: assume scales represent full dimensions, so half-size is scale/2
            half_height = obj.vector['scaleY'] / 2.0
            half_breadth = obj.vector['scaleX'] / 2.0
            half_depth = obj.vector['scaleZ'] / 2.0
            
        return half_height, half_breadth, half_depth


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
                if hasattr(pp, 'vector') and pp.vector.isa('directional_agency') and hasattr(pp.noun_phrase, 'vector'):
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
                    
                    # Update transformation properties from vector (critical for renderer)
                    scene_obj.update_transformations()
                    print(f"🔧 Updated transformation properties: scale={scene_obj.scale}")
        else:
            print(f"🔧 No prepositional phrases found in noun phrase")
        
        # Note: Adjectives are already applied during ATN parsing via NounPhrase.apply_adjective()
        # No need to manually apply them again here

    def _apply_adjective_scaling(self, scene_obj: SceneObject, vp: VerbPhrase):
        """Apply scaling to an object based on adjective complements like 'bigger'."""
        print(f"🔧 _apply_adjective_scaling called with scene_obj: {scene_obj.name}")
        print(f"🔧 adjective_complement: {vp.adjective_complement}")
        
        # Process each adjective complement
        for adjective in vp.adjective_complement:
            print(f"🔧 Processing adjective: {adjective}")
            print(f"🔧 Before scaling: scaleX={scene_obj.vector['scaleX']}, scaleY={scene_obj.vector['scaleY']}, scaleZ={scene_obj.vector['scaleZ']}")
            
            # Apply known scaling factors for common adjectives
            if adjective == 'bigger':
                # Apply scaling factor for "bigger" - use the known value from vocabulary
                scale_factor = 2.4  # This matches the vocabulary value
                scene_obj.vector['scaleX'] = scale_factor
                scene_obj.vector['scaleY'] = scale_factor
                scene_obj.vector['scaleZ'] = scale_factor
                print(f"🔧 Applied 'bigger' scaling factor: {scale_factor}")
            elif adjective == 'smaller':
                # Apply scaling factor for "smaller"
                scale_factor = 0.5  # Default smaller scale
                scene_obj.vector['scaleX'] = scale_factor
                scene_obj.vector['scaleY'] = scale_factor
                scene_obj.vector['scaleZ'] = scale_factor
                print(f"🔧 Applied 'smaller' scaling factor: {scale_factor}")
            elif adjective == 'larger':
                # Apply scaling factor for "larger"
                scale_factor = 2.0  # Similar to bigger but slightly different
                scene_obj.vector['scaleX'] = scale_factor
                scene_obj.vector['scaleY'] = scale_factor
                scene_obj.vector['scaleZ'] = scale_factor
                print(f"🔧 Applied 'larger' scaling factor: {scale_factor}")
            else:
                print(f"🔧 Adjective '{adjective}' is not a recognized scaling adjective")
            
            print(f"🔧 After scaling: scaleX={scene_obj.vector['scaleX']}, scaleY={scene_obj.vector['scaleY']}, scaleZ={scene_obj.vector['scaleZ']}")
        
        # Update transformation properties from vector (critical for renderer)
        scene_obj.update_transformations()
        print(f"🔧 Updated transformation properties: scale={scene_obj.scale}")
    
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
                if hasattr(pp, 'vector') and pp.vector.isa('directional_agency') and hasattr(pp.noun_phrase, 'vector'):
                    vector = pp.noun_phrase.vector
                    print(f"🔧 Vector: locX={vector['locX']}, locY={vector['locY']}, locZ={vector['locZ']}")
                    print(f"🔧 Before rotation: rotX={scene_obj.vector['rotX']}, rotY={scene_obj.vector['rotY']}, rotZ={scene_obj.vector['rotZ']}")
                    
                    # Check if we have a vector literal with X,Y,Z coordinates
                    if vector.isa('vector') and (vector['locX'] != 0.0 or vector['locY'] != 0.0 or vector['locZ'] != 0.0):
                        # Multi-axis rotation from vector coordinates [x,y,z]
                        scene_obj.vector['rotX'] = vector['locX']  # X rotation from locX
                        scene_obj.vector['rotY'] = vector['locY']  # Y rotation from locY
                        scene_obj.vector['rotZ'] = vector['locZ']  # Z rotation from locZ
                        print(f"🔧 Applied multi-axis rotation from vector [{vector['locX']}, {vector['locY']}, {vector['locZ']}]")
                    else:
                        # Single-axis rotation - check for single angle value
                        angle = vector['number'] if hasattr(vector, '__getitem__') else 0.0
                        print(f"🔧 Extracted single angle: {angle}")
                        
                        # Use semantic rotation axis dimensions instead of hardcoded verb strings
                        if hasattr(vp, 'vector') and vp.vector:
                            if vp.vector.isa('rotX'):
                                scene_obj.vector['rotX'] = angle
                            elif vp.vector.isa('rotY'):
                                scene_obj.vector['rotY'] = angle
                            elif vp.vector.isa('rotZ'):
                                scene_obj.vector['rotZ'] = angle
                            else:
                                # Default to Z-axis rotation for generic 'rotate' verb
                                scene_obj.vector['rotZ'] = angle
                        else:
                            # Fallback to Z-axis rotation if no vector information
                            scene_obj.vector['rotZ'] = angle
                    
                    # Update the SceneObject's transformation properties from the vector
                    scene_obj.update_transformations()
                    
                    print(f"🔧 After rotation: rotX={scene_obj.vector['rotX']}, rotY={scene_obj.vector['rotY']}, rotZ={scene_obj.vector['rotZ']}")
                    print(f"🔧 SceneObject rotation: {scene_obj.rotation}")
        else:
            print(f"🔧 No prepositional phrases found in noun phrase")
