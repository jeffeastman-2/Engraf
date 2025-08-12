from pprint import pprint

class SceneObject:
    def __init__(self, name, vector, modifiers=None, object_id=None):
        self.name = name                  # e.g., 'cube' (the base noun)
        self.object_id = object_id or name  # e.g., 'red_cube_1' (unique identifier)
        self.vector = vector              # VectorSpace instance
        self.modifiers = modifiers or []  # nested SceneObjects from PPs
        self.metadata = {}                # Store additional metadata for matching
        
        # Extract transformation properties from vector space
        self._update_transformations_from_vector()
        
        # Create the transformation matrix
        self._update_transform_matrix()
    
    def _update_transformations_from_vector(self):
        """Extract transformation properties from the vector space."""
        if self.vector:
            # Position
            self.position = {
                'x': self.vector['locX'] if 'locX' in self.vector else 0.0,
                'y': self.vector['locY'] if 'locY' in self.vector else 0.0,
                'z': self.vector['locZ'] if 'locZ' in self.vector else 0.0
            }
            
            # Rotation (in degrees)
            self.rotation = {
                'x': self.vector['rotX'] if 'rotX' in self.vector else 0.0,
                'y': self.vector['rotY'] if 'rotY' in self.vector else 0.0,
                'z': self.vector['rotZ'] if 'rotZ' in self.vector else 0.0
            }
            
            # Scale
            self.scale = {
                'x': self.vector['scaleX'] if 'scaleX' in self.vector else 1.0,
                'y': self.vector['scaleY'] if 'scaleY' in self.vector else 1.0,
                'z': self.vector['scaleZ'] if 'scaleZ' in self.vector else 1.0
            }
            
            # Color
            self.color = {
                'r': self.vector['red'] if 'red' in self.vector else 1.0,
                'g': self.vector['green'] if 'green' in self.vector else 1.0,
                'b': self.vector['blue'] if 'blue' in self.vector else 1.0
            }
        else:
            # Default values if no vector
            self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.rotation = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
            self.color = {'r': 1.0, 'g': 1.0, 'b': 1.0}
    
    def _update_transform_matrix(self):
        """Create transformation matrix from position, rotation, and scale."""
        try:
            from engraf.visualizer.transforms.transform_matrix import TransformMatrix
            
            # Create individual transformation matrices
            translation = TransformMatrix.translation(
                self.position['x'], self.position['y'], self.position['z']
            )
            
            rotation_x = TransformMatrix.rotation_x(self.rotation['x'])
            rotation_y = TransformMatrix.rotation_y(self.rotation['y']) 
            rotation_z = TransformMatrix.rotation_z(self.rotation['z'])
            
            scale = TransformMatrix.scale(
                self.scale['x'], self.scale['y'], self.scale['z']
            )
            
            # Compose transformations: Scale -> Rotate Z -> Rotate Y -> Rotate X -> Translate
            self.transform_matrix = translation.compose(
                rotation_x.compose(
                    rotation_y.compose(
                        rotation_z.compose(scale)
                    )
                )
            )
            
        except ImportError:
            # Fallback if TransformMatrix not available
            self.transform_matrix = None
    
    def update_transformations(self):
        """Update transformation properties from vector space (call after vector changes)."""
        self._update_transformations_from_vector()
        self._update_transform_matrix()
    
    def has_rotation(self):
        """Check if the object has any non-zero rotation."""
        return self.rotation['x'] != 0.0 or self.rotation['y'] != 0.0 or self.rotation['z'] != 0.0
    
    def get_rotation_radians(self):
        """Get rotation in radians for rendering."""
        import math
        return {
            'x': math.radians(self.rotation['x']),
            'y': math.radians(self.rotation['y']),
            'z': math.radians(self.rotation['z'])
        }

    def __repr__(self):
        rotation_str = f"rot=[{self.rotation['x']:.1f},{self.rotation['y']:.1f},{self.rotation['z']:.1f}]" if self.has_rotation() else ""
        modifier_names = [mod.name for mod in self.modifiers] if self.modifiers else []
        modifiers_str = f"modifiers={modifier_names}"
        return f"<{self.name} ({self.object_id}) pos=[{self.position['x']},{self.position['y']},{self.position['z']}] {rotation_str} {modifiers_str}>".strip()

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
