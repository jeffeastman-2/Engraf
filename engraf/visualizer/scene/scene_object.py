from pprint import pprint
from .scene_entity import SceneEntity


class SceneObject(SceneEntity):
    def __init__(self, name, vector, object_id=None):
        self.name = name                  # e.g., 'cube' (the base noun)
        self.object_id = object_id or name  # e.g., 'red_cube_1' (unique identifier)
        self.vector = vector              # VectorSpace instance
        
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

    # SceneEntity interface implementation
    @property
    def entity_id(self) -> str:
        """Unique identifier for this entity."""
        return self.object_id
    
    def move_to(self, new_x: float, new_y: float, new_z: float) -> None:
        """Move the object to the specified coordinates."""
        self.position['x'] = new_x
        self.position['y'] = new_y
        self.position['z'] = new_z
        # Update the vector space
        self.vector['locX'] = new_x
        self.vector['locY'] = new_y
        self.vector['locZ'] = new_z
        self._update_transform_matrix()
    
    def scale_by(self, factor_x: float, factor_y: float, factor_z: float) -> None:
        """Scale the object by the specified factors."""
        self.scale['x'] *= factor_x
        self.scale['y'] *= factor_y
        self.scale['z'] *= factor_z
        # Update the vector space
        self.vector['scaleX'] = self.scale['x']
        self.vector['scaleY'] = self.scale['y']
        self.vector['scaleZ'] = self.scale['z']
        self._update_transform_matrix()
    
    def rotate_around_center(self, angle_x: float, angle_y: float, angle_z: float) -> None:
        """Rotate the object around its center by the specified angles (in degrees)."""
        self.rotation['x'] += angle_x
        self.rotation['y'] += angle_y
        self.rotation['z'] += angle_z
        # Update the vector space
        self.vector['rotX'] = self.rotation['x']
        self.vector['rotY'] = self.rotation['y']
        self.vector['rotZ'] = self.rotation['z']
        self._update_transform_matrix()
    
    def get_position(self) -> tuple[float, float, float]:
        """Get the current position of the object."""
        return (self.position['x'], self.position['y'], self.position['z'])
    
    def get_rotation(self) -> tuple[float, float, float]:
        """Get the current rotation of the object (in degrees)."""
        return (self.rotation['x'], self.rotation['y'], self.rotation['z'])
    
    def get_scale(self) -> tuple[float, float, float]:
        """Get the current scale of the object."""
        return (self.scale['x'], self.scale['y'], self.scale['z'])

    def __repr__(self):
        rotation_str = f"rot=[{self.rotation['x']:.1f},{self.rotation['y']:.1f},{self.rotation['z']:.1f}]" if self.has_rotation() else ""
        return f"<{self.name} ({self.object_id}) pos=[{self.position['x']},{self.position['y']},{self.position['z']}] {rotation_str}>".strip()

def scene_object_from_np(noun_phrase):
    """Create a SceneObject from a noun phrase."""
    from pprint import pprint
    pprint(f"🟢 scene from NP = {noun_phrase}")

    obj = SceneObject(
        name=noun_phrase.noun,
        vector=noun_phrase.vector
    )
    return obj
