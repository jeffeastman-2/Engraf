"""
Mock renderer for testing without visual output
"""

class MockRenderer:
    def __init__(self, headless=True):
        self.headless = headless
        self.created_objects = []
        
    def create_object(self, obj_type, position, scale, rotation, color, **kwargs):
        """Mock object creation - just store the parameters"""
        mock_obj = {
            'type': obj_type,
            'position': position,
            'scale': scale,
            'rotation': rotation,
            'color': color,
            'kwargs': kwargs
        }
        self.created_objects.append(mock_obj)
        return mock_obj
    
    def update_object(self, obj_id, **kwargs):
        """Mock object update"""
        # Find and update the object
        for obj in self.created_objects:
            if obj.get('id') == obj_id:
                obj.update(kwargs)
                return obj
        return None
    
    def delete_object(self, obj_id):
        """Mock object deletion"""
        self.created_objects = [obj for obj in self.created_objects if obj.get('id') != obj_id]
    
    def clear_scene(self):
        """Clear all objects"""
        self.created_objects.clear()
    
    def render(self):
        """Mock render - do nothing"""
        pass
    
    def render_scene(self, scene):
        """Mock render scene - do nothing"""
        pass
    
    def close(self):
        """Mock close - do nothing"""
        pass
