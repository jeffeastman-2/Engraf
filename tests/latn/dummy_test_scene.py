from engraf.lexer.vocabulary_builder import vector_from_word
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject

class DummyTestScene():
    """Set up a dummy scene for spatial validation tests.
    
    Scene contains:
    - box (above table)
    - table (reference object)
    - pyramid (left of table)
    - sphere (above pyramid)    
    """
    def __init__(self):
        self.scene = SceneModel()
        
        # Create scene objects with distinct vectors
        box_vector = vector_from_word("box")
        table_vector = vector_from_word("table")
        pyramid_vector = vector_from_word("pyramid")
        sphere_vector = vector_from_word("sphere")
        
        self.box = SceneObject("box", box_vector, object_id="box-1")
        self.table = SceneObject("table", table_vector, object_id="table-1")
        self.pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
        self.sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")

        self.box.position = {"x":0, "y":1, "z":0}      # Box is above the table
        self.table.position = {"x":0, "y":0, "z":0}    # Table is reference
        self.pyramid.position = {"x":-2, "y":0, "z":0} # Table is right of the pyramid  
        self.sphere.position = {"x":-2, "y":2, "z":0}  # Pyramid is under the sphere
        # Update vector positions to match
        self.box.vector['locX'], self.box.vector['locY'], self.box.vector['locZ'] = 0, 1, 0
        self.table.vector['locX'], self.table.vector['locY'], self.table.vector['locZ'] = 0, 0, 0
        self.pyramid.vector['locX'], self.pyramid.vector['locY'], self.pyramid.vector['locZ'] = 2, 0, 0
        self.sphere.vector['locX'], self.sphere.vector['locY'], self.sphere.vector['locZ'] = 0, -2, 0

        # Set reasonable scale values for spatial calculations
        for obj in [self.box, self.table, self.pyramid, self.sphere]:
            obj.vector['scaleX'] = 1.0
            obj.vector['scaleY'] = 1.0 
            obj.vector['scaleZ'] = 1.0
        
        # Add all objects to scene
        self.scene.add_object(self.box)
        self.scene.add_object(self.table)
        self.scene.add_object(self.pyramid) 
        self.scene.add_object(self.sphere)
        
        print(f"🌄 Dummy scene initialized with objects: {self.scene.objects}")

    @staticmethod
    def get_scene(self) -> SceneModel:
        return self.scene