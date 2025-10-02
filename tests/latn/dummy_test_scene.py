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
    @staticmethod
    def get_scene1() -> SceneModel:
        scene = SceneModel()
        
        # Create scene objects with distinct vectors
        box_vector = vector_from_word("box")
        table_vector = vector_from_word("table")
        pyramid_vector = vector_from_word("pyramid")
        sphere_vector = vector_from_word("sphere")
        
        box = SceneObject("box", box_vector, object_id="box-1")
        table = SceneObject("table", table_vector, object_id="table-1")
        pyramid = SceneObject("pyramid", pyramid_vector, object_id="pyramid-1")
        sphere = SceneObject("sphere", sphere_vector, object_id="sphere-1")

        box.position = {"x":0, "y":1, "z":0}      # Box is above the table
        table.position = {"x":0, "y":0, "z":0}    # Table is reference
        pyramid.position = {"x":-2, "y":0, "z":0} # Table is right of the pyramid  
        sphere.position = {"x":-2, "y":2, "z":0}  # Pyramid is under the sphere
        # Update vector positions to match
        box.vector['locX'], box.vector['locY'], box.vector['locZ'] = 0, 1, 0
        table.vector['locX'], table.vector['locY'], table.vector['locZ'] = 0, 0, 0
        pyramid.vector['locX'], pyramid.vector['locY'], pyramid.vector['locZ'] = 2, 0, 0
        sphere.vector['locX'], sphere.vector['locY'], sphere.vector['locZ'] = 0, -2, 0

        # Set reasonable scale values for spatial calculations
        for obj in [box, table, pyramid, sphere]:
            obj.vector['scaleX'] = 1.0
            obj.vector['scaleY'] = 1.0 
            obj.vector['scaleZ'] = 1.0
        
        # Add all objects to scene
        scene.add_object(box)
        scene.add_object(table)
        scene.add_object(pyramid) 
        scene.add_object(sphere)

        print(f"🌄 Dummy scene initialized with objects: {scene.objects}")
        return scene
    
    @staticmethod
    def get_scene2() -> SceneModel:
        scene = DummyTestScene.get_scene1()
        box_vector2 = vector_from_word("box")
        box_vector3 = vector_from_word("box")
        box_vector4 = vector_from_word("box")
        box_vector5 = vector_from_word("box")
        box2 = SceneObject("box", box_vector2, object_id="box-2")
        box3 = SceneObject("box", box_vector3, object_id="box-3")
        box4 = SceneObject("box", box_vector4, object_id="box-4")
        box5 = SceneObject("box", box_vector5, object_id="box-5")
        box2.position = {"x":1, "y":1, "z":0}   # Right above table
        box3.position = {"x":-1, "y":1, "z":0}  # Left above table
        box4.position = {"x":-1, "y":-1, "z":0} # Left below table
        box5.position = {"x":1, "y":-1, "z":0}  # Right below table
        for obj in [box2, box3, box4, box5]:
            obj.vector['locX'], obj.vector['locY'], obj.vector['locZ'] = obj.position['x'], obj.position['y'], obj.position['z']
            obj.vector['scaleX'] = 1.0
            obj.vector['scaleY'] = 1.0 
            obj.vector['scaleZ'] = 1.0
            scene.add_object(obj)
        print(f"🌄 Dummy scene2 initialized with objects: {scene.objects}")
        return scene