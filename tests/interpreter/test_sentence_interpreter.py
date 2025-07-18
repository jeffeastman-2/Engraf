"""Test suite for sentence interpreter."""

import pytest
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.mock_renderer import MockRenderer


class TestSentenceInterpreter:
    """Test class for sentence interpreter."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.interpreter = SentenceInterpreter(renderer=MockRenderer())
        
    def teardown_method(self):
        """Cleanup after each test method."""
        self.interpreter.clear_scene()
    
    def test_init(self):
        """Test interpreter initialization."""
        assert self.interpreter.scene is not None
        assert self.interpreter.renderer is not None
        assert self.interpreter.object_counter == 0
        assert len(self.interpreter.execution_history) == 0
    
    def test_simple_creation_sentence(self):
        """Test simple object creation: 'draw a cube'."""
        result = self.interpreter.interpret("draw a cube")
        
        assert result['success'] == True
        assert len(result['objects_created']) == 1
        assert 'cube' in result['objects_created'][0]
        assert 'draw' in result['actions_performed']
        assert len(self.interpreter.scene.objects) == 1
    
    def test_create_blue_cylinder(self):
        """Test creating a blue cylinder"""
        result = self.interpreter.interpret("create a blue cylinder")
        assert result['success'] == True
        assert len(result['objects_created']) == 1
        
        obj_id = result['objects_created'][0]
        obj = next(obj for obj in self.interpreter.scene.objects if obj.object_id == obj_id)
        assert obj.object_id == obj_id
        assert obj.vector['blue'] == 1.0  # Blue color
        assert obj.vector['red'] == 0.0
        assert obj.vector['green'] == 0.0
    
    def test_creation_with_size_adjectives(self):
        """Test object creation with size adjectives: 'draw a big sphere'."""
        result = self.interpreter.interpret("draw a big sphere")
    
        assert result['success'] == True
        assert len(result['objects_created']) == 1
    
        # Check that object has increased scale (big=2.0 + sphere=0.0 = 2.0)
        obj_id = result['objects_created'][0]
        scene_obj = next(obj for obj in self.interpreter.scene.objects if obj.object_id == obj_id)
        assert scene_obj.vector['scaleX'] == 2.0  # Big scale
        assert scene_obj.vector['scaleY'] == 2.0
        assert scene_obj.vector['scaleZ'] == 2.0

    def test_conjunction_creation(self):
        """Test conjunction creation: 'draw a cube and a sphere'."""
        result = self.interpreter.interpret("draw a cube and a sphere")
        
        assert result['success'] == True
        assert len(result['objects_created']) == 2
        assert any('cube' in obj_id for obj_id in result['objects_created'])
        assert any('sphere' in obj_id for obj_id in result['objects_created'])
        assert len(self.interpreter.scene.objects) == 2
    
    def test_multiple_adjectives(self):
        """Test multiple adjectives: 'draw a big red cube'."""
        result = self.interpreter.interpret("draw a big red cube")
    
        assert result['success'] == True
        assert len(result['objects_created']) == 1
    
        # Check both color and size
        obj_id = result['objects_created'][0]
        scene_obj = next(obj for obj in self.interpreter.scene.objects if obj.object_id == obj_id)
        assert scene_obj.vector['red'] == 1.0  # Red color
        assert scene_obj.vector['green'] == 0.0
        assert scene_obj.vector['blue'] == 0.0
        assert scene_obj.vector['scaleX'] == 2.0  # Big scale
        assert scene_obj.vector['scaleY'] == 2.0
        assert scene_obj.vector['scaleZ'] == 2.0

    def test_scene_summary(self):
        """Test scene summary functionality."""
        self.interpreter.interpret("draw a red cube")
        self.interpreter.interpret("draw a blue sphere")
        
        summary = self.interpreter.get_scene_summary()
        
        assert summary['total_objects'] == 2
        assert 'cube' in summary['object_types']
        assert 'sphere' in summary['object_types']
        assert summary['execution_history'] == 2
        assert len(self.interpreter.scene.objects) == 2
    
    def test_clear_scene(self):
        """Test scene clearing functionality."""
        self.interpreter.interpret("draw a cube")
        assert len(self.interpreter.scene.objects) == 1
        
        self.interpreter.clear_scene()
        assert len(self.interpreter.scene.objects) == 0
    
    def test_object_counter_increments(self):
        """Test that object counter increments correctly."""
        initial_count = self.interpreter.object_counter
        
        self.interpreter.interpret("draw a cube")
        assert self.interpreter.object_counter == initial_count + 1
        
        self.interpreter.interpret("draw a sphere")
        assert self.interpreter.object_counter == initial_count + 2
    
    def test_execution_history_tracking(self):
        """Test that execution history is tracked."""
        initial_history_len = len(self.interpreter.execution_history)
        
        self.interpreter.interpret("draw a cube")
        assert len(self.interpreter.execution_history) == initial_history_len + 1
        
        self.interpreter.interpret("draw a sphere")
        assert len(self.interpreter.execution_history) == initial_history_len + 2
    
    def test_result_structure(self):
        """Test the structure of interpretation results."""
        result = self.interpreter.interpret("draw a cube")
        
        # Required keys
        assert 'success' in result
        assert 'message' in result
        assert 'objects_created' in result
        assert 'actions_performed' in result
        assert 'sentence_parsed' in result
        
        # Correct types
        assert isinstance(result['success'], bool)
        assert isinstance(result['message'], str)
        assert isinstance(result['objects_created'], list)
        assert isinstance(result['actions_performed'], list)
    
    def test_creation_verb_synonyms(self):
        """Test that creation verb synonyms work."""
        verbs = ['draw', 'create', 'make', 'build']
        
        for verb in verbs:
            self.interpreter.clear_scene()
            result = self.interpreter.interpret(f"{verb} a cube")
            
            assert result['success'] == True
            assert len(result['objects_created']) == 1
            assert 'cube' in result['objects_created'][0]
            assert verb in result['actions_performed']
    
    def test_default_object_properties(self):
        """Test that objects get default properties."""
        result = self.interpreter.interpret("draw a cube")
        
        assert result['success'] == True
        obj_id = result['objects_created'][0]
        scene_obj = next(obj for obj in self.interpreter.scene.objects if obj.object_id == obj_id)
        
        # Check default position
        assert scene_obj.vector['locX'] == 0.0
        assert scene_obj.vector['locY'] == 0.0
        assert scene_obj.vector['locZ'] == 0.0
        
        # Check default size
        assert scene_obj.vector['scaleX'] == 1.0
        assert scene_obj.vector['scaleY'] == 1.0
        assert scene_obj.vector['scaleZ'] == 1.0
        
        # Check default color (white)
        assert scene_obj.vector['red'] == 1.0
        assert scene_obj.vector['green'] == 1.0
        assert scene_obj.vector['blue'] == 1.0
    
    def test_empty_sentence(self):
        """Test handling of empty sentences."""
        result = self.interpreter.interpret("")
        
        assert result['success'] == False
        assert "Empty sentence" in result['message']
    
    def test_invalid_sentence(self):
        """Test handling of invalid sentences."""
        result = self.interpreter.interpret("asdf xyz blah")
        
        assert result['success'] == False
        assert 'Unknown token' in result['message']
    
    def test_complex_comparative_sentence(self):
        """Test complex comparative sentence parsing."""
        # This tests the parsing of comparative constructions with pronouns
        # and spatial relationships
        result = self.interpreter.interpret("make them more transparent than the purple circle at [3, 3, 3]")
        
        # The sentence should parse successfully even if execution might fail
        # due to missing objects to reference with "them"
        assert result['sentence_parsed'] is not None
        assert result['sentence_parsed'].predicate.verb == 'make'
        assert result['sentence_parsed'].predicate.noun_phrase.vector['pronoun'] == 1.0
        assert result['sentence_parsed'].predicate.noun_phrase.vector['plural'] == 1.0
        assert result['sentence_parsed'].predicate.noun_phrase.vector['transparency'] == 3.0  # 2.0 * 1.5 scaling
        
        # Check the comparative prepositional phrase
        pp = result['sentence_parsed'].predicate.noun_phrase.preps[0]
        assert pp.preposition == 'than'
        assert pp.noun_phrase.noun == 'circle'
        assert pp.noun_phrase.vector['red'] == 0.5
        assert pp.noun_phrase.vector['blue'] == 0.5
        assert pp.noun_phrase.vector['green'] == 0.0
        
        # Check the spatial location
        spatial_pp = pp.noun_phrase.preps[0]
        assert spatial_pp.preposition == 'at'
        assert spatial_pp.noun_phrase.vector['locX'] == 3.0
        assert spatial_pp.noun_phrase.vector['locY'] == 3.0
        assert spatial_pp.noun_phrase.vector['locZ'] == 3.0
