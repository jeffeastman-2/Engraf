#!/usr/bin/env python3
"""
Unit tests for synthetic_generator.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from engraf.llm_layer6.synthetic_generator import (
    generate_random_scene,
    SentenceGenerator,
    SHAPE_NOUNS, COLOR_ADJECTIVES, ALL_SPATIAL_PREPS, ACTION_VERBS,
    create_mock_example, parse_object_id_to_description
)


class TestSyntheticGenerator(unittest.TestCase):
    """Test the synthetic dataset generator."""

    def test_vocabulary_sizes(self):
        """Test that vocabularies are properly defined."""
        self.assertGreater(len(SHAPE_NOUNS), 5)
        self.assertGreater(len(COLOR_ADJECTIVES), 5)
        self.assertGreater(len(ALL_SPATIAL_PREPS), 5)
        self.assertGreater(len(ACTION_VERBS), 0)

    def test_generate_random_scene(self):
        """Test that random scene generation works."""
        scene = generate_random_scene(4)
        self.assertEqual(len(scene.objects), 4)
        
        # Check objects have required properties
        for obj in scene.objects:
            self.assertIsNotNone(obj.object_id)
            self.assertIsNotNone(obj.name)
            self.assertIsNotNone(obj.vector)

    def test_sentence_generator_imperatives(self):
        """Test imperative sentence generation."""
        scene = generate_random_scene(3)
        generator = SentenceGenerator(scene)
        
        imperatives = generator.generate_imperatives()
        self.assertGreater(len(imperatives), 0)
        
        # Check structure
        sentence, answer, obj_ids = imperatives[0]
        self.assertIsInstance(sentence, str)
        self.assertIsInstance(answer, str)
        self.assertIsInstance(obj_ids, list)
        
        # Check imperative starts with verb
        first_word = sentence.split()[0]
        self.assertIn(first_word, ACTION_VERBS)
        
        # Check answer has proper gerund
        self.assertTrue(answer.endswith('.'))
        self.assertTrue(answer[0].isupper())

    def test_sentence_generator_declaratives(self):
        """Test declarative sentence generation."""
        scene = generate_random_scene(3)
        generator = SentenceGenerator(scene)
        
        declaratives = generator.generate_declaratives()
        self.assertGreater(len(declaratives), 0)
        
        sentence, answer, obj_ids = declaratives[0]
        
        # Declaratives start with "the"
        self.assertTrue(sentence.startswith('the '))
        self.assertIn(' is ', sentence)

    def test_sentence_generator_interrogatives(self):
        """Test interrogative sentence generation."""
        scene = generate_random_scene(3)
        generator = SentenceGenerator(scene)
        
        interrogatives = generator.generate_interrogatives()
        self.assertGreater(len(interrogatives), 0)
        
        sentence, answer, obj_ids = interrogatives[0]
        
        # Interrogatives start with "is"
        self.assertTrue(sentence.startswith('is '))
        
        # Answers start with Yes or No
        self.assertTrue(answer.startswith('Yes,') or answer.startswith('No,'))

    def test_spatial_relationship_computation(self):
        """Test that spatial relationships are computed correctly."""
        scene = generate_random_scene(2)
        generator = SentenceGenerator(scene)
        
        obj1, obj2 = scene.objects[0], scene.objects[1]
        relationships = generator.compute_spatial_relationship(obj1, obj2)
        
        # Should have all expected keys
        expected_keys = ['above', 'below', 'over', 'under', 'left of', 'right of',
                        'in front of', 'behind', 'on', 'near', 'at']
        for key in expected_keys:
            self.assertIn(key, relationships)
            # Accept numpy bool or Python bool
            self.assertIn(type(relationships[key]).__name__, ['bool', 'bool_'])

    def test_generate_all(self):
        """Test combined generation."""
        scene = generate_random_scene(3)
        generator = SentenceGenerator(scene)
        
        all_examples = generator.generate_all()
        
        # Should have examples from all types
        # 3 objects = 6 ordered pairs
        # Each pair: 3 verbs × 11 preps (imperatives) + 11 preps (declaratives) + 11 preps (interrogatives)
        # Plus what/where questions
        self.assertGreater(len(all_examples), 100)

    def test_parse_object_id_to_description(self):
        """Test object ID parsing."""
        self.assertEqual(parse_object_id_to_description('red_cube_1'), 'red cube')
        self.assertEqual(parse_object_id_to_description('blue_sphere_2'), 'blue sphere')
        self.assertEqual(parse_object_id_to_description('green_cylinder_1'), 'green cylinder')

    def test_create_mock_example(self):
        """Test mock example creation."""
        scene = generate_random_scene(2)
        obj_ids = [obj.object_id for obj in scene.objects]
        
        example = create_mock_example(
            "move the red cube above the blue sphere",
            "Moving the red cube above the blue sphere.",
            obj_ids,
            scene
        )
        
        # Check required keys
        self.assertIn('structural_tokens', example)
        self.assertIn('semantic_vectors', example)
        self.assertIn('scene_grounding', example)
        self.assertIn('input_string', example)
        self.assertIn('target_string', example)
        self.assertIn('question', example)
        
        # Check Layer-6 structure
        tokens = example['structural_tokens']
        self.assertIn('[SP', tokens)
        self.assertIn(']SP', tokens)
        self.assertIn('[NP', tokens)
        self.assertIn(']NP', tokens)


class TestSentenceVariety(unittest.TestCase):
    """Test sentence variety and coverage."""

    def test_all_prepositions_used(self):
        """Test that all spatial prepositions are used."""
        scene = generate_random_scene(2)
        generator = SentenceGenerator(scene)
        
        all_examples = generator.generate_all()
        
        # Collect all prepositions used
        used_preps = set()
        for sentence, _, _ in all_examples:
            for prep in ALL_SPATIAL_PREPS:
                if f' {prep} ' in sentence or sentence.endswith(f' {prep}'):
                    used_preps.add(prep)
        
        # Should use all prepositions
        self.assertEqual(used_preps, set(ALL_SPATIAL_PREPS))

    def test_all_verbs_used(self):
        """Test that all action verbs are used in imperatives."""
        scene = generate_random_scene(2)
        generator = SentenceGenerator(scene)
        
        imperatives = generator.generate_imperatives()
        
        used_verbs = set()
        for sentence, _, _ in imperatives:
            first_word = sentence.split()[0]
            used_verbs.add(first_word)
        
        # Should use all verbs
        self.assertEqual(used_verbs, set(ACTION_VERBS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
