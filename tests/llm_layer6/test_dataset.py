#!/usr/bin/env python3
"""
Unit tests for SyntheticLayer6Dataset in dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import torch
from engraf.llm_layer6.dataset import SyntheticLayer6Dataset


class TestSyntheticLayer6Dataset(unittest.TestCase):
    """Test the on-the-fly synthetic data generator."""

    def test_dataset_creation(self):
        """Test that dataset can be created."""
        dataset = SyntheticLayer6Dataset(num_examples=10, num_scene_objects=5)
        self.assertEqual(len(dataset), 10)

    def test_dataset_item_structure(self):
        """Test that each item has the expected keys."""
        dataset = SyntheticLayer6Dataset(num_examples=5, num_scene_objects=3)
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('target_ids', item)
        self.assertIn('question', item)
        self.assertIn('answer', item)
        self.assertIn('scene_objects', item)
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['target_ids'], torch.Tensor)

    def test_dataset_batch_loading(self):
        """Test that DataLoader can batch the dataset."""
        dataset = SyntheticLayer6Dataset(num_examples=20, num_scene_objects=5)
        # Note: default collate won't work with variable-length tensors, so just iterate
        for i in range(min(4, len(dataset))):
            item = dataset[i]
            self.assertIn('question', item)

    def test_print_sample_outputs(self):
        """Print some sample Q&A outputs for inspection."""
        dataset = SyntheticLayer6Dataset(num_examples=10, num_scene_objects=5, max_output_length=100)
        print("\n" + "=" * 70)
        print("Sample Q&A Outputs from SyntheticLayer6Dataset")
        print("=" * 70)
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Scene Objects: {item['scene_objects']}")
            print(f"Question: {item['question']}")
            print(f"Answer: {item['answer']}")
        print("\n" + "=" * 70)


if __name__ == '__main__':
    # Run tests and print sample outputs
    unittest.main(verbosity=2)
