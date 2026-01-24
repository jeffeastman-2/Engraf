#!/usr/bin/env python3
"""
Generate Layer-6 training dataset from demo_scene_setup.

Synthesizes spatial reasoning questions/answers using the 4-object scene,
converts them to Layer-6 structural representation, and writes JSONL dataset.

NOTE: This is a template — it assumes you have a way to parse sentences
through LATN Layers 1-5. For now, it shows the *intended* workflow.
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info
from engraf.llm_layer6.dataset_extractor import create_training_pair_from_hyp, write_jsonl
from engraf.lexer.vector_space import VECTOR_LENGTH
import itertools

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


def generate_spatial_questions(scene):
    """Generate spatial reasoning questions from scene objects.
    
    Args:
        scene: SceneModel with 4 objects
    
    Yields:
        Tuples of (question_str, answer_str, expected_object_ids)
    """
    objs = {obj.object_id: obj for obj in scene.objects}
    
    # Get all pairwise spatial relationships
    obj_pairs = list(itertools.combinations(objs.keys(), 2))
    
    for obj1_id, obj2_id in obj_pairs:
        obj1 = objs[obj1_id]
        obj2 = objs[obj2_id]
        
        # Extract positions
        x1, y1, z1 = obj1.vector['locX'], obj1.vector['locY'], obj1.vector['locZ']
        x2, y2, z2 = obj2.vector['locX'], obj2.vector['locY'], obj2.vector['locZ']
        
        # Generate questions based on relationships
        
        # Y-axis (vertical/above/below)
        if y1 > y2:
            yield (
                f"Is the {obj1.name} above the {obj2.name}?",
                f"Yes, the {obj1.name} is above the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} above the {obj1.name}?",
                f"No, the {obj2.name} is below the {obj1.name}.",
                [obj1_id, obj2_id]
            )
        elif y1 < y2:
            yield (
                f"Is the {obj1.name} below the {obj2.name}?",
                f"Yes, the {obj1.name} is below the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} below the {obj1.name}?",
                f"No, the {obj2.name} is above the {obj1.name}.",
                [obj1_id, obj2_id]
            )
        
        # X-axis (left/right)
        if x1 > x2:
            yield (
                f"Is the {obj1.name} to the right of the {obj2.name}?",
                f"Yes, the {obj1.name} is to the right of the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} to the right of the {obj1.name}?",
                f"No, the {obj2.name} is to the left of the {obj1.name}.",
                [obj1_id, obj2_id]
            )
        elif x1 < x2:
            yield (
                f"Is the {obj1.name} to the left of the {obj2.name}?",
                f"Yes, the {obj1.name} is to the left of the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} to the left of the {obj1.name}?",
                f"No, the {obj2.name} is to the right of the {obj1.name}.",
                [obj1_id, obj2_id]
            )
        
        # Z-axis (front/behind)
        if z1 > z2:
            yield (
                f"Is the {obj1.name} in front of the {obj2.name}?",
                f"Yes, the {obj1.name} is in front of the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} in front of the {obj1.name}?",
                f"No, the {obj2.name} is behind the {obj1.name}.",
                [obj1_id, obj2_id]
            )
        elif z1 < z2:
            yield (
                f"Is the {obj1.name} behind the {obj2.name}?",
                f"Yes, the {obj1.name} is behind the {obj2.name}.",
                [obj1_id, obj2_id]
            )
            yield (
                f"Is the {obj2.name} behind the {obj1.name}?",
                f"No, the {obj2.name} is in front of the {obj1.name}.",
                [obj1_id, obj2_id]
            )


def generate_grounding_examples(scene):
    """Generate simple NP grounding examples (noun phrase → scene object).
    
    Args:
        scene: SceneModel with 4 objects
    
    Yields:
        Tuples of (question_str, answer_str, obj_ids)
    """
    objs = scene.objects
    
    for obj in objs:
        # Simple identification
        yield (
            f"What is the {obj.name}?",
            f"It is a {obj.name} called {obj.object_id}.",
            [obj.object_id]
        )
        
        # Color-based grounding
        if obj.vector['red'] > 0.5:
            yield (
                f"What is red?",
                f"The {obj.name} is red.",
                [obj.object_id]
            )
        if obj.vector['blue'] > 0.5:
            yield (
                f"What is blue?",
                f"The {obj.name} is blue.",
                [obj.object_id]
            )
        if obj.vector['green'] > 0.5:
            yield (
                f"What is green?",
                f"The {obj.name} is green.",
                [obj.object_id]
            )


def create_mock_hypothesis_for_question(question, answer, obj_ids, scene):
    """Create a mock Layer-6 hypothesis for a question.
    
    NOTE: This is a *placeholder* that demonstrates the format.
    In the real system, you'd run the question through LATN Layers 1-5
    to get the actual Layer-6 representation.
    
    Args:
        question: Question string
        answer: Answer string
        obj_ids: List of grounded object IDs
        scene: SceneModel for looking up objects
    
    Returns:
        dict: Training pair (input as mock, since we don't have LATN yet)
    """
    from engraf.lexer.hypothesis import TokenizationHypothesis
    from engraf.lexer.vector_space import VectorSpace
    import numpy as np
    
    # Create a hypothesis
    hyp = TokenizationHypothesis(
        tokens=[VectorSpace() for _ in range(len(question.split()))],
        confidence=0.95,
        description=f"Synthetic: {question}"
    )
    
    # Initialize Layer-6
    hyp.initialize_layer6_structural()
    
    # Mock: add NPs for each grounded object
    obj_map = {obj.object_id: obj for obj in scene.objects}
    for obj_id in obj_ids:
        if obj_id in obj_map:
            obj = obj_map[obj_id]
            # Add a mock NP phrase
            hyp.add_layer6_phrase(
                phrase_type="NP",
                phrase_vector=obj.vector.as_numpy_array() if hasattr(obj.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM),
                scene_object=obj
            )
    
    # Wrap in VP then SP
    if len(hyp.layer6_tokens) > 0:
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="VP",
            phrase_vector=np.random.randn(SEMANTIC_VECTOR_DIM),
            scene_object=None
        )
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="SP",
            phrase_vector=np.random.randn(SEMANTIC_VECTOR_DIM),
            scene_object=None
        )
    
    # Convert to training pair
    pair = create_training_pair_from_hyp(hyp, answer)
    pair['question'] = question  # Add question for reference
    
    return pair


def generate_dataset(output_path="layer6_training_data.jsonl", num_examples=None):
    """Generate and save training dataset from demo scene.
    
    Args:
        output_path: Path to write JSONL file
        num_examples: Max number of examples to generate (None = all)
    """
    print("=" * 70)
    print("Generating Layer-6 Training Dataset from Demo Scene")
    print("=" * 70)
    print()
    
    # Setup scene
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    # Generate all questions
    all_q_a = list(generate_spatial_questions(scene))
    all_q_a.extend(generate_grounding_examples(scene))
    
    if num_examples:
        all_q_a = all_q_a[:num_examples]
    
    print(f"Generated {len(all_q_a)} question/answer pairs")
    print()
    
    # Convert to training pairs
    print("Converting to Layer-6 training format...")
    examples = []
    for i, (question, answer, obj_ids) in enumerate(all_q_a):
        try:
            pair = create_mock_hypothesis_for_question(question, answer, obj_ids, scene)
            examples.append(pair)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(all_q_a)} examples processed")
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            continue
    
    print()
    print(f"Successfully converted {len(examples)} examples")
    print()
    
    # Write to JSONL
    print(f"Writing to {output_path}...")
    write_jsonl(output_path, examples)
    print(f"✓ Wrote {len(examples)} examples to {output_path}")
    print()
    
    # Show a sample
    if examples:
        print("=" * 70)
        print("SAMPLE TRAINING EXAMPLE")
        print("=" * 70)
        ex = examples[0]
        print(f"Question: {ex.get('question', 'N/A')}")
        print(f"Input (Layer-6): {ex['input_string']}")
        print(f"Target: {ex['target_string']}")
        print(f"Structural tokens: {ex['structural_tokens']}")
        print(f"Scene grounding: {[str(r) if r else None for r in ex['scene_grounding']]}")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Layer-6 training dataset')
    parser.add_argument('--output', default='layer6_training_data.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max examples to generate')
    args = parser.parse_args()
    
    generate_dataset(output_path=args.output, num_examples=args.limit)
