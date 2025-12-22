#!/usr/bin/env python3
"""
Generate Layer-6 training dataset using ACTUAL Layer 5 output vectors.

This processes questions through LATN Layers 1-5 and converts the resulting
hypotheses to Layer-6 training examples with REAL composed semantic vectors
(not random placeholders).
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.llm_layer6.dataset_extractor import create_training_pair_from_hyp, write_jsonl
import itertools


def generate_synthetic_questions(scene):
    """Generate questions that LATN can actually parse.
    
    Since Layer 5 expects imperative/action sentences, we generate
    those instead of yes/no questions.
    
    Args:
        scene: SceneModel with objects
    
    Yields:
        Tuples of (question_str, answer_str, expected_object_ids)
    """
    objs = {obj.object_id: obj for obj in scene.objects}
    
    # Generate imperative/descriptive phrases that Layer 5 can handle
    questions = [
        ("the red cube", "The red cube is a cube.", ["red_cube_1"]),
        ("the blue sphere", "The blue sphere is a sphere.", ["blue_sphere_1"]),
        ("the green cylinder", "The green cylinder is a cylinder.", ["green_cylinder_1"]),
        ("the table", "The table is a table.", ["table_1"]),
        ("move the sphere above the cube", "Moving the sphere above the cube.", ["blue_sphere_1", "red_cube_1"]),
        ("place the cube on the table", "Placing the cube on the table.", ["red_cube_1", "table_1"]),
        ("move the cube above the table", "Moving the cube above the table.", ["red_cube_1", "table_1"]),
    ]
    
    for q, a, obj_ids in questions:
        yield (q, a, obj_ids)


def populate_layer6_from_hypothesis(hyp):
    """Populate Layer-6 representation from Layer-5 hypothesis tokens.
    
    Builds the structural Layer-6 representation from the final tokens,
    using real vectors from the LATN pipeline.
    
    Args:
        hyp: TokenizationHypothesis from Layer 5
    """
    # Initialize Layer-6
    hyp.initialize_layer6_structural()
    
    # Add tokens from the final hypothesis
    # In Layer-5, each token should be a phrase (NP, PP, VP, or SP)
    for i, token in enumerate(hyp.tokens):
        phrase_type = None
        scene_obj = None
        token_vec = token.vector.as_numpy_array() if hasattr(token.vector, 'as_numpy_array') else None
        
        # Determine phrase type
        if token.isa("NP"):
            phrase_type = "NP"
            # Try to get grounding
            if hasattr(token, 'phrase') and token.phrase:
                phrase = token.phrase
                if hasattr(phrase, 'grounding') and phrase.grounding:
                    if isinstance(phrase.grounding, dict):
                        if 'scene_objects' in phrase.grounding:
                            scene_obj = phrase.grounding['scene_objects'][0] if phrase.grounding['scene_objects'] else None
                        elif 'scene_object' in phrase.grounding:
                            scene_obj = phrase.grounding['scene_object']
        
        # For now, just add simple NP phrases with real vectors
        if phrase_type == "NP" and token_vec is not None:
            hyp.add_layer6_phrase(phrase_type, token_vec, scene_obj)


def process_question_through_layer5(executor, question, scene):
    """Run a question through LATN Layers 1-5 with real vectors.
    
    Args:
        executor: LATNLayerExecutor instance
        question: Question string
        scene: SceneModel
    
    Returns:
        TokenizationHypothesis with Layer-6 populated from real L5 vectors
    """
    try:
        result = executor.execute_layer5(question, report=False)
        
        if not result.success or not result.hypotheses:
            return None
        
        hyp = result.hypotheses[0]
        
        # Populate Layer-6 from the hypothesis
        populate_layer6_from_hypothesis(hyp)
        
        # Only return if we got some Layer-6 tokens
        if hyp.layer6_tokens:
            return hyp
        
        return None
    except Exception as e:
        return None


def generate_dataset_with_layer5(output_path="layer6_training_data_l5.jsonl", num_examples=None):
    """Generate and save training dataset using actual Layer 5 output.
    
    Args:
        output_path: Path to write JSONL file
        num_examples: Max number of examples to generate (None = all)
    """
    print("=" * 70)
    print("Generating Layer-6 Dataset with ACTUAL Layer 5 Output Vectors")
    print("=" * 70)
    print()
    
    # Setup scene and executor
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    executor = LATNLayerExecutor(scene)
    
    # Generate questions
    all_questions = list(generate_synthetic_questions(scene))
    if num_examples:
        all_questions = all_questions[:num_examples]
    
    print(f"Generated {len(all_questions)} questions")
    print()
    
    # Process questions through Layer 5
    print("Processing questions through LATN Layers 1-5...")
    examples = []
    skipped = 0
    
    for i, (question, answer, obj_ids) in enumerate(all_questions):
        print(f"  [{i+1:2d}/{len(all_questions)}] {question[:50]}...", end=" ")
        
        # Run through Layer 5
        hyp = process_question_through_layer5(executor, question, scene)
        
        if hyp and hyp.layer6_tokens:
            try:
                # Convert to training pair
                pair = create_training_pair_from_hyp(hyp, answer)
                pair['question'] = question
                pair['expected_objects'] = obj_ids
                examples.append(pair)
                print("✓")
            except Exception as e:
                print(f"✗ (conversion error: {str(e)[:30]})")
                skipped += 1
        else:
            print("✗ (no L5 output)")
            skipped += 1
    
    print()
    print(f"Successfully processed {len(examples)}/{len(all_questions)} examples")
    if skipped > 0:
        print(f"Skipped: {skipped}")
    print()
    
    # Write to JSONL
    print(f"Writing to {output_path}...")
    write_jsonl(output_path, examples)
    print(f"✓ Wrote {len(examples)} examples to {output_path}")
    print()
    
    # Show a sample
    if examples:
        print("=" * 70)
        print("SAMPLE TRAINING EXAMPLE (with real Layer 5 vectors)")
        print("=" * 70)
        ex = examples[0]
        print(f"Question: {ex.get('question', 'N/A')}")
        print(f"Input (Layer-6): {ex['input_string']}")
        print(f"Target: {ex['target_string']}")
        print(f"Structural tokens: {ex['structural_tokens']}")
        print(f"Scene grounding: {[str(r) if r else None for r in ex['scene_grounding']]}")
        print()
        print("Semantic vectors (actual from Layer 5):")
        print(f"  Total: {len(ex['semantic_vectors'])} vectors × 76 dims each")
        for i, vec in enumerate(ex['semantic_vectors'][:3]):
            non_zeros = sum(1 for v in vec if abs(v) > 0.001)
            print(f"  Vec[{i}]: {non_zeros:3d} non-zero values")
        if len(ex['semantic_vectors']) > 3:
            print(f"  ... and {len(ex['semantic_vectors']) - 3} more")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Layer-6 training dataset with L5 vectors')
    parser.add_argument('--output', default='layer6_training_data_l5.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--limit', type=int, default=10,
                        help='Max examples to generate')
    args = parser.parse_args()
    
    generate_dataset_with_layer5(output_path=args.output, num_examples=args.limit)
