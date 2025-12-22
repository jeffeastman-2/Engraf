#!/usr/bin/env python3
"""
Generate expanded training dataset with variations on base examples.

Takes the Layer-6 dataset and creates variations with different
objects, prepositions, and question formats.
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import json
from pathlib import Path
from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.llm_layer6.dataset_generator_l5_v2 import process_question_through_layer5, populate_layer6_from_sentence_phrase
from engraf.llm_layer6.dataset_extractor import create_training_pair_from_hyp, write_jsonl


def expand_dataset(source_file, output_file, num_variations=3):
    """
    Load a dataset and create variations by modifying sentences.
    
    Since we have limited unique scene configurations, we create
    variations by:
    1. Changing question phrasing (imperative vs interrogative)
    2. Adding negations for false examples
    3. Changing object combinations
    """
    print("=" * 70)
    print("Expanding Layer-6 Dataset with Variations")
    print("=" * 70)
    print()
    
    # Load existing examples
    with open(source_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(examples)} base examples from {source_file}")
    print()
    
    # Setup scene for generating new examples
    scene = setup_demo_scene()
    executor = LATNLayerExecutor(scene)
    
    # Base question patterns with answers
    base_patterns = [
        # Imperatives
        ("move the {obj1} {prep} the {obj2}", "Moving the {obj1} {prep} the {obj2}."),
        ("place the {obj1} {prep} the {obj2}", "Placing the {obj1} {prep} the {obj2}."),
        
        # Interrogatives
        ("is the {obj1} {prep} the {obj2}", "Yes, the {obj1} is {prep} the {obj2}."),
        ("is the {obj2} {prep} the {obj1}", "No, the {obj2} is not {prep} the {obj1}."),
        
        # Declaratives
        ("the {obj1} is {prep} the {obj2}", "The {obj1} is {prep} the {obj2}."),
    ]
    
    # Objects available in scene
    objects = ['sphere', 'cube', 'cylinder', 'table']
    
    # Spatial prepositions
    prepositions = ['above', 'below', 'on', 'left of', 'right of', 'in front of', 'behind']
    
    expanded = list(examples)
    
    print(f"Generating {num_variations} variations per base pattern...")
    print(f"Base patterns: {len(base_patterns)}")
    print(f"Objects: {len(objects)}")
    print(f"Prepositions: {len(prepositions)}")
    print()
    
    generated_count = 0
    for pattern_idx, (question_template, answer_template) in enumerate(base_patterns):
        print(f"Pattern {pattern_idx + 1}/{len(base_patterns)}: {question_template[:40]}")
        
        # Generate variations
        for obj_idx, (obj1, obj2) in enumerate([
            ('sphere', 'cube'),
            ('cube', 'table'),
            ('cylinder', 'sphere'),
            ('table', 'cube'),
            ('sphere', 'table'),
        ][:num_variations]):
            
            for prep_idx, prep in enumerate(prepositions[:3]):  # Use first 3 prepositions per combo
                question = question_template.format(obj1=obj1, obj2=obj2, prep=prep)
                answer = answer_template.format(obj1=obj1, obj2=obj2, prep=prep)
                
                try:
                    hyp = process_question_through_layer5(executor, question, scene)
                    
                    if hyp and hyp.layer6_tokens:
                        pair = create_training_pair_from_hyp(hyp, answer)
                        pair['question'] = question
                        
                        # Check if we already have this example
                        if not any(ex['question'] == question for ex in expanded):
                            expanded.append(pair)
                            generated_count += 1
                except Exception as e:
                    pass  # Skip failed generations
        
        print(f"  Generated {generated_count} new examples so far")
    
    print()
    print(f"Total expanded examples: {len(expanded)}")
    print(f"New examples created: {generated_count}")
    print()
    
    # Write expanded dataset
    print(f"Writing to {output_file}...")
    write_jsonl(output_file, expanded)
    print(f"✓ Wrote {len(expanded)} total examples")
    print()
    
    # Stats
    print("=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    print(f"Total examples: {len(expanded)}")
    print(f"Vocabulary size: {len(set(word for ex in expanded for word in ex['target_string'].lower().split()))}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='layer6_training_data_large.jsonl')
    parser.add_argument('--output', default='layer6_training_data_expanded.jsonl')
    parser.add_argument('--variations', type=int, default=3)
    args = parser.parse_args()
    
    expand_dataset(args.source, args.output, args.variations)
