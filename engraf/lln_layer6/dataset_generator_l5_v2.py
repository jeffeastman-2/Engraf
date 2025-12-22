#!/usr/bin/env python3
"""
Dataset generator using ACTUAL Layer 5 output with proper structure traversal.

This correctly extracts NPs/PPs/VPs from the SentencePhrase structure
and uses real vectors from the LATN pipeline.
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

from engraf.An_N_Space_Model.demo_scene_setup import setup_demo_scene, print_scene_info
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.llm_layer6.dataset_extractor import create_training_pair_from_hyp, write_jsonl
import numpy as np


def populate_layer6_from_sentence_phrase(hyp, sentence_phrase):
    """Extract Layer-6 structure from a parsed SentencePhrase.
    
    Recursively traverses the SentencePhrase tree and builds Layer-6
    representation with REAL vectors from the parse tree.
    
    Args:
        hyp: TokenizationHypothesis to populate
        sentence_phrase: SentencePhrase from Layer 5
    """
    hyp.initialize_layer6_structural()
    
    # Process the predicate (which contains NPs and PPs)
    if sentence_phrase.predicate:
        vp = sentence_phrase.predicate
        
        # Add the main NP (direct object)
        if vp.noun_phrase:
            np_phrase = vp.noun_phrase
            vec = np_phrase.vector.as_numpy_array() if hasattr(np_phrase.vector, 'as_numpy_array') else np.zeros(76)
            grounding = np_phrase.grounding if hasattr(np_phrase, 'grounding') else None
            scene_obj = None
            
            if grounding and isinstance(grounding, dict):
                if 'scene_objects' in grounding and grounding['scene_objects']:
                    scene_obj = grounding['scene_objects'][0]
                elif 'scene_object' in grounding:
                    scene_obj = grounding['scene_object']
            
            hyp.add_layer6_phrase("NP", vec, scene_obj)
        
        # Add prepositional phrases
        for pp in vp.prepositions:
            # Add opening PP marker
            pp_vec = pp.vector.as_numpy_array() if hasattr(pp.vector, 'as_numpy_array') else np.zeros(76)
            
            # First add the nested NP in the PP
            if pp.noun_phrase:
                np_phrase = pp.noun_phrase
                np_vec = np_phrase.vector.as_numpy_array() if hasattr(np_phrase.vector, 'as_numpy_array') else np.zeros(76)
                np_grounding = np_phrase.grounding if hasattr(np_phrase, 'grounding') else None
                np_scene_obj = None
                
                if np_grounding and isinstance(np_grounding, dict):
                    if 'scene_objects' in np_grounding and np_grounding['scene_objects']:
                        np_scene_obj = np_grounding['scene_objects'][0]
                    elif 'scene_object' in np_grounding:
                        np_scene_obj = np_grounding['scene_object']
                
                hyp.add_layer6_phrase("NP", np_vec, np_scene_obj)
            
            # Now wrap the NP with PP
            if len(hyp.layer6_tokens) >= 2:
                hyp.wrap_layer6_with_phrase(
                    start_idx=len(hyp.layer6_tokens) - 2,
                    end_idx=len(hyp.layer6_tokens) - 1,
                    phrase_type="PP",
                    phrase_vector=pp_vec,
                    scene_object=None
                )
        
        # Wrap everything with VP
        if len(hyp.layer6_tokens) > 0:
            vp_vec = vp.vector.as_numpy_array() if hasattr(vp.vector, 'as_numpy_array') else np.zeros(76)
            hyp.wrap_layer6_with_phrase(
                start_idx=0,
                end_idx=len(hyp.layer6_tokens) - 1,
                phrase_type="VP",
                phrase_vector=vp_vec,
                scene_object=None
            )
    
    # Wrap everything with SP
    if len(hyp.layer6_tokens) > 0:
        sp_vec = hyp.tokens[0].as_numpy_array() if hasattr(hyp.tokens[0], 'as_numpy_array') else np.zeros(76)
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="SP",
            phrase_vector=sp_vec,
            scene_object=None
        )


def process_question_through_layer5(executor, question, scene):
    """Run a question through LATN Layers 1-5.
    
    Args:
        executor: LATNLayerExecutor instance
        question: Question/imperative string
        scene: SceneModel
    
    Returns:
        TokenizationHypothesis with Layer-6 populated from L5 parse tree
    """
    try:
        result = executor.execute_layer5(question, report=False)
        
        if not result.success or not result.hypotheses:
            return None
        
        hyp = result.hypotheses[0]
        
        # Extract the SentencePhrase from the final token
        if hyp.tokens and hasattr(hyp.tokens[0], 'phrase'):
            sentence_phrase = hyp.tokens[0].phrase
            
            # Populate Layer-6 from the sentence structure
            populate_layer6_from_sentence_phrase(hyp, sentence_phrase)
            
            # Only return if we got some Layer-6 tokens
            if hyp.layer6_tokens:
                return hyp
        
        return None
    except Exception as e:
        return None


def generate_dataset_with_layer5_real(output_path="layer6_training_data_l5.jsonl", num_examples=None):
    """Generate dataset using real Layer 5 parse trees.
    
    Args:
        output_path: Path to write JSONL file
        num_examples: Max examples (None = all)
    """
    print("=" * 70)
    print("Generating Layer-6 Dataset with REAL Layer 5 Parse Trees")
    print("=" * 70)
    print()
    
    # Setup
    scene = setup_demo_scene()
    print_scene_info(scene)
    
    executor = LATNLayerExecutor(scene)
    
    # Test sentences that Layer 5 can parse
    # NOTE: Layer 5 requires PPs/adjuncts for full VP structure
    # Mix of imperatives, declaratives, interrogatives, and false/incorrect statements
    test_sentences = [
        # IMPERATIVES - Action with spatial relationship
        ("move the sphere above the cube", "Moving the sphere above the cube."),
        ("place the cube above the table", "Placing the cube above the table."),
        ("move the cylinder above the sphere", "Moving the cylinder above the sphere."),
        ("place the cube on the table", "Placing the cube on the table."),
        ("place the sphere below the cube", "Placing the sphere below the cube."),
        
        # DECLARATIVES - True spatial relationships
        ("the sphere is above the cube", "The sphere is positioned above the cube."),
        ("the cube is on the table", "The cube is on the table."),
        ("the cylinder is behind the table", "The cylinder is behind the table."),
        ("the sphere is right of the cube", "The sphere is right of the cube."),
        ("the cube is left of the sphere", "The cube is left of the sphere."),
        
        # INTERROGATIVES - Questions about spatial relationships
        ("is the sphere above the cube", "Yes, the sphere is above the cube."),
        ("is the cube on the table", "Yes, the cube is on the table."),
        ("is the cylinder behind the table", "Yes, the cylinder is behind the table."),
        ("is the sphere right of the cube", "Yes, the sphere is right of the cube."),
        
        # FALSE EXAMPLES - Incorrect spatial relationships (negative training)
        # These contradict the actual scene layout with explicit negation responses
        ("move the cube right of the sphere", "No, the cube is not right of the sphere."),  # False: cube is LEFT
        ("place the sphere left of the cube", "No, the sphere is not left of the cube."),  # False: sphere is RIGHT
        ("the cube is below the table", "No, the cube is not below the table."),  # False: cube is ABOVE
        ("the sphere is behind the cube", "No, the sphere is not behind the cube."),  # False: sphere is to the right
        ("the cylinder is above the table", "No, the cylinder is not above the table."),  # False: same height
        ("is the sphere below the table", "No, the sphere is not below the table."),  # False question
        ("is the cube right of the sphere", "No, the cube is not right of the sphere."),  # False question
        
        # IMPERATIVES - More spatial variations
        ("move the cube left of the sphere", "Moving the cube left of the sphere."),
        ("place the sphere right of the cube", "Placing the sphere right of the cube."),
        ("move the cylinder in front of the table", "Moving the cylinder in front of the table."),
        ("place the cube behind the sphere", "Placing the cube behind the sphere."),
        ("move the sphere above the table", "Moving the sphere above the table."),
    ]
    
    if num_examples:
        test_sentences = test_sentences[:num_examples]
    
    print(f"Test sentences: {len(test_sentences)}")
    print()
    print("Processing through LATN Layers 1-5...")
    
    examples = []
    for i, (question, answer) in enumerate(test_sentences):
        print(f"  [{i+1:2d}/{len(test_sentences)}] {question:40s}", end=" ")
        
        hyp = process_question_through_layer5(executor, question, scene)
        
        if hyp and hyp.layer6_tokens:
            try:
                pair = create_training_pair_from_hyp(hyp, answer)
                pair['question'] = question
                examples.append(pair)
                print("✓")
            except Exception as e:
                print(f"✗ ({str(e)[:25]})")
        else:
            print("✗")
    
    print()
    print(f"Successfully processed {len(examples)}/{len(test_sentences)} examples")
    print()
    
    # Write to JSONL
    print(f"Writing to {output_path}...")
    write_jsonl(output_path, examples)
    print(f"✓ Wrote {len(examples)} examples")
    print()
    
    # Show samples
    if examples:
        print("=" * 70)
        print("SAMPLE TRAINING EXAMPLES (Real Layer 5 Vectors)")
        print("=" * 70)
        for idx, ex in enumerate(examples[:2]):
            print()
            print(f"Example {idx+1}:")
            print(f"  Question: {ex['question']}")
            print(f"  Input: {ex['input_string']}")
            print(f"  Target: {ex['target_string']}")
            print(f"  Tokens: {ex['structural_tokens']}")
            print(f"  Grounding: {ex['scene_grounding']}")
            print(f"  Vectors: {len(ex['semantic_vectors'])} × 76 dims")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='layer6_training_data_l5.jsonl')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    generate_dataset_with_layer5_real(output_path=args.output, num_examples=args.limit)
