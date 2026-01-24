#!/usr/bin/env python3
"""
Expanded Synthetic Dataset Generator for Layer-6 LLM

Generates comprehensive training examples by:
1. Creating random scene configurations with diverse objects
2. Generating all sentence types: imperatives, declaratives, interrogatives
3. Processing through LATN L1-L5 to get real structural representations
4. Using combinatorial expansion for maximum coverage

This replaces the simpler SyntheticLayer6Dataset with a more robust generator
that produces Layer-6 compatible training data.
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import random
import itertools
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.vector_space import vector_from_features, VECTOR_LENGTH
from engraf.llm_layer6.dataset_extractor import create_training_pair_from_hyp, write_jsonl

# Semantic vector dimension
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH  # Currently 69


# =============================================================================
# VOCABULARY DEFINITIONS
# =============================================================================

# Shape nouns for scene objects
SHAPE_NOUNS = [
    'cube', 'box', 'sphere', 'table', 'cylinder', 'cone', 
    'pyramid', 'prism', 'triangle', 'circle', 'rectangle'
]

# Color adjectives
COLOR_ADJECTIVES = [
    'red', 'green', 'blue', 'yellow', 'purple', 'orange',
    'black', 'white', 'gray', 'brown'
]

# Size adjectives
SIZE_ADJECTIVES = [
    'large', 'big', 'huge', 'small', 'tiny', 'tall', 'short', 'wide'
]

# Intensity adverbs
INTENSITY_ADVERBS = ['very', 'extremely', 'slightly']

# Spatial prepositions (pure location)
SPATIAL_PREPS_LOCATION = [
    'above', 'below', 'over', 'under', 
    'behind', 'in front of', 'left of', 'right of'
]

# Spatial prepositions (proximity/contact)
SPATIAL_PREPS_PROXIMITY = ['on', 'near', 'at']

# All spatial prepositions combined
ALL_SPATIAL_PREPS = SPATIAL_PREPS_LOCATION + SPATIAL_PREPS_PROXIMITY

# Action verbs for imperatives
ACTION_VERBS = ['move', 'place', 'position']

# =============================================================================
# SCENE GENERATION
# =============================================================================

def generate_random_position() -> Tuple[float, float, float]:
    """Generate a random 3D position within reasonable bounds."""
    return (
        random.uniform(-5, 5),   # X
        random.uniform(0, 5),    # Y (usually positive for above-ground)
        random.uniform(-5, 5)    # Z
    )


def create_scene_object(name: str, color: str, size: Optional[str], 
                        position: Tuple[float, float, float],
                        object_id: str) -> SceneObject:
    """Create a SceneObject with the given properties."""
    # Build vector features
    color_values = {
        'red': [1.0, 0.0, 0.0],
        'green': [0.0, 1.0, 0.0],
        'blue': [0.0, 0.0, 1.0],
        'yellow': [1.0, 1.0, 0.0],
        'purple': [0.5, 0.0, 0.5],
        'orange': [1.0, 0.5, 0.0],
        'black': [0.0, 0.0, 0.0],
        'white': [1.0, 1.0, 1.0],
        'gray': [0.5, 0.5, 0.5],
        'brown': [0.6, 0.3, 0.1],
    }
    
    rgb = color_values.get(color, [0.5, 0.5, 0.5])
    
    vector = vector_from_features(
        "noun",
        red=rgb[0], green=rgb[1], blue=rgb[2],
        locX=position[0], locY=position[1], locZ=position[2]
    )
    
    return SceneObject(name, vector, object_id=object_id)


def generate_random_scene(num_objects: int = 4) -> SceneModel:
    """Generate a random scene with diverse objects at random positions."""
    scene = SceneModel()
    
    used_combinations = set()
    
    for i in range(num_objects):
        # Pick unique color+shape combination
        while True:
            shape = random.choice(SHAPE_NOUNS)
            color = random.choice(COLOR_ADJECTIVES)
            combo = (color, shape)
            if combo not in used_combinations:
                used_combinations.add(combo)
                break
        
        position = generate_random_position()
        object_id = f"{color}_{shape}_{i+1}"
        
        obj = create_scene_object(shape, color, None, position, object_id)
        scene.add_object(obj)
    
    return scene


# =============================================================================
# OBJECT DESCRIPTION GENERATION
# =============================================================================

def generate_object_description(color: str, shape: str, 
                                 size: Optional[str] = None,
                                 adverb: Optional[str] = None) -> str:
    """Generate a natural language description of an object.
    
    Examples:
        - "red cube"
        - "large blue sphere"
        - "very tall green cylinder"
    """
    parts = []
    
    if adverb and size:
        parts.append(adverb)
    if size:
        parts.append(size)
    parts.append(color)
    parts.append(shape)
    
    return " ".join(parts)


def parse_object_id_to_description(object_id: str) -> str:
    """Convert object_id like 'red_cube_1' to description 'red cube'."""
    parts = object_id.rsplit('_', 1)  # Remove the _N suffix
    if len(parts) == 2:
        color_shape = parts[0]
        return color_shape.replace('_', ' ')
    return object_id


# =============================================================================
# SENTENCE GENERATION
# =============================================================================

class SentenceGenerator:
    """Generates diverse spatial reasoning sentences."""
    
    def __init__(self, scene: SceneModel):
        self.scene = scene
        self.objects = list(scene.objects)
    
    def get_object_pairs(self) -> List[Tuple[SceneObject, SceneObject]]:
        """Get all ordered pairs of objects (A, B) where A != B."""
        return [(a, b) for a in self.objects for b in self.objects if a != b]
    
    def compute_spatial_relationship(self, obj1: SceneObject, obj2: SceneObject) -> Dict[str, bool]:
        """Compute actual spatial relationships between two objects."""
        x1, y1, z1 = obj1.vector['locX'], obj1.vector['locY'], obj1.vector['locZ']
        x2, y2, z2 = obj2.vector['locX'], obj2.vector['locY'], obj2.vector['locZ']
        
        return {
            'above': y1 > y2 + 0.1,
            'below': y1 < y2 - 0.1,
            'over': y1 > y2 + 0.1,
            'under': y1 < y2 - 0.1,
            'left of': x1 < x2 - 0.1,
            'right of': x1 > x2 + 0.1,
            'in front of': z1 > z2 + 0.1,
            'behind': z1 < z2 - 0.1,
            'on': abs(y1 - y2) < 0.5 and abs(x1 - x2) < 1.0 and abs(z1 - z2) < 1.0,
            'near': np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) < 3.0,
            'at': np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) < 1.0,
        }
    
    # -------------------------------------------------------------------------
    # IMPERATIVE SENTENCES (commands)
    # -------------------------------------------------------------------------
    
    def generate_imperatives(self) -> List[Tuple[str, str, List[str]]]:
        """Generate imperative sentences like 'move the cube above the table'.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        examples = []
        
        # Verb to gerund mapping for proper grammar
        verb_to_gerund = {
            'move': 'Moving',
            'place': 'Placing',
            'position': 'Positioning'
        }
        
        for obj1, obj2 in self.get_object_pairs():
            desc1 = parse_object_id_to_description(obj1.object_id)
            desc2 = parse_object_id_to_description(obj2.object_id)
            
            for verb in ACTION_VERBS:
                for prep in ALL_SPATIAL_PREPS:
                    sentence = f"{verb} the {desc1} {prep} the {desc2}"
                    # Use proper gerund form
                    gerund = verb_to_gerund.get(verb, verb.capitalize() + 'ing')
                    answer = f"{gerund} the {desc1} {prep} the {desc2}."
                    examples.append((sentence, answer, [obj1.object_id, obj2.object_id]))
        
        return examples
    
    # -------------------------------------------------------------------------
    # DECLARATIVE SENTENCES (statements)
    # -------------------------------------------------------------------------
    
    def generate_declaratives(self) -> List[Tuple[str, str, List[str]]]:
        """Generate declarative sentences like 'the cube is above the table'.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        examples = []
        
        for obj1, obj2 in self.get_object_pairs():
            desc1 = parse_object_id_to_description(obj1.object_id)
            desc2 = parse_object_id_to_description(obj2.object_id)
            relationships = self.compute_spatial_relationship(obj1, obj2)
            
            for prep in ALL_SPATIAL_PREPS:
                sentence = f"the {desc1} is {prep} the {desc2}"
                
                # Generate appropriate answer based on actual relationship
                is_true = relationships.get(prep, False)
                if is_true:
                    answer = f"The {desc1} is {prep} the {desc2}."
                else:
                    answer = f"The {desc1} is not {prep} the {desc2}."
                
                examples.append((sentence, answer, [obj1.object_id, obj2.object_id]))
        
        return examples
    
    # -------------------------------------------------------------------------
    # INTERROGATIVE SENTENCES (questions)
    # -------------------------------------------------------------------------
    
    def generate_interrogatives(self) -> List[Tuple[str, str, List[str]]]:
        """Generate interrogative sentences like 'is the cube above the table'.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        examples = []
        
        for obj1, obj2 in self.get_object_pairs():
            desc1 = parse_object_id_to_description(obj1.object_id)
            desc2 = parse_object_id_to_description(obj2.object_id)
            relationships = self.compute_spatial_relationship(obj1, obj2)
            
            for prep in ALL_SPATIAL_PREPS:
                sentence = f"is the {desc1} {prep} the {desc2}"
                
                # Generate yes/no answer based on actual relationship
                is_true = relationships.get(prep, False)
                if is_true:
                    answer = f"Yes, the {desc1} is {prep} the {desc2}."
                else:
                    answer = f"No, the {desc1} is not {prep} the {desc2}."
                
                examples.append((sentence, answer, [obj1.object_id, obj2.object_id]))
        
        return examples
    
    # -------------------------------------------------------------------------
    # WHAT/WHERE QUESTIONS
    # -------------------------------------------------------------------------
    
    def generate_what_questions(self) -> List[Tuple[str, str, List[str]]]:
        """Generate 'what is X' style questions.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        examples = []
        
        for obj in self.objects:
            desc = parse_object_id_to_description(obj.object_id)
            
            # What is the [color] [shape]?
            sentence = f"what is the {desc}"
            answer = f"The {desc} is an object in the scene."
            examples.append((sentence, answer, [obj.object_id]))
            
            # What color is the [shape]?
            parts = desc.split()
            if len(parts) >= 2:
                color, shape = parts[0], parts[-1]
                sentence = f"what color is the {shape}"
                answer = f"The {shape} is {color}."
                examples.append((sentence, answer, [obj.object_id]))
        
        return examples
    
    def generate_where_questions(self) -> List[Tuple[str, str, List[str]]]:
        """Generate 'where is X' style questions.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        examples = []
        
        for obj in self.objects:
            desc = parse_object_id_to_description(obj.object_id)
            pos = [obj.vector['locX'], obj.vector['locY'], obj.vector['locZ']]
            
            sentence = f"where is the {desc}"
            answer = f"The {desc} is at position [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]."
            examples.append((sentence, answer, [obj.object_id]))
        
        return examples
    
    # -------------------------------------------------------------------------
    # COMBINED GENERATION
    # -------------------------------------------------------------------------
    
    def generate_all(self, include_imperatives: bool = True,
                    include_declaratives: bool = True,
                    include_interrogatives: bool = True,
                    include_what_where: bool = True) -> List[Tuple[str, str, List[str]]]:
        """Generate all sentence types.
        
        Returns:
            List of (sentence, expected_answer, object_ids)
        """
        all_examples = []
        
        if include_imperatives:
            all_examples.extend(self.generate_imperatives())
        
        if include_declaratives:
            all_examples.extend(self.generate_declaratives())
        
        if include_interrogatives:
            all_examples.extend(self.generate_interrogatives())
        
        if include_what_where:
            all_examples.extend(self.generate_what_questions())
            all_examples.extend(self.generate_where_questions())
        
        return all_examples


# =============================================================================
# LAYER-6 PROCESSING
# =============================================================================

def populate_layer6_from_sentence_phrase(hyp, sentence_phrase):
    """Extract Layer-6 structure from a parsed SentencePhrase.
    
    Recursively traverses the SentencePhrase tree and builds Layer-6
    representation with real vectors from the parse tree.
    
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
            vec = np_phrase.vector.as_numpy_array() if hasattr(np_phrase.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
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
            pp_vec = pp.vector.as_numpy_array() if hasattr(pp.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
            
            # First add the nested NP in the PP
            if pp.noun_phrase:
                np_phrase = pp.noun_phrase
                np_vec = np_phrase.vector.as_numpy_array() if hasattr(np_phrase.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
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
            vp_vec = vp.vector.as_numpy_array() if hasattr(vp.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
            hyp.wrap_layer6_with_phrase(
                start_idx=0,
                end_idx=len(hyp.layer6_tokens) - 1,
                phrase_type="VP",
                phrase_vector=vp_vec,
                scene_object=None
            )
    
    # Wrap everything with SP
    if len(hyp.layer6_tokens) > 0:
        sp_vec = hyp.tokens[0].as_numpy_array() if hasattr(hyp.tokens[0], 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="SP",
            phrase_vector=sp_vec,
            scene_object=None
        )


def process_through_layer5(executor, sentence: str, scene: SceneModel):
    """Run a sentence through LATN Layers 1-5.
    
    Args:
        executor: LATNLayerExecutor instance
        sentence: Sentence string
        scene: SceneModel
    
    Returns:
        TokenizationHypothesis with Layer-6 populated, or None if failed
    """
    try:
        result = executor.execute_layer5(sentence, report=False)
        
        if not result.success or not result.hypotheses:
            return None
        
        hyp = result.hypotheses[0]
        
        # Extract the SentencePhrase from the final token
        if hyp.tokens and hasattr(hyp.tokens[0], 'phrase'):
            sentence_phrase = hyp.tokens[0].phrase
            populate_layer6_from_sentence_phrase(hyp, sentence_phrase)
            
            if hyp.layer6_tokens:
                return hyp
        
        return None
    except Exception as e:
        return None


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_synthetic_dataset(
    output_path: str = "layer6_synthetic_dataset.jsonl",
    num_scenes: int = 10,
    objects_per_scene: int = 4,
    process_through_latn: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generate a comprehensive synthetic training dataset.
    
    Args:
        output_path: Path to write JSONL output
        num_scenes: Number of random scenes to generate
        objects_per_scene: Objects per scene
        process_through_latn: Whether to process through LATN L1-L5
        max_examples: Optional limit on total examples
        seed: Random seed for reproducibility
    
    Returns:
        List of training examples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("Generating Synthetic Layer-6 Training Dataset")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Scenes: {num_scenes}")
    print(f"  Objects per scene: {objects_per_scene}")
    print(f"  Process through LATN: {process_through_latn}")
    print(f"  Max examples: {max_examples or 'unlimited'}")
    print()
    
    # Initialize LATN executor if needed
    executor = None
    if process_through_latn:
        try:
            from engraf.lexer.latn_layer_executor import LATNLayerExecutor
            # We'll create executor per scene
        except ImportError:
            print("Warning: LATNLayerExecutor not available, using mock hypotheses")
            process_through_latn = False
    
    all_examples = []
    
    for scene_idx in range(num_scenes):
        print(f"Scene {scene_idx + 1}/{num_scenes}...")
        
        # Generate random scene
        scene = generate_random_scene(objects_per_scene)
        
        # Create executor for this scene
        if process_through_latn:
            from engraf.lexer.latn_layer_executor import LATNLayerExecutor
            executor = LATNLayerExecutor(scene)
        
        # Generate sentences
        generator = SentenceGenerator(scene)
        scene_sentences = generator.generate_all()
        
        print(f"  Generated {len(scene_sentences)} sentence variations")
        
        # Process each sentence
        processed = 0
        for sentence, answer, obj_ids in scene_sentences:
            if max_examples and len(all_examples) >= max_examples:
                break
            
            try:
                if process_through_latn:
                    hyp = process_through_layer5(executor, sentence, scene)
                    
                    if hyp and hyp.layer6_tokens:
                        pair = create_training_pair_from_hyp(hyp, answer)
                        pair['question'] = sentence
                        pair['expected_objects'] = obj_ids
                        all_examples.append(pair)
                        processed += 1
                else:
                    # Create mock example without LATN processing
                    example = create_mock_example(sentence, answer, obj_ids, scene)
                    all_examples.append(example)
                    processed += 1
                    
            except Exception as e:
                continue
        
        print(f"  Successfully processed: {processed}")
        
        if max_examples and len(all_examples) >= max_examples:
            print(f"\nReached max examples limit ({max_examples})")
            break
    
    print()
    print(f"Total examples generated: {len(all_examples)}")
    
    # Write to file
    if output_path:
        print(f"Writing to {output_path}...")
        write_jsonl(output_path, all_examples)
        print(f"✓ Wrote {len(all_examples)} examples")
    
    print()
    
    # Print statistics
    print_dataset_stats(all_examples)
    
    return all_examples


def create_mock_example(sentence: str, answer: str, obj_ids: List[str], 
                        scene: SceneModel) -> Dict[str, Any]:
    """Create a mock training example without LATN processing.
    
    Used as fallback when LATN executor is not available.
    """
    from engraf.lexer.hypothesis import TokenizationHypothesis
    from engraf.lexer.vector_space import VectorSpace
    
    # Create hypothesis with mock tokens
    hyp = TokenizationHypothesis(
        tokens=[VectorSpace() for _ in range(len(sentence.split()))],
        confidence=0.95,
        description=f"Mock: {sentence}"
    )
    
    # Initialize Layer-6
    hyp.initialize_layer6_structural()
    
    # Add NPs for each referenced object
    obj_map = {obj.object_id: obj for obj in scene.objects}
    for obj_id in obj_ids:
        if obj_id in obj_map:
            obj = obj_map[obj_id]
            vec = obj.vector.as_numpy_array() if hasattr(obj.vector, 'as_numpy_array') else np.zeros(SEMANTIC_VECTOR_DIM)
            hyp.add_layer6_phrase("NP", vec, obj)
    
    # Wrap in VP then SP
    if len(hyp.layer6_tokens) > 0:
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="VP",
            phrase_vector=np.zeros(SEMANTIC_VECTOR_DIM),
            scene_object=None
        )
        hyp.wrap_layer6_with_phrase(
            start_idx=0,
            end_idx=len(hyp.layer6_tokens) - 1,
            phrase_type="SP",
            phrase_vector=np.zeros(SEMANTIC_VECTOR_DIM),
            scene_object=None
        )
    
    pair = create_training_pair_from_hyp(hyp, answer)
    pair['question'] = sentence
    pair['expected_objects'] = obj_ids
    
    return pair


def print_dataset_stats(examples: List[Dict[str, Any]]):
    """Print statistics about the generated dataset."""
    print("=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    
    # Count by sentence type
    imperatives = sum(1 for ex in examples if ex['question'].split()[0] in ACTION_VERBS)
    interrogatives = sum(1 for ex in examples if ex['question'].startswith('is ') or 
                         ex['question'].startswith('what ') or ex['question'].startswith('where '))
    declaratives = len(examples) - imperatives - interrogatives
    
    print(f"Imperatives: {imperatives}")
    print(f"Declaratives: {declaratives}")
    print(f"Interrogatives: {interrogatives}")
    
    # Vocabulary stats
    all_words = set()
    for ex in examples:
        all_words.update(ex['question'].lower().split())
        all_words.update(ex['target_string'].lower().split())
    
    print(f"Vocabulary size: {len(all_words)} unique words")
    print()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic Layer-6 training data')
    parser.add_argument('--output', default='layer6_synthetic_dataset.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--scenes', type=int, default=10,
                        help='Number of random scenes to generate')
    parser.add_argument('--objects', type=int, default=4,
                        help='Objects per scene')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Maximum total examples')
    parser.add_argument('--no-latn', action='store_true',
                        help='Skip LATN processing (use mock hypotheses)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        output_path=args.output,
        num_scenes=args.scenes,
        objects_per_scene=args.objects,
        process_through_latn=not args.no_latn,
        max_examples=args.max_examples,
        seed=args.seed
    )
