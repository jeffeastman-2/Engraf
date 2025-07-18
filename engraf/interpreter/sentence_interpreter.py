"""
Basic Sentence Interpreter for ENGRAF

This module provides a high-level interface for interpreting English sentences
and converting them into 3D scene operations. It integrates the ATN parsing system
with the VPython renderer and scene management.

Step 1.5 - Basic Sentence Interpreter Implementation
"""

from typing import Optional, List, Dict, Any, Union
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
# VPython renderer will be imported conditionally to avoid hanging
from engraf.visualizer.transforms.transform_interpreter import TransformInterpreter
from engraf.lexer.vector_space import VectorSpace


class SentenceInterpreter:
    """
    Main sentence interpreter that processes natural language commands
    and executes them in a 3D scene environment.
    """
    
    def __init__(self, renderer=None):
        """
        Initialize the sentence interpreter.
        
        Args:
            renderer: The renderer to use for visualization (e.g., VPythonRenderer, MockRenderer)
        """
        if renderer is None:
            # Import VPython renderer only when needed
            from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer
            renderer = VPythonRenderer(headless=True)
        
        self.renderer = renderer
        self.scene = SceneModel()  # Initialize the scene
        self.transform_interpreter = TransformInterpreter()
        self.object_counter = 0
        self.execution_history = []
        
    def interpret(self, sentence: str) -> Dict[str, Any]:
        """
        Interpret a natural language sentence and execute the corresponding actions.
        
        Args:
            sentence: The English sentence to interpret
            
        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Step 1: Tokenize the sentence
            tokens = tokenize(sentence)
            if not tokens:
                return self._create_result(False, "Empty sentence", sentence)
            
            # Step 2: Parse the sentence using ATN
            ts = TokenStream(tokens)
            parsed_sentence = run_sentence(ts)
            
            if parsed_sentence is None:
                return self._create_result(False, "Failed to parse sentence", sentence)
            
            # Step 3: Execute the parsed sentence
            result = self._execute_sentence(parsed_sentence, sentence)
            
            # Step 4: Update the visual scene
            if result['success']:
                self.renderer.render_scene(self.scene)
            
            return result
            
        except Exception as e:
            return self._create_result(False, f"Error interpreting sentence: {str(e)}", sentence)
    
    def _execute_sentence(self, parsed_sentence: SentencePhrase, original_sentence: str) -> Dict[str, Any]:
        """Execute a parsed sentence in the 3D scene."""
        try:
            result = {
                'success': True,
                'message': f"Successfully executed: {original_sentence}",
                'sentence': original_sentence,
                'sentence_parsed': parsed_sentence,
                'objects_created': [],
                'objects_modified': [],
                'actions_performed': []
            }
            
            # Handle different sentence types
            if hasattr(parsed_sentence, 'predicate') and parsed_sentence.predicate:
                predicate_result = self._execute_predicate(parsed_sentence.predicate)
                result.update(predicate_result)
            
            # Handle subject if present
            if hasattr(parsed_sentence, 'subject') and parsed_sentence.subject:
                subject_result = self._execute_subject(parsed_sentence.subject)
                result.update(subject_result)
            
            # Handle tobe sentences (e.g., "the cube is red")
            if hasattr(parsed_sentence, 'tobe') and parsed_sentence.tobe:
                tobe_result = self._execute_tobe_sentence(parsed_sentence)
                result.update(tobe_result)
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            return self._create_result(False, f"Error executing sentence: {str(e)}", original_sentence)
    
    def _execute_predicate(self, predicate: Union[VerbPhrase, ConjunctionPhrase]) -> Dict[str, Any]:
        """Execute a predicate (verb phrase or conjunction of verb phrases)."""
        result = {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
        
        if isinstance(predicate, ConjunctionPhrase):
            # Handle conjunction of predicates (e.g., "draw a cube and move it")
            left_result = self._execute_predicate(predicate.left)
            right_result = self._execute_predicate(predicate.right)
            
            # Merge results
            result['objects_created'].extend(left_result.get('objects_created', []))
            result['objects_created'].extend(right_result.get('objects_created', []))
            result['objects_modified'].extend(left_result.get('objects_modified', []))
            result['objects_modified'].extend(right_result.get('objects_modified', []))
            result['actions_performed'].extend(left_result.get('actions_performed', []))
            result['actions_performed'].extend(right_result.get('actions_performed', []))
            
        elif isinstance(predicate, VerbPhrase):
            # Handle single verb phrase
            result = self._execute_verb_phrase(predicate)
        
        return result
    
    def _execute_verb_phrase(self, vp: VerbPhrase) -> Dict[str, Any]:
        """Execute a single verb phrase."""
        result = {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
        
        verb = vp.verb
        result['actions_performed'].append(verb)
        
        # Handle creation verbs
        if verb in ['draw', 'create', 'make', 'build']:
            created_objects = self._handle_creation_verb(vp)
            result['objects_created'].extend(created_objects)
        
        # Handle modification verbs
        elif verb in ['move', 'color', 'scale', 'rotate', 'resize']:
            modified_objects = self._handle_modification_verb(vp)
            result['objects_modified'].extend(modified_objects)
        
        # Handle other verbs
        else:
            print(f"⚠️  Unknown verb: {verb}")
        
        return result
    
    def _handle_creation_verb(self, vp: VerbPhrase) -> List[str]:
        """Handle creation verbs like 'draw', 'create', 'make'."""
        created_objects = []
        
        if vp.noun_phrase:
            objects = self._extract_objects_from_np(vp.noun_phrase)
            
            for obj_info in objects:
                obj_id = self._create_scene_object(obj_info)
                if obj_id:
                    created_objects.append(obj_id)
        
        return created_objects
    
    def _handle_modification_verb(self, vp: VerbPhrase) -> List[str]:
        """Handle modification verbs like 'move', 'color', 'scale'."""
        modified_objects = []
        
        # Find target objects (could be from noun phrase or pronoun)
        target_objects = self._resolve_target_objects(vp)
        
        for obj_id in target_objects:
            if self._modify_scene_object(obj_id, vp):
                modified_objects.append(obj_id)
        
        return modified_objects
    
    def _extract_objects_from_np(self, np: Union[NounPhrase, ConjunctionPhrase]) -> List[Dict[str, Any]]:
        """Extract object information from a noun phrase or conjunction."""
        objects = []
        
        if isinstance(np, ConjunctionPhrase):
            # Handle conjunctions (e.g., "a cube and a sphere")
            objects.extend(self._extract_objects_from_np(np.left))
            objects.extend(self._extract_objects_from_np(np.right))
        
        elif isinstance(np, NounPhrase):
            # Extract object information from single noun phrase
            obj_info = self._extract_single_object_info(np)
            if obj_info:
                objects.append(obj_info)
        
        return objects
    
    def _extract_single_object_info(self, np: NounPhrase) -> Optional[Dict[str, Any]]:
        """Extract object information from a single noun phrase."""
        if not np.noun:
            return None
        
        # Basic object info
        obj_info = {
            'type': np.noun,
            'determiner': np.determiner,
            'adjectives': getattr(np, 'adjectives', []),
            'vector_space': np.vector if hasattr(np, 'vector') else None
        }
        
        # Handle prepositional phrases for positioning
        if hasattr(np, 'preps') and np.preps:
            obj_info['prepositional_phrases'] = np.preps
        
        return obj_info
    
    def _create_scene_object(self, obj_info: Dict[str, Any]) -> Optional[str]:
        """Create a new scene object from object information."""
        try:
            self.object_counter += 1
            
            # Create descriptive ID that includes key adjectives
            obj_id = self._generate_descriptive_id(obj_info)
            
            # Create vector space for the object
            vector_space = obj_info.get('vector_space') or VectorSpace()
            
            # Apply default properties based on object type
            self._apply_default_properties(vector_space, obj_info)
            
            # Create and add scene object with metadata
            scene_object = SceneObject(
                name=obj_info['type'],  # Store the base noun (e.g., 'cube')
                vector=vector_space,
                object_id=obj_id       # Store the descriptive ID (e.g., 'red_cube_1')
            )
            
            # Store descriptive metadata for later matching
            scene_object.metadata = {
                'type': obj_info['type'],
                'adjectives': obj_info.get('adjectives', []),
                'colors': self._extract_colors_from_vector(vector_space),
                'sizes': self._extract_sizes_from_vector(vector_space),
                'determiner': obj_info.get('determiner')
            }
            
            # Apply prepositional phrases (positioning) if present
            if 'prepositional_phrases' in obj_info:
                for prep_phrase in obj_info['prepositional_phrases']:
                    self._apply_movement(scene_object, prep_phrase)
            
            self.scene.add_object(scene_object)
            
            print(f"✅ Created object: {obj_id}")
            return obj_id
            
        except Exception as e:
            print(f"❌ Failed to create object: {e}")
            return None
    
    def _generate_descriptive_id(self, obj_info: Dict[str, Any]) -> str:
        """Generate a descriptive ID that includes key adjectives."""
        base_type = obj_info['type']
        vector_space = obj_info.get('vector_space')
        
        # Extract key descriptive terms
        descriptors = []
        
        # Add color descriptor
        if vector_space:
            if vector_space['red'] > 0.5:
                descriptors.append('red')
            elif vector_space['green'] > 0.5:
                descriptors.append('green')
            elif vector_space['blue'] > 0.5:
                descriptors.append('blue')
            elif vector_space['red'] > 0.5 and vector_space['green'] > 0.5:
                descriptors.append('yellow')
            elif vector_space['red'] > 0.5 and vector_space['blue'] > 0.5:
                descriptors.append('purple')
            elif vector_space['green'] > 0.5 and vector_space['blue'] > 0.5:
                descriptors.append('cyan')
            
            # Add size descriptor
            if vector_space['scaleX'] > 1.5 or vector_space['scaleY'] > 1.5 or vector_space['scaleZ'] > 1.5:
                descriptors.append('big')
            elif vector_space['scaleX'] < 0.75 or vector_space['scaleY'] < 0.75 or vector_space['scaleZ'] < 0.75:
                descriptors.append('small')
        
        # Create descriptive ID
        if descriptors:
            return f"{'_'.join(descriptors)}_{base_type}_{self.object_counter}"
        else:
            return f"{base_type}_{self.object_counter}"
    
    def _extract_colors_from_vector(self, vector_space: VectorSpace) -> List[str]:
        """Extract color names from vector space."""
        colors = []
        if vector_space['red'] > 0.5:
            colors.append('red')
        if vector_space['green'] > 0.5:
            colors.append('green')
        if vector_space['blue'] > 0.5:
            colors.append('blue')
        return colors
    
    def _extract_sizes_from_vector(self, vector_space: VectorSpace) -> List[str]:
        """Extract size descriptors from vector space."""
        sizes = []
        if vector_space['scaleX'] > 1.5 or vector_space['scaleY'] > 1.5 or vector_space['scaleZ'] > 1.5:
            sizes.append('big')
        elif vector_space['scaleX'] < 0.75 or vector_space['scaleY'] < 0.75 or vector_space['scaleZ'] < 0.75:
            sizes.append('small')
        return sizes
    
    def _apply_default_properties(self, vector_space: VectorSpace, obj_info: Dict[str, Any]):
        """Apply default properties to a vector space based on object information."""
        # Default position
        vector_space['locX'] = 0.0
        vector_space['locY'] = 0.0
        vector_space['locZ'] = 0.0
        
        # Default size if not already set
        if vector_space['scaleX'] == 0.0:
            vector_space['scaleX'] = 1.0
        if vector_space['scaleY'] == 0.0:
            vector_space['scaleY'] = 1.0
        if vector_space['scaleZ'] == 0.0:
            vector_space['scaleZ'] = 1.0
        
        # Default color (white) only if no color is already set
        if vector_space['red'] == 0.0 and vector_space['green'] == 0.0 and vector_space['blue'] == 0.0:
            vector_space['red'] = 1.0
            vector_space['green'] = 1.0
            vector_space['blue'] = 1.0
        
        # Apply adjectives if present
        adjectives = obj_info.get('adjectives', [])
        for adj in adjectives:
            self._apply_adjective(vector_space, adj)
    
    def _apply_adjective(self, vector_space: VectorSpace, adjective: str):
        """Apply an adjective to a vector space."""
        # Color adjectives
        color_map = {
            'red': {'red': 1.0, 'green': 0.0, 'blue': 0.0},
            'green': {'red': 0.0, 'green': 1.0, 'blue': 0.0},
            'blue': {'red': 0.0, 'green': 0.0, 'blue': 1.0},
            'yellow': {'red': 1.0, 'green': 1.0, 'blue': 0.0},
            'purple': {'red': 0.8, 'green': 0.0, 'blue': 0.8},
            'orange': {'red': 1.0, 'green': 0.5, 'blue': 0.0},
            'white': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
            'black': {'red': 0.0, 'green': 0.0, 'blue': 0.0},
        }
        
        if adjective in color_map:
            color = color_map[adjective]
            vector_space['red'] = color['red']
            vector_space['green'] = color['green']
            vector_space['blue'] = color['blue']
        
        # Size adjectives
        elif adjective in ['big', 'large', 'huge']:
            vector_space['scaleX'] = 2.0
            vector_space['scaleY'] = 2.0
            vector_space['scaleZ'] = 2.0
        elif adjective in ['small', 'little', 'tiny']:
            vector_space['scaleX'] = 0.5
            vector_space['scaleY'] = 0.5
            vector_space['scaleZ'] = 0.5
        elif adjective == 'tall':
            vector_space['scaleY'] = 2.0
        elif adjective == 'wide':
            vector_space['scaleX'] = 2.0
            vector_space['scaleZ'] = 2.0
    
    def _resolve_target_objects(self, vp: VerbPhrase) -> List[str]:
        """Resolve target objects for modification verbs."""
        target_objects = []
        
        # Check for pronouns (e.g., "move it")
        if vp.noun_phrase and hasattr(vp.noun_phrase, 'pronoun'):
            # Get the most recently created object
            if self.scene.objects:
                target_objects.append(self.scene.objects[-1].object_id)
        
        # Check for specific noun phrases
        elif vp.noun_phrase:
            # Find objects by type/description
            objects = self._find_objects_by_description(vp.noun_phrase)
            target_objects.extend(objects)
        
        return target_objects
    
    def _find_objects_by_description(self, np: NounPhrase) -> List[str]:
        """Find objects in the scene that match a noun phrase description using vector distance."""
        # Get all objects with their match scores
        object_scores = []
        
        for scene_obj in self.scene.objects:
            if self._object_matches_description(scene_obj, np):
                # Calculate distance for ranking (lower is better)
                distance = 0.0
                if hasattr(np, 'vector') and np.vector:
                    distance = self._calculate_vector_distance(scene_obj.vector, np.vector)
                
                object_scores.append((scene_obj.object_id, distance))
        
        # Sort by distance (best matches first) and return object IDs
        object_scores.sort(key=lambda x: x[1])
        return [obj_id for obj_id, _ in object_scores]
    
    def _object_matches_description(self, scene_obj: SceneObject, np: NounPhrase) -> bool:
        """Check if a scene object matches a noun phrase description using vector distance."""
        # First check if basic noun type matches
        if np.noun and np.noun != scene_obj.name:
            return False
        
        # If no vector information in the noun phrase, just match by type
        if not hasattr(np, 'vector') or not np.vector:
            return True
        
        # Use vector distance calculation for sophisticated matching
        distance = self._calculate_vector_distance(scene_obj.vector, np.vector)
        
        # Lower distance means better match - threshold for acceptance
        MATCH_THRESHOLD = 0.3
        return distance <= MATCH_THRESHOLD
    
    def _calculate_vector_distance(self, obj_vector: VectorSpace, query_vector: VectorSpace) -> float:
        """Calculate sophisticated vector distance between object and query."""
        # Define feature categories with different weights
        feature_weights = {
            # Color features (high weight - very specific)
            'red': 2.0, 'green': 2.0, 'blue': 2.0,
            # Size features (medium weight)
            'scaleX': 1.5, 'scaleY': 1.5, 'scaleZ': 1.5,
            # Position features (low weight - less relevant for matching)
            'locX': 0.5, 'locY': 0.5, 'locZ': 0.5,
            # Shape features (high weight)
            'noun': 2.0, 'adj': 1.0
        }
        
        total_distance = 0.0
        total_weight = 0.0
        
        # Calculate weighted distance for each feature
        for feature, weight in feature_weights.items():
            try:
                obj_value = obj_vector[feature]
                query_value = query_vector[feature]
                
                # Special handling for binary features (colors)
                if feature in ['red', 'green', 'blue']:
                    # For colors, check if both are "on" (>0.5) or both are "off" (<0.5)
                    obj_on = obj_value > 0.5
                    query_on = query_value > 0.5
                    
                    if obj_on == query_on:
                        # Perfect match for this color
                        distance = 0.0
                    else:
                        # Mismatch - one has color, other doesn't
                        distance = 1.0
                
                # Handle scale features
                elif feature.startswith('scale'):
                    # For scale, check if both indicate same size category
                    obj_size = self._categorize_scale(obj_value)
                    query_size = self._categorize_scale(query_value)
                    
                    if obj_size == query_size:
                        distance = 0.0
                    else:
                        distance = abs(obj_value - query_value)
                
                # Handle other features with simple distance
                else:
                    distance = abs(obj_value - query_value)
                
                total_distance += distance * weight
                total_weight += weight
                
            except (ValueError, IndexError):
                # Feature not found in vector, skip it
                continue
        
        # Return normalized distance (0.0 = perfect match, 1.0 = completely different)
        return total_distance / total_weight if total_weight > 0 else 1.0
    
    def _categorize_scale(self, scale_value: float) -> str:
        """Categorize a scale value into size categories."""
        if scale_value >= 1.5:
            return 'large'
        elif scale_value <= 0.75:
            return 'small' 
        else:
            return 'normal'
    
    def _modify_scene_object(self, obj_id: str, vp: VerbPhrase) -> bool:
        """Modify a scene object based on a verb phrase."""
        try:
            # Find the scene object
            scene_obj = None
            for obj in self.scene.objects:
                if obj.object_id == obj_id:
                    scene_obj = obj
                    break
            
            if not scene_obj:
                return False
            
            verb = vp.verb
            
            # Handle different modification verbs
            if verb == 'color' and hasattr(vp, 'adjective_complement'):
                # Color the object
                for adj in vp.adjective_complement:
                    self._apply_adjective(scene_obj.vector_space, adj)
            
            elif verb == 'move' and hasattr(vp, 'preposition'):
                # Move the object
                self._apply_movement(scene_obj, vp.preposition)
            
            elif verb == 'scale' and vp.noun_phrase:
                # Scale the object
                self._apply_scaling(scene_obj, vp.noun_phrase)
            
            print(f"✅ Modified object: {obj_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to modify object {obj_id}: {e}")
            return False
    
    def _apply_movement(self, scene_obj: SceneObject, preposition):
        """Apply movement to an object based on prepositional phrase."""
        # This would integrate with the transform interpreter
        # For now, simple implementation
        if hasattr(preposition, 'noun_phrase') and hasattr(preposition.noun_phrase, 'vector'):
            vector = preposition.noun_phrase.vector
            if vector['locX'] != 0.0:
                scene_obj.vector['locX'] = vector['locX']
            if vector['locY'] != 0.0:
                scene_obj.vector['locY'] = vector['locY']
            if vector['locZ'] != 0.0:
                scene_obj.vector['locZ'] = vector['locZ']
    
    def _apply_scaling(self, scene_obj: SceneObject, noun_phrase: NounPhrase):
        """Apply scaling to an object based on noun phrase."""
        # Simple scaling based on adjectives
        if hasattr(noun_phrase, 'adjectives'):
            for adj in noun_phrase.adjectives:
                self._apply_adjective(scene_obj.vector_space, adj)
    
    def _execute_subject(self, subject: Union[NounPhrase, ConjunctionPhrase]) -> Dict[str, Any]:
        """Execute subject-related operations."""
        # For now, subjects are mainly used for context
        return {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
    
    def _execute_tobe_sentence(self, sentence: SentencePhrase) -> Dict[str, Any]:
        """Execute 'to be' sentences (e.g., 'the cube is red')."""
        result = {
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': ['describe']
        }
        
        # Find the subject object and apply the adjective
        if sentence.subject:
            target_objects = self._find_objects_by_description(sentence.subject)
            
            for obj_id in target_objects:
                # Find the scene object
                scene_obj = None
                for obj in self.scene.objects:
                    if obj.object_id == obj_id:
                        scene_obj = obj
                        break
                
                if scene_obj and hasattr(sentence, 'vector'):
                    # Apply adjective properties from the sentence vector
                    self._apply_sentence_vector(scene_obj, sentence.vector)
                    result['objects_modified'].append(obj_id)
        
        return result
    
    def _apply_sentence_vector(self, scene_obj: SceneObject, sentence_vector: VectorSpace):
        """Apply sentence vector properties to a scene object."""
        # Apply color properties
        if 'red' in sentence_vector:
            scene_obj.vector_space['red'] = sentence_vector['red']
        if 'green' in sentence_vector:
            scene_obj.vector_space['green'] = sentence_vector['green']
        if 'blue' in sentence_vector:
            scene_obj.vector_space['blue'] = sentence_vector['blue']
        
        # Apply size properties
        if 'scaleX' in sentence_vector:
            scene_obj.vector_space['scaleX'] = sentence_vector['scaleX']
        if 'scaleY' in sentence_vector:
            scene_obj.vector_space['scaleY'] = sentence_vector['scaleY']
        if 'scaleZ' in sentence_vector:
            scene_obj.vector_space['scaleZ'] = sentence_vector['scaleZ']
    
    def _create_result(self, success: bool, message: str, sentence: str) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        return {
            'success': success,
            'message': message,
            'sentence': sentence,
            'sentence_parsed': None,
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
    
    def get_scene_summary(self) -> Dict[str, Any]:
        """Get a summary of the current scene state."""
        return {
            'total_objects': len(self.scene.objects),
            'object_types': list(set(obj.name for obj in self.scene.objects)),
            'object_ids': [obj.object_id for obj in self.scene.objects],
            'execution_history': len(self.execution_history),
            'last_action': self.execution_history[-1] if self.execution_history else None
        }
    
    def clear_scene(self):
        """Clear the current scene."""
        self.scene.objects = []
        self.scene.recent = []
        self.renderer.clear_scene()
        self.object_counter = 0
        self.execution_history = []
        print("✅ Scene cleared")
    
    def set_renderer(self, renderer):
        """Set a new renderer for the interpreter."""
        self.renderer = renderer
        print("✅ Renderer updated")
