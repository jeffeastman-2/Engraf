"""
Refactored Sentence Interpreter for ENGRAF

This module provides a high-level interface for interpreting English sentences
and converting them into 3D scene operations. It orchestrates specialized handlers
for different aspects of sentence interpretation.

Refactored for better maintainability and separation of concerns.
"""

from typing import Dict, Any, Union, Optional
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.temporal_scenes import TemporalScenes
# VPython renderer will be imported conditionally to avoid hanging

# Import specialized handlers
from .handlers import ObjectCreator, ObjectModifier, ObjectResolver, SceneManager, AssemblyCreator

# Import semantic agreement validation
from .semantic_validator import SemanticAgreementValidator


class SentenceInterpreter:
    """
    Main sentence interpreter that orchestrates natural language command processing
    using specialized handlers for different aspects of interpretation.
    """
    
    def __init__(self, renderer=None):
        """
        Initialize the sentence interpreter with specialized handlers.
        
        Args:
            renderer: The renderer to use for visualization (e.g., VPythonRenderer, MockRenderer)
        """
        if renderer is None:
            # Import VPython renderer only when needed
            from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer
            renderer = VPythonRenderer(headless=True)
        
        # Core components
        self.renderer = renderer
        self.temporal_scenes = TemporalScenes()
        self.scene = self.temporal_scenes.get_current_scene()  # For backward compatibility
        
        # State tracking using references for handlers
        self._object_counter = [0]  # Use list for mutable reference
        self._execution_history = []  # Use underscore for consistency
        self._last_acted_object = [None]  # Use list for mutable reference
        self._assembly_counter = [0]  # Assembly counter for unique IDs
        
        # Initialize specialized handlers
        self.object_resolver = ObjectResolver(self.scene, self._last_acted_object)
        self.object_creator = ObjectCreator(self.scene, self._object_counter)
        self.object_modifier = ObjectModifier(self.scene, self.renderer, self.object_resolver)
        self.assembly_creator = AssemblyCreator(self.scene, self._assembly_counter, self.object_resolver)
        self.scene_manager = SceneManager(
            self.scene, 
            self.renderer, 
            self._execution_history, 
            self._object_counter,
            self.object_resolver
        )
        
        # Initialize semantic agreement validator
        self.semantic_validator = SemanticAgreementValidator(self.scene)
        
    def interpret(self, sentence: str) -> Dict[str, Any]:
        """
        Interpret a natural language sentence and execute the corresponding actions.
        
        Args:
            sentence: The English sentence to interpret
            
        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Step 1: Check for temporal navigation commands
            sentence_lower = sentence.lower().strip()
            if "go back in time" in sentence_lower:
                return self.go_back_in_time()
            elif "go forward in time" in sentence_lower:
                return self.go_forward_in_time()
            
            # Step 2: Parse the sentence using LATN
            result = LATNLayerExecutor().execute_layer5(sentence)
            best_hypothesis = None
            if result.success and result.hypotheses:
                best_hypothesis = result.hypotheses[0]
            
            if best_hypothesis is None or len(best_hypothesis.tokens)!=1 :
                return self.scene_manager.create_result(False, "Failed to parse sentence", sentence)
            
            # Store the parsed sentence for access in transform methods
            self._current_sentence_parsed = best_hypothesis.tokens[0].phrase
            
            # Step 3: Validate semantic agreement with scene state
            is_valid, error_msg = self.semantic_validator.validate_command(self._current_sentence_parsed, sentence)
            if not is_valid:
                return self.scene_manager.create_result(False, error_msg, sentence)
            
            # Step 4: Execute the parsed sentence
            result = self._execute_sentence(self._current_sentence_parsed, sentence)
            
            # Step 5: Update the visual scene
            if result['success']:
                # Take a snapshot after successful operations that modify the scene
                if result.get('objects_created') or result.get('objects_modified'):
                    self.temporal_scenes.add_scene_snapshot(self.scene)
                self.renderer.render_scene(self.scene)
            
            return result
            
        except Exception as e:
            import traceback
            print(f"🚨 Exception caught in interpret:")
            print(f"🚨 Exception type: {type(e)}")
            print(f"🚨 Exception message: {str(e)}")
            print(f"🚨 Traceback:")
            traceback.print_exc()
            return self.scene_manager.create_result(False, f"Error interpreting sentence: {str(e)}", sentence)
    
    def _execute_sentence(self, parsed_sentence: SentencePhrase, original_sentence: str) -> Dict[str, Any]:
        """Execute a parsed sentence in the 3D scene using specialized handlers."""
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
                subject_result = self.scene_manager.execute_subject(parsed_sentence.subject)
                result.update(subject_result)
            
            # Handle tobe sentences (e.g., "the cube is red")
            if hasattr(parsed_sentence, 'tobe') and parsed_sentence.tobe:
                tobe_result = self.scene_manager.execute_tobe_sentence(parsed_sentence)
                print(f"🔍 DEBUG: tobe_result type: {type(tobe_result)}")
                print(f"🔍 DEBUG: tobe_result: {tobe_result}")
                result.update(tobe_result)
            
            self._execution_history.append(result)
            return result
            
        except Exception as e:
            return self.scene_manager.create_result(False, f"Error executing sentence: {str(e)}", original_sentence)
    
    def _execute_single_predicate(self, predicate: VerbPhrase, result) -> bool:
            phrase_result = self._execute_verb_phrase(predicate)
            result['objects_created'].extend(phrase_result.get('objects_created', []))
            result['objects_modified'].extend(phrase_result.get('objects_modified', []))
            result['actions_performed'].extend(phrase_result.get('actions_performed', []))
            success = phrase_result.get('success', True)
            result['success'] = result['success'] and success
            return success


    def _execute_predicate(self, predicate: Union[VerbPhrase, ConjunctionPhrase]) -> Dict[str, Any]:
        """Execute a predicate (verb phrase or conjunction of verb phrases)."""
        result = {
            'success': True,  # Default to success
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
        
        predicate.evaluate_boolean_function(lambda phrase: self._execute_single_predicate(phrase, result))

        return result
    
    def _execute_verb_phrase(self, vp: VerbPhrase) -> Dict[str, Any]:
        """Execute a single verb phrase using specialized handlers."""
        result = {
            'success': True,  # Default to success, handlers can override
            'objects_created': [],
            'objects_modified': [],
            'actions_performed': []
        }
        
        verb = vp.verb
        result['actions_performed'].append(verb)
        
        # Check if verb phrase has vector space information
        if hasattr(vp, 'vector') and vp.vector:
            # Check if this is a modification verb (with adjective complement like "bigger")
            # This takes priority over creation verbs for cases like "make it bigger"
            if (hasattr(vp, 'adjective_complement') and vp.adjective_complements and 
                vp.vector.isa('transform')):
                modified_objects = self._handle_modification_verb(vp)
                result['objects_modified'].extend(modified_objects)
            
            # Handle creation verbs using vector space
            elif vp.vector.isa('create'):
                created_objects = self._handle_creation_verb(vp)
                result['objects_created'].extend(created_objects)
            
            # Handle modification verbs using vector space (fallback for verbs without adjective complement)
            elif vp.vector.isa('transform'):
                modified_objects = self._handle_modification_verb(vp)
                result['objects_modified'].extend(modified_objects)
            
            # Handle organize verbs (grouping, assembly creation)
            elif vp.vector.isa('organize'):
                assembly_id = self._handle_organize_verb(vp)
                if assembly_id:
                    result['assemblies_created'] = result.get('assemblies_created', [])
                    result['assemblies_created'].append(assembly_id)
                    result['success'] = True
                else:
                    result['success'] = False
                    result['error'] = "Failed to create assembly"
            
            # Handle other vector space intents
            elif vp.vector.isa('edit') or vp.vector.isa('select'):
                print(f"⚠️  Unsupported verb intent for: {verb}")
            
            else:
                print(f"⚠️  No recognized intent vector for: {verb}")
        
        else:
            print(f"⚠️  No vector space information for verb: {verb}")
        
        return result
    
    def _handle_creation_verb(self, vp: VerbPhrase) -> list[str]:
        """Handle creation verbs using the ObjectCreator."""
        created_objects = []
        
        if vp.noun_phrase:
            objects = self.object_creator.extract_objects_from_np(vp.noun_phrase)
            
            for obj_info in objects:
                if vp.noun_phrase and vp.prepositions:
                    for prep in vp.prepositions:
                        obj_info['prepositional_phrases'] = []
                        if prep.vector.isa('spatial_proximity'): 
                            obj_info['prepositional_phrases'].append(prep)
                            break  
                        else:
                            continue

                obj_id = self.object_creator.create_scene_object(obj_info)
                if obj_id:
                    created_objects.append(obj_id)
                    # Update the most recently acted upon object
                    self._last_acted_object[0] = obj_id
        
        return created_objects
    
    def _handle_organize_verb(self, vp: VerbPhrase) -> Optional[str]:
        """Handle organize verbs like 'group' using the AssemblyCreator."""
        if vp.verb == 'group':
            return self.assembly_creator.create_assembly_from_verb_phrase(vp)
        else:
            print(f"⚠️  Unsupported organize verb: {vp.verb}")
            return None
    
    def _handle_modification_verb(self, vp: VerbPhrase) -> list[str]:
        """Handle modification verbs using the ObjectModifier."""
        print(f"🔧 _handle_modification_verb called with verb: {vp.verb}")
        print(f"🔧 vp.noun_phrase: {vp.noun_phrase}")
        if vp.noun_phrase and vp.prepositions:
            print(f"🔧 vp.prepositions: {vp.prepositions}")
        else:
            print(f"🔧 No prepositional phrases found")
        
        modified_objects = []
        
        # Find target objects using the ObjectResolver
        target_objects = self.object_resolver.resolve_target_objects(vp)
        
        for obj_id in target_objects:
            if self.object_modifier.modify_scene_object(obj_id, vp):
                modified_objects.append(obj_id)
                # Update the most recently acted upon object
                self._last_acted_object[0] = obj_id
        
        return modified_objects
    
    # Temporal navigation methods
    def go_back_in_time(self) -> Dict[str, Any]:
        """Go back to previous scene state."""
        if self.temporal_scenes.can_go_back():
            success = self.temporal_scenes.go_back()
            if success:
                self.scene = self.temporal_scenes.get_current_scene()
                # Update all handlers to use the new scene reference
                self._update_handlers_scene_reference()
                self.renderer.render_scene(self.scene)
                return {
                    'success': True,
                    'message': 'Traveled back in time',
                    'current_scene_index': self.temporal_scenes.get_current_index(),
                    'total_scenes': len(self.temporal_scenes)
                }
            else:
                return {'success': False, 'message': 'Failed to go back in time'}
        else:
            return {'success': False, 'message': 'Cannot go back any further'}
    
    def go_forward_in_time(self) -> Dict[str, Any]:
        """Go forward to next scene state."""
        if self.temporal_scenes.can_go_forward():
            success = self.temporal_scenes.go_forward()
            if success:
                self.scene = self.temporal_scenes.get_current_scene()
                # Update all handlers to use the new scene reference
                self._update_handlers_scene_reference()
                self.renderer.render_scene(self.scene)
                return {
                    'success': True,
                    'message': 'Traveled forward in time',
                    'current_scene_index': self.temporal_scenes.get_current_index(),
                    'total_scenes': len(self.temporal_scenes)
                }
            else:
                return {'success': False, 'message': 'Failed to go forward in time'}
        else:
            return {'success': False, 'message': 'Cannot go forward any further'}
    
    def get_temporal_status(self) -> Dict[str, Any]:
        """Get current temporal navigation status."""
        return {
            'current_scene_index': self.temporal_scenes.get_current_index(),
            'total_scenes': len(self.temporal_scenes),
            'can_go_back': self.temporal_scenes.can_go_back(),
            'can_go_forward': self.temporal_scenes.can_go_forward()
        }
    
    def _update_handlers_scene_reference(self):
        """Update all handlers to reference the current scene."""
        # Update the handlers to use the new scene reference
        self.object_resolver = ObjectResolver(self.scene, self._last_acted_object)
        self.object_creator = ObjectCreator(self.scene, self._object_counter)
        self.object_modifier = ObjectModifier(self.scene, self.renderer, self.object_resolver)
        self.scene_manager = SceneManager(
            self.scene, 
            self.renderer, 
            self._execution_history,
            self._object_counter,
            self.object_resolver
        )
        self.semantic_validator = SemanticAgreementValidator(self.scene)

    # Public interface methods delegating to SceneManager
    def get_scene_summary(self) -> Dict[str, Any]:
        """Get a summary of the current scene state."""
        return self.scene_manager.get_scene_summary()
    
    def clear_scene(self):
        """Clear the current scene."""
        self.scene_manager.clear_scene()
    
    def set_renderer(self, renderer):
        """Set a new renderer for the interpreter."""
        self.renderer = renderer
        self.object_modifier.renderer = renderer  # Update modifier's renderer reference
        self.scene_manager.renderer = renderer    # Update scene manager's renderer reference
        print("✅ Renderer updated")
    
    # Property accessors for backward compatibility
    @property 
    def object_counter(self):
        """Get the current object counter value."""
        return self._object_counter[0]
    
    @object_counter.setter
    def object_counter(self, value):
        """Set the object counter value."""
        self._object_counter[0] = value
    
    @property
    def last_acted_object(self):
        """Get the last acted upon object."""
        return self._last_acted_object[0]
    
    @last_acted_object.setter  
    def last_acted_object(self, value):
        """Set the last acted upon object."""
        self._last_acted_object[0] = value
