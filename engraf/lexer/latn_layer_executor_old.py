#!/usr/bin/env python3
"""
LATN Layer Executor

This module provides layered execution entry points for the LATN system.
Each layer automatically executes all prerequisite layers, allowing clean
entry at any layer for testing and processing.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from engraf.lexer.latn_tokenizer import latn_tokenize, TokenizationHypothesis
from engraf.lexer.latn_tokenizer_layer2 import latn_tokenize_layer2, NPTokenizationHypothesis
from engraf.lexer.latn_tokenizer_layer3 import latn_tokenize_layer3, PPTokenizationHypothesis
from engraf.lexer.latn_tokenizer_layer4 import latn_tokenize_layer4, VPTokenizationHypothesis
from engraf.lexer.semantic_grounding_layer2 import Layer2SemanticGrounder, Layer2GroundingResult
from engraf.lexer.semantic_grounding_layer3 import Layer3SemanticGrounder, Layer3GroundingResult
from engraf.lexer.semantic_grounding_layer4 import Layer4SemanticGrounder, Layer4GroundingResult
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.verb_phrase import VerbPhrase


@dataclass
class Layer1Result:
    """Result of Layer 1 execution (lexical tokenization)."""
    hypotheses: List[TokenizationHypothesis]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer2Result:
    """Result of Layer 2 execution (NP tokenization + grounding)."""
    layer1_result: Layer1Result
    hypotheses: List[NPTokenizationHypothesis]
    noun_phrases: List[NounPhrase]
    grounding_results: List[Layer2GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer3Result:
    """Result of Layer 3 execution (PP tokenization + grounding)."""
    layer2_result: Layer2Result
    hypotheses: List[PPTokenizationHypothesis]
    prepositional_phrases: List[PrepositionalPhrase]
    grounding_results: List[Layer3GroundingResult]
    success: bool
    confidence: float
    description: str = ""


@dataclass
class Layer4Result:
    """Result of Layer 4 execution (VP tokenization + execution)."""
    layer3_result: Layer3Result
    hypotheses: List[VPTokenizationHypothesis]
    verb_phrases: List[VerbPhrase]
    success: bool
    confidence: float
    description: str = ""


class LATNLayerExecutor:
    """Coordinates execution across all LATN layers with clean delegation to grounders."""
    
    def __init__(self, scene_model: Optional[SceneModel] = None):
        self.scene = scene_model
        self.layer2_grounder = Layer2SemanticGrounder(scene_model) if scene_model else None
        self.layer3_grounder = Layer3SemanticGrounder(scene_model) if scene_model else None
        self.layer4_grounder = Layer4SemanticGrounder(scene_model) if scene_model else None
    
    def execute_layer1(self, sentence: str) -> Layer1Result:
        """Execute Layer 1: Multi-hypothesis lexical tokenization.
        
        Args:
            sentence: Input sentence to tokenize
            
        Returns:
            Layer1Result with tokenization hypotheses
        """
        try:
            hypotheses = latn_tokenize(sentence)
            
            if not hypotheses or not sentence.strip():
                return Layer1Result(
                    hypotheses=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 1 tokenization failed for: '{sentence}'"
                )
            
            # Use confidence from best hypothesis
            best_confidence = hypotheses[0].confidence if hypotheses else 0.0
            
            return Layer1Result(
                hypotheses=hypotheses,
                success=True,
                confidence=best_confidence,
                description=f"Layer 1 tokenized '{sentence}' into {len(hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer1Result(
                hypotheses=[],
                success=False,
                confidence=0.0,
                description=f"Layer 1 failed: {e}"
            )
    
    def execute_layer2(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = False) -> Layer2Result:
        """Execute Layer 2: NP tokenization (requires Layer 1).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding
            return_all_matches: Whether to return all grounding matches
            
        Returns:
            Layer2Result with NP processing results
        """
        # First execute Layer 1
        layer1_result = self.execute_layer1(sentence)
        
        if not layer1_result.success:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed due to Layer 1 failure: {layer1_result.description}"
            )
        
        try:
            # Execute Layer 2 NP tokenization
            layer2_hypotheses = latn_tokenize_layer2(layer1_result.hypotheses)
            
            # Semantic grounding with hypothesis multiplication (if enabled)
            if enable_semantic_grounding and self.layer2_grounder:
                grounded_hypotheses, all_grounding_results = self.layer2_grounder.multiply_hypotheses_with_grounding(
                    layer2_hypotheses, return_all_matches
                )
            else:
                # No grounding - keep original hypotheses
                grounded_hypotheses = layer2_hypotheses
                all_grounding_results = []
            
            # Extract noun phrases from the (possibly grounded) hypotheses
            all_noun_phrases = self.layer2_grounder.extract_noun_phrases(grounded_hypotheses) if self.layer2_grounder else []
            
            # Calculate confidence based on best hypothesis
            layer2_confidence = grounded_hypotheses[0].confidence if grounded_hypotheses else layer1_result.confidence
            overall_confidence = (layer1_result.confidence + layer2_confidence) / 2
            
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=grounded_hypotheses,
                noun_phrases=all_noun_phrases,
                grounding_results=all_grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 2 processed {len(all_noun_phrases)} noun phrases in {len(grounded_hypotheses)} hypotheses"
            )
            
        except Exception as e:
            return Layer2Result(
                layer1_result=layer1_result,
                hypotheses=[],
                noun_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 2 failed: {e}"
            )
    
    def execute_layer3(self, sentence: str, enable_semantic_grounding: bool = True,
                      return_all_matches: bool = False) -> Layer3Result:
        """Execute Layer 3: PP tokenization (requires Layer 1 + 2).
        
        Args:
            sentence: Input sentence to process
            enable_semantic_grounding: Whether to perform semantic grounding
            return_all_matches: Whether to return all grounding matches
            
        Returns:
            Layer3Result with complete LATN processing results
        """
        # First execute Layer 2 (which includes Layer 1)
        layer2_result = self.execute_layer2(sentence, enable_semantic_grounding, return_all_matches)
        
        if not layer2_result.success:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed due to Layer 2 failure: {layer2_result.description}"
            )
        
        try:
            # Execute Layer 3 PP tokenization
            layer3_hypotheses = latn_tokenize_layer3(layer2_result.hypotheses)
            
            # Extract PrepositionalPhrase objects
            prepositional_phrases = self.layer3_grounder.extract_prepositional_phrases(layer3_hypotheses) if self.layer3_grounder else []
            
            # Two-pass PP attachment resolution
            if enable_semantic_grounding and self.layer3_grounder:
                # Use Layer 3 grounder for 2-pass PP attachment processing
                layer3_hypotheses = self.layer3_grounder.process_pp_attachments(layer3_hypotheses, return_all_matches)
                
                # Ground the validated prepositional phrases
                grounding_results = self.layer3_grounder.ground_multiple(
                    prepositional_phrases, return_all_matches=return_all_matches
                )
            else:
                # No grounding - use original hypotheses
                grounding_results = []
            
            # Calculate confidence
            layer3_confidence = layer3_hypotheses[0].confidence if layer3_hypotheses else layer2_result.confidence
            overall_confidence = (layer2_result.confidence + layer3_confidence) / 2
            
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=layer3_hypotheses,
                prepositional_phrases=prepositional_phrases,
                grounding_results=grounding_results,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 3 processed {len(prepositional_phrases)} prepositional phrases"
            )
            
        except Exception as e:
            return Layer3Result(
                layer2_result=layer2_result,
                hypotheses=[],
                prepositional_phrases=[],
                grounding_results=[],
                success=False,
                confidence=0.0,
                description=f"Layer 3 failed: {e}"
            )
    
    def execute_layer4(self, sentence: str, enable_action_execution: bool = True) -> Layer4Result:
        """Execute Layer 4: VP tokenization and action execution (requires Layer 1-3).
        
        Args:
            sentence: Input sentence to process
            enable_action_execution: Whether to execute VP actions (create objects, etc.)
            
        Returns:
            Layer4Result with complete LATN processing results
        """
        try:
            # Execute Layer 3 first (which includes Layer 1 and 2)
            layer3_result = self.execute_layer3(sentence, enable_semantic_grounding=True)
            
            if not layer3_result.success:
                return Layer4Result(
                    layer3_result=layer3_result,
                    hypotheses=[],
                    verb_phrases=[],
                    success=False,
                    confidence=0.0,
                    description=f"Layer 4 failed due to Layer 3 failure: {layer3_result.description}"
                )
            
            # Execute Layer 4 VP tokenization
            layer4_hypotheses = latn_tokenize_layer4(layer3_result.hypotheses)
            
            # Extract verb phrases
            verb_phrases = self.layer4_grounder.extract_verb_phrases(layer4_hypotheses) if self.layer4_grounder else []
            
            # Execute actions if enabled
            if enable_action_execution and verb_phrases and self.layer4_grounder:
                action_results = self.layer4_grounder.execute_verb_phrase_actions(verb_phrases)
            
            layer4_confidence = layer4_hypotheses[0].confidence if layer4_hypotheses else layer3_result.confidence
            overall_confidence = (layer3_result.confidence + layer4_confidence) / 2
            
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=layer4_hypotheses,
                verb_phrases=verb_phrases,
                success=True,
                confidence=overall_confidence,
                description=f"Layer 4 processed {len(verb_phrases)} verb phrases"
            )
            
        except Exception as e:
            return Layer4Result(
                layer3_result=layer3_result,
                hypotheses=[],
                verb_phrases=[],
                success=False,
                confidence=0.0,
                description=f"Layer 4 failed: {e}"
            )


# Convenience functions for direct layer execution
def execute_layer1(sentence: str) -> Layer1Result:
    """Convenience function to execute Layer 1 only."""
    executor = LATNLayerExecutor()
    return executor.execute_layer1(sentence)


def execute_layer2(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer2Result:
    """Convenience function to execute Layer 2 (includes Layer 1)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer2(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer3(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer3Result:
    """Convenience function to execute Layer 3 (includes Layers 1 & 2)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer3(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer4(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_action_execution: bool = True) -> Layer4Result:
    """Convenience function to execute Layer 4 (includes Layers 1-3)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer4(sentence, enable_action_execution)
# Convenience functions for direct layer execution
def execute_layer1(sentence: str) -> Layer1Result:
            )
    
    def _extract_verb_phrases(self, layer4_hypotheses: List[VPTokenizationHypothesis]) -> List[VerbPhrase]:
        """Extract VerbPhrase objects from Layer 4 processing."""
        verb_phrases = []
        
        for hypothesis in layer4_hypotheses:
            # Extract from VP replacements (this is the authoritative source)
            if hasattr(hypothesis, 'vp_replacements'):
                for start_idx, end_idx, vp_token in hypothesis.vp_replacements:
                    if hasattr(vp_token, '_original_vp') and isinstance(vp_token._original_vp, VerbPhrase):
                        verb_phrases.append(vp_token._original_vp)
        
        return verb_phrases
    
    def _execute_verb_phrase_actions(self, verb_phrases: List[VerbPhrase]):
        """Execute actions specified by verb phrases."""
        for vp in verb_phrases:
            if not vp.verb:
                continue
                
            verb_word = vp.verb.word if hasattr(vp.verb, 'word') else str(vp.verb)
            
            if verb_word in ["create", "make", "build"] and vp.noun_phrase:
                self._execute_create_action(vp)
            elif verb_word in ["move", "translate"] and vp.noun_phrase:
                self._execute_move_action(vp)
            elif verb_word in ["rotate", "turn"] and vp.noun_phrase:
                self._execute_rotate_action(vp)
            elif verb_word in ["delete", "remove", "destroy"] and vp.noun_phrase:
                self._execute_delete_action(vp)
    
    def _execute_create_action(self, vp: VerbPhrase):
        """Execute a create action from a verb phrase."""
        if not self.scene_model or not vp.noun_phrase:
            return
            
        # Extract object properties from the noun phrase
        # Try vector property first, then to_vector() method
        np_vector = None
        if hasattr(vp.noun_phrase, 'vector') and vp.noun_phrase.vector:
            np_vector = vp.noun_phrase.vector
        elif hasattr(vp.noun_phrase, 'to_vector'):
            try:
                np_vector = vp.noun_phrase.to_vector()
            except AttributeError:
                # Fallback if to_vector() fails due to missing noun
                np_vector = vp.noun_phrase.vector if hasattr(vp.noun_phrase, 'vector') else None
        
        if not np_vector:
            return
            
        # Determine shape from the word content
        shape = "box"  # default
        if hasattr(np_vector, 'word') and np_vector.word:
            word_content = np_vector.word.lower()
            if "sphere" in word_content:
                shape = "sphere"
            elif "cube" in word_content:
                shape = "cube" 
            elif "cylinder" in word_content:
                shape = "cylinder"
            elif "box" in word_content:
                shape = "box"
            
        # Create the object
        from engraf.visualizer.scene.scene_object import SceneObject
        import uuid
        
        obj_id = f"{shape}_{str(uuid.uuid4())[:8]}"
        
        # Create a copy of the vector for the scene object
        scene_vector = np_vector.copy() if hasattr(np_vector, 'copy') else np_vector
        
        scene_obj = SceneObject(
            name=shape,
            vector=scene_vector,
            object_id=obj_id
        )
        
        self.scene_model.add_object(scene_obj)
        print(f"✅ Created {shape} with ID {obj_id}")
    
    def _execute_move_action(self, vp: VerbPhrase):
        """Execute a move action from a verb phrase."""
        # Placeholder for move action
        print(f"🚀 Move action requested for {vp.noun_phrase}")
    
    def _execute_rotate_action(self, vp: VerbPhrase):
        """Execute a rotate action from a verb phrase."""
        # Placeholder for rotate action
        print(f"🔄 Rotate action requested for {vp.noun_phrase}")
    
    def _execute_delete_action(self, vp: VerbPhrase):
        """Execute a delete action from a verb phrase."""
        # Placeholder for delete action
        print(f"🗑️ Delete action requested for {vp.noun_phrase}")
    
    def _extract_noun_phrases(self, layer2_hypotheses: List[NPTokenizationHypothesis]) -> List[NounPhrase]:
        """Extract NounPhrase objects from Layer 2 processing.
        
        The token stream is the single source of truth - NP tokens are already
        properly placed in the tokens list by replace_np_sequences().
        
        Clean semantics:
        - If grounding was enabled: Return SceneObjectPhrase for grounded NPs + NounPhrase for unbound NPs
        - If grounding was disabled: Return only NounPhrase objects (all NPs)
        """
        noun_phrases = []
        
        for i, hypothesis in enumerate(layer2_hypotheses):
            # Look for NP tokens in the hypothesis token stream
            for j, token in enumerate(hypothesis.tokens):
                if hasattr(token, '_original_np') and isinstance(token._original_np, NounPhrase):
                    # Check if this token has been grounded
                    if hasattr(token, '_grounded_phrase'):
                        # Use the grounded SceneObjectPhrase
                        noun_phrases.append(token._grounded_phrase)
                    else:
                        # Use the original NounPhrase (unbound)
                        noun_phrases.append(token._original_np)
        
        return noun_phrases
    
    def _multiply_hypotheses_with_grounding(self, layer2_hypotheses, return_all_matches):
        """Multiply hypotheses based on grounding results using two-pass algorithm.
        
        Pass 1: Collect all scene object matches for each NP in each hypothesis
        Pass 2: Generate combinatorial hypotheses with one object per NP
        
        Returns:
            tuple: (grounded_hypotheses, all_grounding_results)
        """
        import copy
        from itertools import product
        
        all_grounded_hypotheses = []
        all_grounding_results = []
        
        for base_hypothesis in layer2_hypotheses:
            # Pass 1: Collect all matches for each NP in this hypothesis
            np_matches = []  # List of (np_object, [matching_scene_objects])
            
            for token in base_hypothesis.tokens:
                if hasattr(token, '_original_np') and isinstance(token._original_np, NounPhrase):
                    np_obj = token._original_np
                    
                    # Ground this NP to get all possible matches
                    grounding_result = self.layer2_grounder.ground(np_obj, return_all_matches=True)
                    
                    if grounding_result.success:
                        # Collect best match + alternatives
                        matches = [grounding_result.resolved_object]
                        if grounding_result.alternative_matches:
                            matches.extend([obj for confidence, obj in grounding_result.alternative_matches])
                        np_matches.append((np_obj, matches))
                        all_grounding_results.append(grounding_result)
                    else:
                        # No matches - this NP grounds to nothing
                        np_matches.append((np_obj, []))
                        all_grounding_results.append(grounding_result)
            
            # Pass 2: Generate combinatorial hypotheses
            if np_matches:
                # Handle mixed success/failure: some NPs ground, others don't
                grounded_matches = [(np_obj, matches) for np_obj, matches in np_matches if matches]
                ungrounded_nps = [np_obj for np_obj, matches in np_matches if not matches]
                
                if grounded_matches:
                    # Extract match lists for successfully grounded NPs
                    match_lists = [matches for np_obj, matches in grounded_matches]
                    
                    # Generate all combinations using Cartesian product
                    for combination in product(*match_lists):
                        # Create new hypothesis with this specific grounding combination
                        new_hypothesis = copy.deepcopy(base_hypothesis)
                        
                        # Update each NP in the hypothesis with its specific grounding
                        grounded_index = 0
                        for token_idx, token in enumerate(new_hypothesis.tokens):
                            if hasattr(token, '_original_np') and isinstance(token._original_np, NounPhrase):
                                np_obj = token._original_np
                                
                                # Check if this NP was successfully grounded
                                grounded_np_objs = [grounded_np for grounded_np, matches in grounded_matches]
                                
                                # Compare NPs by content, not object identity
                                np_grounded = False
                                matching_grounded_index = -1
                                for idx, grounded_np in enumerate(grounded_np_objs):
                                    if (np_obj.noun == grounded_np.noun and 
                                        np_obj.determiner == grounded_np.determiner):
                                        np_grounded = True
                                        matching_grounded_index = idx
                                        break
                                
                                if np_grounded:
                                    # This NP was successfully grounded
                                    specific_object = combination[grounded_index]
                                    # Create a SceneObjectPhrase for this grounding
                                    from engraf.pos.scene_object_phrase import SceneObjectPhrase
                                    scene_object_phrase = SceneObjectPhrase.from_noun_phrase(np_obj)
                                    scene_object_phrase.resolve_to_scene_object(specific_object)
                                    # Store the grounded version in the token
                                    token._grounded_phrase = scene_object_phrase
                                    grounded_index += 1
                        
                        # Update hypothesis description
                        grounded_object_ids = [obj.object_id for obj in combination]
                        ungrounded_count = len(ungrounded_nps)
                        new_hypothesis.description = f"{base_hypothesis.description} → grounded {len(grounded_object_ids)} to {grounded_object_ids}, {ungrounded_count} ungrounded"
                        
                        all_grounded_hypotheses.append(new_hypothesis)
                else:
                    # No NPs could be grounded - keep original hypothesis
                    ungrounded_hypothesis = copy.deepcopy(base_hypothesis)
                    ungrounded_hypothesis.description = f"{base_hypothesis.description} → no valid grounding"
                    all_grounded_hypotheses.append(ungrounded_hypothesis)
        
        # Sort hypotheses by confidence
        all_grounded_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return all_grounded_hypotheses, all_grounding_results
    
    def _extract_prepositional_phrases(self, layer3_hypotheses: List[PPTokenizationHypothesis]) -> List[PrepositionalPhrase]:
        """Extract PrepositionalPhrase objects from Layer 3 processing."""
        prepositional_phrases = []
        
        for hypothesis in layer3_hypotheses:
            # Look for PP tokens in the hypothesis
            for token in hypothesis.tokens:
                if hasattr(token, '_original_pp') and isinstance(token._original_pp, PrepositionalPhrase):
                    prepositional_phrases.append(token._original_pp)
            
            # Also check PP replacements
            if hasattr(hypothesis, 'pp_replacements'):
                for start_idx, end_idx, pp_token in hypothesis.pp_replacements:
                    if hasattr(pp_token, '_original_pp') and isinstance(pp_token._original_pp, PrepositionalPhrase):
                        prepositional_phrases.append(pp_token._original_pp)
        
        return prepositional_phrases


def execute_layer1(sentence: str) -> Layer1Result:
    
    def _validate_spatial_attachments(self, attachment_hypotheses: List[PPTokenizationHypothesis]) -> List[PPTokenizationHypothesis]:
        """Pass 2: Validate PP attachments using spatial reasoning.
        
        Applies prep-specific spatial tests to filter out impossible combinations.
        
        Args:
            attachment_hypotheses: Hypotheses with PP attachment combinations
            
        Returns:
            List of spatially validated hypotheses
        """
        validated = []
        
        for hypothesis in attachment_hypotheses:
            is_valid = True
            validation_scores = []
            
            # Check each PP attachment for spatial validity
            for token in hypothesis.tokens:
                if (hasattr(token, '_attachment_info') and 
                    hasattr(token, 'word') and token.word and token.word.startswith('PP(')):
                    
                    # Extract preposition and noun phrase from PP token
                    pp_content = token.word[3:-1]  # Remove 'PP(' and ')'
                    prep, np_part = pp_content.split(' ', 1) if ' ' in pp_content else (pp_content, '')
                    
                    # Get attachment target
                    target_idx = token._attachment_info.get('attaches_to')
                    
                    # Apply prep-specific spatial validation
                    spatial_score = self._validate_prep_spatial_relationship(prep, np_part, target_idx, hypothesis)
                    validation_scores.append(spatial_score)
                    
                    # Filter out impossible relationships (score < 0.1)
                    if spatial_score < 0.1:
                        is_valid = False
                        break
            
            if is_valid and validation_scores:
                # Update hypothesis confidence based on spatial validation
                avg_spatial_score = sum(validation_scores) / len(validation_scores)
                hypothesis.confidence = hypothesis.confidence * avg_spatial_score
                validated.append(hypothesis)
        
        # Sort by confidence (best first)
        validated.sort(key=lambda h: h.confidence, reverse=True)
        return validated
    
    def _validate_prep_spatial_relationship(self, prep: str, np_part: str, target_idx: Optional[int], 
                                          hypothesis: PPTokenizationHypothesis) -> float:
        """Validate a specific prepositional relationship using spatial reasoning.
        
        Args:
            prep: Preposition ('on', 'under', 'beside', etc.)
            np_part: Noun phrase part of the PP
            target_idx: Index of attachment target in token stream
            hypothesis: Current hypothesis for context
            
        Returns:
            Spatial validity score (0.0 = impossible, 1.0 = perfect)
        """
        if not self.scene or target_idx is None:
            return 0.5  # Neutral score if no scene or no attachment
        
        # Get scene objects involved in the relationship
        try:
            # Extract object names from the PP and target
            pp_object_name = self._extract_object_name_from_np(np_part)
            target_token = hypothesis.tokens[target_idx]
            target_object_name = self._extract_object_name_from_token(target_token)
            
            if not pp_object_name or not target_object_name:
                return 0.5  # Can't determine objects
            
            # Find actual scene objects
            pp_object = self._find_scene_object(pp_object_name)
            target_object = self._find_scene_object(target_object_name)
            
            if not pp_object or not target_object:
                return 0.3  # Objects not found in scene
            
            # Apply prep-specific spatial tests
            return self._apply_prep_spatial_test(prep, target_object, pp_object)
            
        except Exception:
            return 0.2  # Error in validation
    
    def _apply_prep_spatial_test(self, prep: str, obj1, obj2) -> float:
        """Apply prep-specific spatial relationship test.
        
        Args:
            prep: Preposition to test
            obj1: First object (attachment target)
            obj2: Second object (PP object)
            
        Returns:
            Spatial validity score
        """
        if not hasattr(obj1, 'position') or not hasattr(obj2, 'position'):
            return 0.5
        
        pos1, pos2 = obj1.position, obj2.position
        dx, dy, dz = pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        # Prep-specific spatial tests
        if prep == 'on':
            # 'on' requires vertical alignment and obj2 above obj1
            if distance < 0.1:  # Co-located objects
                return 0.1  # Can't be "on" if at same position
            return 1.0 if dy > 0.5 and abs(dx) < 0.5 and abs(dz) < 0.5 else 0.2
            
        elif prep == 'under':
            # 'under' requires obj2 below obj1
            if distance < 0.1:
                return 0.1
            return 1.0 if dy < -0.5 and abs(dx) < 0.5 and abs(dz) < 0.5 else 0.2
            
        elif prep == 'beside':
            # 'beside' requires lateral proximity
            if distance < 0.1:
                return 0.1
            return 1.0 if abs(dy) < 0.5 and (abs(dx) > 0.5 or abs(dz) > 0.5) else 0.3
            
        elif prep == 'to':
            # 'to' is a movement target, always valid for spatial purposes
            return 0.9
            
        else:
            # Unknown preposition
            return 0.5
    
    def _extract_object_name_from_np(self, np_part: str) -> Optional[str]:
        """Extract object name from noun phrase text."""
        # Simple extraction - look for known object words
        words = np_part.lower().split()
        object_words = ['box', 'table', 'pyramid', 'sphere', 'cube', 'ball']
        for word in words:
            if word in object_words:
                return word
        return None
    
    def _extract_object_name_from_token(self, token) -> Optional[str]:
        """Extract object name from a token."""
        if hasattr(token, 'word') and token.word:
            if token.word.startswith('NP(') or token.word.startswith('PP('):
                content = token.word[3:-1]  # Remove prefix and parentheses
                return self._extract_object_name_from_np(content)
        return None
    
    def _find_scene_object(self, object_name: str):
        """Find scene object by name."""
        if not self.scene:
            return None
        for obj in self.scene.objects:
            if hasattr(obj, 'word') and obj.word == object_name:
                return obj
            if hasattr(obj, 'object_id') and object_name in obj.object_id:
                return obj
        return None
    
    def update_scene_model(self, scene_model: SceneModel):
        """Update the scene model used for semantic grounding."""
        self.scene_model = scene_model
        if scene_model:
            self.layer2_grounder = Layer2SemanticGrounder(scene_model)
            self.layer3_grounder = Layer3SemanticGrounder(scene_model)
        else:
            self.layer2_grounder = None
            self.layer3_grounder = None
    
    def get_layer_analysis(self, sentence: str, target_layer: int = 3) -> Dict[str, Any]:
        """Get detailed analysis for debugging/inspection up to target layer.
        
        Args:
            sentence: Input sentence
            target_layer: Execute up to this layer (1, 2, or 3)
            
        Returns:
            Detailed analysis dictionary
        """
        analysis = {
            'input': sentence,
            'target_layer': target_layer
        }
        
        if target_layer >= 1:
            layer1_result = self.execute_layer1(sentence)
            analysis['layer1'] = {
                'success': layer1_result.success,
                'confidence': layer1_result.confidence,
                'description': layer1_result.description,
                'hypothesis_count': len(layer1_result.hypotheses),
                'hypotheses': [
                    {
                        'tokens': [tok.word for tok in hyp.tokens],
                        'confidence': hyp.confidence,
                        'description': hyp.description
                    }
                    for hyp in layer1_result.hypotheses[:5]  # Limit to first 5 for readability
                ]
            }
        
        if target_layer >= 2:
            layer2_result = self.execute_layer2(sentence, enable_semantic_grounding=True, return_all_matches=True)
            analysis['layer2'] = {
                'success': layer2_result.success,
                'confidence': layer2_result.confidence,
                'description': layer2_result.description,
                'hypothesis_count': len(layer2_result.hypotheses),
                'noun_phrase_count': len(layer2_result.noun_phrases),
                'noun_phrases': [
                    {
                        'text': str(np),
                        'noun': getattr(np, 'noun', None),
                        'determiner': getattr(np, 'determiner', None),
                        'consumed_tokens': [tok.word if hasattr(tok, 'word') else str(tok) for tok in getattr(np, 'consumed_tokens', [])],
                        'resolved': getattr(np, 'is_resolved', lambda: False)()
                    }
                    for np in layer2_result.noun_phrases
                ],
                'grounding_results': [
                    {
                        'success': gr.success,
                        'confidence': gr.confidence,
                        'description': gr.description,
                        'resolved_object_id': getattr(gr.resolved_object, 'object_id', None) if gr.resolved_object else None,
                        'alternative_count': len(gr.alternative_matches)
                    }
                    for gr in layer2_result.grounding_results
                ]
            }
        
        if target_layer >= 3:
            layer3_result = self.execute_layer3(sentence, enable_semantic_grounding=True, return_all_matches=True)
            analysis['layer3'] = {
                'success': layer3_result.success,
                'confidence': layer3_result.confidence,
                'description': layer3_result.description,
                'hypothesis_count': len(layer3_result.hypotheses),
                'pp_count': len(layer3_result.prepositional_phrases),
                'prepositional_phrases': [
                    {
                        'text': str(pp),
                        'preposition': getattr(pp, 'preposition', None),
                        'vector_text': getattr(pp, 'vector_text', None),
                        'has_np': hasattr(pp, 'np') and pp.np is not None
                    }
                    for pp in layer3_result.prepositional_phrases
                ],
                'grounding_results': [
                    {
                        'success': gr.success,
                        'confidence': gr.confidence,
                        'description': gr.description,
                        'resolved_type': type(gr.resolved_object).__name__ if gr.resolved_object else None,
                        'alternative_count': len(gr.alternative_matches)
                    }
                    for gr in layer3_result.grounding_results
                ]
            }
        
        return analysis


# Convenience functions for direct layer access
def execute_layer1(sentence: str, scene_model: Optional[SceneModel] = None) -> Layer1Result:
    """Convenience function to execute only Layer 1."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer1(sentence)


def execute_layer2(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer2Result:
    """Convenience function to execute Layer 2 (includes Layer 1)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer2(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer3(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_semantic_grounding: bool = True, return_all_matches: bool = False) -> Layer3Result:
    """Convenience function to execute Layer 3 (includes Layers 1 & 2)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer3(sentence, enable_semantic_grounding, return_all_matches)


def execute_layer4(sentence: str, scene_model: Optional[SceneModel] = None,
                  enable_action_execution: bool = True) -> Layer4Result:
    """Convenience function to execute Layer 4 (includes Layers 1-3)."""
    executor = LATNLayerExecutor(scene_model)
    return executor.execute_layer4(sentence, enable_action_execution)
