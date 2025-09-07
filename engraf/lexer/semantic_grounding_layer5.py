#!/usr/bin/env python3
"""
LATN Layer 5 Semantic Grounding

This module provides semantic grounding and execution capabilities for LATN Layer 5 
Sentence tokens. It takes well-formed sentences and executes them in the scene.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.hypothesis import TokenizationHypothesis
from engraf.lexer.vector_space import VectorSpace


@dataclass
class Layer5GroundingResult:
    """Result of Layer 5 semantic grounding operation."""
    success: bool
    confidence: float
    executed_actions: List[str]
    scene_changes: List[str]
    description: str = ""


class Layer5SemanticGrounder:
    """Semantic grounding and execution for LATN Layer 5 Sentence tokens.
    
    Layer 5 performs sentence execution - taking well-formed sentence structures
    and executing the commands they represent in the scene.
    """
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
    
    def extract_sentence_phrases(self, layer5_hypotheses: List[TokenizationHypothesis]) -> List[SentencePhrase]:
        """Extract SentencePhrase objects from Layer 5 hypotheses.
        
        Args:
            layer5_hypotheses: List of Layer 5 tokenization hypotheses
            
        Returns:
            List of SentencePhrase objects found in the hypotheses
        """
        sentence_phrases = []
        
        for hypothesis in layer5_hypotheses:
            for token in hypothesis.tokens:
                if hasattr(token, '_sentence_phrase') and isinstance(token._sentence_phrase, SentencePhrase):
                    sentence_phrases.append(token._sentence_phrase)
        
        return sentence_phrases
    
    def execute_sentences(self, sentence_phrases: List[SentencePhrase]) -> List[Layer5GroundingResult]:
        """Execute sentence commands in the scene.
        
        Args:
            sentence_phrases: List of SentencePhrase objects to execute
            
        Returns:
            List of execution results
        """
        results = []
        
        for sentence in sentence_phrases:
            try:
                result = self._execute_sentence(sentence)
                results.append(result)
                
            except Exception as e:
                results.append(Layer5GroundingResult(
                    success=False,
                    confidence=0.0,
                    executed_actions=[],
                    scene_changes=[],
                    description=f"Sentence execution failed: {e}"
                ))
        
        return results
    
    def _execute_sentence(self, sentence: SentencePhrase) -> Layer5GroundingResult:
        """Execute a single sentence command.
        
        Args:
            sentence: SentencePhrase to execute
            
        Returns:
            Layer5GroundingResult with execution details
        """
        executed_actions = []
        scene_changes = []
        
        # Imperative sentences (VP only): "move it", "create cube"
        if sentence.predicate and not sentence.subject:
            return self._execute_imperative(sentence.predicate, sentence.prepositional_phrases, executed_actions, scene_changes)
        
        # Declarative sentences (NP + VP): "the cube moves"
        elif sentence.subject and sentence.predicate:
            return self._execute_declarative(sentence.subject, sentence.predicate, sentence.prepositional_phrases, executed_actions, scene_changes)
        
        # Standalone noun phrases: "the red cube" (for selection/identification)
        elif sentence.subject and not sentence.predicate:
            return self._execute_identification(sentence.subject, executed_actions, scene_changes)
        
        else:
            return Layer5GroundingResult(
                success=False,
                confidence=0.0,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description="Unrecognized sentence structure"
            )
    
    def _execute_imperative(self, predicate, prepositional_phrases: List, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute imperative sentence: VP + optional PPs.
        
        Examples: "move it to [2,3,4]", "create red cube", "rotate it by 90 degrees"
        """
        if not hasattr(predicate, '_original_vp') or not isinstance(predicate._original_vp, VerbPhrase):
            return Layer5GroundingResult(
                success=False,
                confidence=0.0,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description="Predicate is not a verb phrase"
            )
        
        vp = predicate._original_vp
        verb_word = vp.verb.word if hasattr(vp.verb, 'word') else str(vp.verb)
        
        # Extract target objects from VP noun phrase
        target_objects = self._extract_objects_from_np(vp.noun_phrase)
        
        if verb_word in ['move', 'translate']:
            return self._execute_move_command(target_objects, prepositional_phrases, executed_actions, scene_changes)
        elif verb_word in ['create', 'make', 'draw']:
            return self._execute_create_command(vp.noun_phrase, prepositional_phrases, executed_actions, scene_changes)
        elif verb_word in ['rotate', 'turn']:
            return self._execute_rotate_command(target_objects, prepositional_phrases, executed_actions, scene_changes)
        elif verb_word in ['color', 'paint']:
            return self._execute_color_command(target_objects, vp.noun_phrase, executed_actions, scene_changes)
        elif verb_word in ['scale', 'resize']:
            return self._execute_scale_command(target_objects, vp.noun_phrase, executed_actions, scene_changes)
        elif verb_word in ['group', 'organize']:
            return self._execute_group_command(target_objects, executed_actions, scene_changes)
        else:
            executed_actions.append(f"Analyzed verb: {verb_word}")
            return Layer5GroundingResult(
                success=True,
                confidence=0.6,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description=f"Recognized but did not execute action: {verb_word}"
            )
    
    def _execute_declarative(self, subject, predicate, prepositional_phrases: List, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute declarative sentence: NP + VP + optional PPs."""
        executed_actions.append("Processed declarative sentence")
        return Layer5GroundingResult(
            success=True,
            confidence=0.7,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description="Declarative sentence processed (not implemented for execution)"
        )
    
    def _execute_identification(self, subject, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute identification sentence: NP only."""
        objects = self._extract_objects_from_np(subject)
        executed_actions.append(f"Identified {len(objects)} object(s)")
        return Layer5GroundingResult(
            success=True,
            confidence=0.8,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description=f"Identified {len(objects)} scene object(s)"
        )
    
    def _extract_objects_from_np(self, np_token) -> List[SceneObject]:
        """Extract scene objects from an NP token."""
        if not np_token:
            return []
        
        # Check if this is a grounded NP with scene objects
        if hasattr(np_token, '_grounding_info') and np_token._grounding_info:
            grounding_info = np_token._grounding_info
            if hasattr(grounding_info, 'resolved_objects'):
                return grounding_info.resolved_objects
            elif hasattr(grounding_info, 'scene_objects'):
                return grounding_info.scene_objects
        
        # Check for single scene object
        if hasattr(np_token, 'scene_object') and np_token.scene_object:
            return [np_token.scene_object]
        
        return []
    
    def _execute_move_command(self, objects: List[SceneObject], pps: List, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute move command."""
        if not objects:
            return Layer5GroundingResult(
                success=False,
                confidence=0.0,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description="No objects to move"
            )
        
        # Extract target location from prepositional phrases
        target_location = None
        for pp in pps:
            if hasattr(pp, 'word') and ('to' in pp.word or 'at' in pp.word):
                # Extract coordinates from PP
                if '[' in pp.word and ']' in pp.word:
                    coord_str = pp.word[pp.word.find('['):pp.word.find(']')+1]
                    try:
                        # Parse coordinates like "[2, 3, 4]"
                        coords = eval(coord_str)  # Safe here since we control the input
                        if isinstance(coords, list) and len(coords) >= 3:
                            target_location = coords[:3]
                            break
                    except:
                        continue
        
        if target_location:
            for obj in objects:
                old_pos = obj.position.copy()
                obj.position = target_location
                executed_actions.append(f"Moved {obj.name}")
                scene_changes.append(f"Object {obj.name} moved from {old_pos} to {target_location}")
            
            return Layer5GroundingResult(
                success=True,
                confidence=0.9,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description=f"Successfully moved {len(objects)} object(s) to {target_location}"
            )
        else:
            executed_actions.append("Attempted move without target location")
            return Layer5GroundingResult(
                success=False,
                confidence=0.3,
                executed_actions=executed_actions,
                scene_changes=scene_changes,
                description="No target location found for move command"
            )
    
    def _execute_create_command(self, np_token, pps: List, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute create command."""
        executed_actions.append("Analyzed creation command")
        scene_changes.append("Creation command recognized (execution not implemented)")
        return Layer5GroundingResult(
            success=True,
            confidence=0.6,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description="Create command analyzed (actual creation not implemented)"
        )
    
    def _execute_rotate_command(self, objects: List[SceneObject], pps: List, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute rotate command."""
        executed_actions.append(f"Analyzed rotation for {len(objects)} object(s)")
        return Layer5GroundingResult(
            success=True,
            confidence=0.7,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description="Rotate command analyzed (rotation execution not implemented)"
        )
    
    def _execute_color_command(self, objects: List[SceneObject], color_np, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute color change command."""
        executed_actions.append(f"Analyzed color change for {len(objects)} object(s)")
        return Layer5GroundingResult(
            success=True,
            confidence=0.7,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description="Color command analyzed (color change execution not implemented)"
        )
    
    def _execute_scale_command(self, objects: List[SceneObject], scale_np, executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute scale command."""
        executed_actions.append(f"Analyzed scaling for {len(objects)} object(s)")
        return Layer5GroundingResult(
            success=True,
            confidence=0.7,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description="Scale command analyzed (scaling execution not implemented)"
        )
    
    def _execute_group_command(self, objects: List[SceneObject], executed_actions: List[str], scene_changes: List[str]) -> Layer5GroundingResult:
        """Execute group command."""
        executed_actions.append(f"Grouped {len(objects)} object(s)")
        scene_changes.append(f"Logical grouping of {len(objects)} objects")
        return Layer5GroundingResult(
            success=True,
            confidence=0.8,
            executed_actions=executed_actions,
            scene_changes=scene_changes,
            description=f"Successfully grouped {len(objects)} object(s)"
        )
    
    def ground_layer5(self, hypotheses: List[TokenizationHypothesis], return_all_matches: bool = True) -> tuple[List[TokenizationHypothesis], List[Layer5GroundingResult]]:
        """Apply grounding to hypotheses following the pattern of other layers.
        
        Args:
            hypotheses: Input hypotheses to ground
            return_all_matches: Whether to return all matches
            
        Returns:
            Tuple of (grounded_hypotheses, grounding_results)
        """
        if not hypotheses:
            return [], []
        
        grounded_hypotheses = []
        all_grounding_results = []
        
        for hypothesis in hypotheses:
            # Extract and execute sentences from this hypothesis
            sentence_phrases = []
            for token in hypothesis.tokens:
                if hasattr(token, '_sentence_phrase') and isinstance(token._sentence_phrase, SentencePhrase):
                    sentence_phrases.append(token._sentence_phrase)
            
            if sentence_phrases:
                grounding_results = self.execute_sentences(sentence_phrases)
                all_grounding_results.extend(grounding_results)
                
                # Create enhanced hypothesis with execution results
                enhanced_hypothesis = hypothesis
                enhanced_hypothesis.description += f" + L5G: Executed {len(sentence_phrases)} sentence(s)"
                grounded_hypotheses.append(enhanced_hypothesis)
            else:
                # No sentences found - keep original hypothesis
                grounded_hypotheses.append(hypothesis)
        
        return grounded_hypotheses, all_grounding_results
