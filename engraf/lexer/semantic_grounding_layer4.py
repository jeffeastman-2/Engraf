#!/usr/bin/env python3
"""
LATN Layer 4 Semantic Grounding

This module provides semantic grounding capabilities for LATN Layer 4 VerbPhrase tokens.
It handles verb phrase extraction and action execution.
"""

from typing import List, Optional
from dataclasses import dataclass

from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.visualizer.scene.scene_model import SceneModel
from engraf.visualizer.scene.scene_object import SceneObject
from engraf.lexer.hypothesis import TokenizationHypothesis


@dataclass
class Layer4GroundingResult:
    """Result of Layer 4 semantic grounding operation."""
    success: bool
    confidence: float
    description: str = ""


class Layer4SemanticGrounder:
    """Semantic grounding for LATN Layer 4 VerbPhrase tokens.
    
    Layer 4 performs semantic grounding of verb phrases - understanding their meaning
    and context within the scene. It does NOT execute actions or modify the scene.
    """
    
    def __init__(self, scene_model: SceneModel):
        self.scene_model = scene_model
        
    def validate_vp_with_np(self, vp: VerbPhrase, np: NounPhrase) -> bool:
        vp_has_pp  = len(vp.prepositions) > 0
        vp_has_adj = len(vp.adjective_complements) > 0
        vp_has_amount = vp.amount is not None
        if (vp_has_adj and vp_has_pp) or (vp_has_adj and vp_has_amount) or (vp_has_pp and vp_has_amount):
            # multiple vp complements not allowed
            return False
        np_preps_have_spatial_pp = False
        for prep in np.prepositions:
            if prep.vector.isa("spatial_location") or prep.vector.isa("spatial_proximity"):
                np_preps_have_spatial_pp = True
                break

        # --- STYLE / STATE-CHANGE: make, color, texture ---
        # Expect: grounded NP + adjective complement (resulting state).
        # NOTE: color/texture carry both "transform" and "style"; prefer style rule.
        if vp.vector.isa("style") or (vp.vector.isa("transform") and not (
            vp.vector.isa("move") or vp.vector.isa("rotate") or vp.vector.isa("scale")
        )):
            if not np.grounding:            # must operate on an existing object
                return False
            if not vp_has_adj:                 # need an adjective/state complement (e.g., red, rough)
                return False
            return True

        # --- MOTION / ORIENTATION / SIZE: move, rotate, scale (incl. x/y/zrotate) ---
        # Expect: grounded NP + PP (to/by/around/etc.). Adjective complements not appropriate here.
        if vp.vector.isa("move") or vp.vector.isa("rotate") or vp.vector.isa("scale"):
            if not np.grounding:
                return False
            if not vp_has_pp:
                return False
            if vp_has_adj:
                return False
            return True

        # --- ORGANIZE: align, position, group, ungroup ---
        # Expect: grounded NP; PP often present (align with, position at, group into) but not required.
        if vp.vector.isa("organize"):
            if not np.grounding:
                return False
            return True

        # --- EDIT / SELECT / NAMING: delete/copy/remove/paste, select, call/name ---
        # Expect: grounded NP; PP optional (e.g., remove from, paste into). Adjective complement not used.
        if vp.vector.isa("edit") or vp.vector.isa("select") or vp.vector.isa("naming"):
            if not np.grounding:
                return False
            return True

        # --- CREATE: create, draw, build, place (introduce new object) ---
        # Expect: UNgrounded NP (type introduction). Adj complement disallowed here;
        # PP (on/in/above) is allowed but not required.
        if vp.vector.isa("create"):
            if np.grounding:
                is_ok = False
                if vp_has_pp:
                    if vp.prepositions[0].vector.isa("spatial_location") or vp.prepositions[0].vector.isa("spatial_proximity"):
                        # grounded NP with spatial PP - probably a location specifier, not object type
                        # unground the NP and allow further processing
                        np.grounding = None
                        is_ok = True
                if not is_ok:
                    return False
            if vp_has_adj:
                return False
            # PP optional for placement; both "draw a cube" and "draw a cube on the table" are ok.
            # if not vp_has_pp and np_preps_have_spatial_pp the PP was 
            # probably intended to specify location, so reject if missing.
            if not vp_has_pp and np_preps_have_spatial_pp:
                return False
            return True

        # Fallback: accept if no specific constraints apply.
        return True
        
    def validate_vp(self, vp: VerbPhrase) -> bool:
        np = vp.noun_phrase
        if isinstance(np, ConjunctionPhrase):
            for cnp in np.phrases:
                if not self.validate_vp_with_np(vp, cnp):
                    return False
        else:
            if not self.validate_vp_with_np(vp, np):
                return False
        return True

    def ground_layer4(self, hypotheses: List[TokenizationHypothesis]) -> List[Layer4GroundingResult]:
        """Semantically ground verb phrases by analyzing their meaning and scene context.

        This performs semantic analysis of verb phrases without executing actions.
        Eliminate verb phrase hypotheses that do not have valid verb phrases.
        
        Args:
            hypotheses: List of TokenizationHypothesis objects to ground

        Returns:
            Processed hypotheses with validated VP semantic grounding
        """
        result_hypotheses = []
        for hyp in hypotheses:
            # Validate the verb phrases in the hypothesis (can have >= 0)
            valid = True
            for token in hyp.tokens:
                if token.isa("VP"):
                    if token.isa("conj"):
                        vps = token.phrase.phrases
                        for vp in vps:
                            if not self.validate_vp(vp):
                                valid = False
                                break
                    else:
                        if not self.validate_vp(token.phrase):
                            valid = False
                            break
                elif token.isa("NP"):
                    continue
                elif token.isa("conj"):
                    continue
                else:
                    break
            if valid:
                result_hypotheses.append(hyp)
        return result_hypotheses
