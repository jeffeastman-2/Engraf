#!/usr/bin/env python3
"""
LATN Tokenization Hypothesis

This module defines a unified hypothesis class for all layers of the 
LATN (Layer-Aware Tokenization Network) system.
"""

from typing import List, Optional, Any
from dataclasses import dataclass, field

# Import moved here to avoid circular imports
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase


@dataclass
class TokenizationHypothesis:
    """Unified tokenization hypothesis for all LATN layers.
    
    This provides a consistent interface across Layer 1, 2, 3, and 4.
    The replacements field is used only for debugging/analysis to track
    what transformations each layer performed.
    """
    tokens: List[VectorSpace]
    confidence: float
    description: str
    replacements: List[tuple] = field(default_factory=list)  # For debugging: (start_idx, end_idx, new_token)
    
    def __len__(self):
        """Return the number of tokens in this hypothesis."""
        return len(self.tokens)
    
    def __getitem__(self, index):
        """Allow indexing into the tokens list."""
        return self.tokens[index]
    
    def __iter__(self):
        """Allow iteration over the tokens."""
        return iter(self.tokens)
    
    def get_tokens_of_type(self, token_type: str) -> List[VectorSpace]:
        """Get all tokens of a specific type (e.g., 'NP', 'PP', 'VP')."""
        return [token for token in self.tokens if token.isa(token_type)]
    
    def has_token_type(self, token_type: str) -> bool:
        """Check if this hypothesis contains any tokens of the specified type."""
        return any(token.isa(token_type) for token in self.tokens)
    
    def token_words(self) -> List[str]:
        """Get the word strings for all tokens."""
        return [token.word for token in self.tokens]

    spaces = "        "

    def printNP(self, i, token):
        """Print a noun phrase token."""
        original_np = token._original_np
        if token.isa("conj") and token._original_np:
            str = original_np.printString()
            print(f"{self.spaces}[{i}] [CONJ-NP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word} = {token.non_zero_dims()}")

    def printPP(self, i, token):
        if token.isa("conj") and token._original_pp:
            str = token._original_pp.printString()
            print(f"{self.spaces}[{i}] [CONJ-PP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word} = {token.non_zero_dims()}")

    def printVP(self, i, token):
        """Print a verb phrase token."""
        original_vp = token._original_vp
        if token.isa("conj") and token._original_vp:
            str = original_vp.printString()
            print(f"{self.spaces}[{i}] [CONJ-VP] {str}")
        else:
            original_vp_np = original_vp.noun_phrase
            if isinstance(original_vp_np, ConjunctionPhrase):
                str = original_vp_np.printString()
                print(f"{self.spaces}[{i}] [CONJ-VP] {original_vp.verb} => {str}")
            elif isinstance(original_vp_np, NounPhrase):
                str = original_vp_np.printString()
                print(f"{self.spaces}[{i}] [VP] {original_vp.verb} => {str}")
            else:
                print(f"{self.spaces}[{i}] {token.word}")

    def printSentence(self, i, token):
        """Print a sentence token."""
        print(f"{self.spaces}[{i}] [Sentence] {token.word}")

    def print_tokens(self):
        """Print all tokens, each on a new line. Useful for demo examples."""
        for i, token in enumerate(self.tokens):
            if (token).isa("NP") :
                self.printNP(i,token)
            elif (token).isa("PP") :
                self.printPP(i,token)
            elif token.isa("VP") :
                self.printVP(i,token)
            elif token.isa("Sentence") :
                self.printSentence(i,token)
            else:
                print(f"{self.spaces}[{i}] {token}")

    def print_detailed(token): #
        if hasattr(token, '_grounded_phrase') and token._grounded_phrase:
            grounded_phrase = token._grounded_phrase
            grounding_info = grounded_phrase.grounding
            if grounding_info is not None:
                # Check for multiple objects first (new approach)
                if 'scene_objects' in grounding_info and grounding_info['scene_objects']:
                    scene_objects = grounding_info['scene_objects']
                    confidence = grounding_info.get('confidence', 'unknown')
                    
                    # Display multiple objects if more than one
                    if len(scene_objects) > 1:
                        object_ids = [obj.object_id for obj in scene_objects]
                        object_display = f"[{', '.join(object_ids)}]"
                        print(f"      [{i}] GNP({object_display} @conf={confidence:.3f}) [PLURAL: {len(scene_objects)} objects]")
                    else:
                        # Single object
                        scene_obj = scene_objects[0]
                        print(f"      [{i}] GNP({scene_obj.object_id} @conf={confidence:.3f})")
                
                # Fallback to old single object approach for backward compatibility
                elif 'scene_object' in grounding_info:
                    scene_obj = grounding_info['scene_object']
                    confidence = grounding_info.get('confidence', 'unknown')
                    
                    # Check if this NP has attached preps to display
                    preps_info = ""
                    if hasattr(token, '_original_np') and token._original_np:
                        np = token._original_np
                        if hasattr(np, 'preps') and np.preps:
                            prep_descs = []
                            for prep in np.preps:
                                if hasattr(prep, 'preposition') and prep.preposition:
                                    prep_desc = prep.preposition
                                    if hasattr(prep, 'noun_phrase') and prep.noun_phrase:
                                        if hasattr(prep.noun_phrase, 'get_original_text'):
                                            prep_desc += f" {prep.noun_phrase.get_original_text()}"
                                    prep_descs.append(prep_desc)
                            if prep_descs:
                                preps_info = f" +PP({', '.join(prep_descs)})"
                    
                    print(f"      [{i}] GNP({scene_obj.object_id} @conf={confidence:.3f}){preps_info})")
                else:
                    print(f"      [{i}]  → GROUNDED: {grounding_info}")
            else:
                print(f"      [{i}] {token}")
        elif hasattr(token, '_original_pp') and token._original_pp:
            # Handle PP tokens - show grounding status of contained NP
            pp = token._original_pp
            np = pp.noun_phrase
            if np:
                if hasattr(np, 'grounding') and np.grounding:
                    grounding_info = np.grounding
                    if 'scene_object' in grounding_info:
                        scene_obj = grounding_info['scene_object']
                        confidence = grounding_info.get('confidence', 'unknown')
                        print(f"      [{i}] GPP({token.word} → {scene_obj.object_id} @{confidence:.3f})")
                    else:
                        print(f"      [{i}] GPP({token.word} → grounded)")
                else:
                    print(f"      [{i}] PP({pp.preposition} {np})")
            else:
                print(f"      [{i}] {token.word}???")
        elif hasattr(token, '_original_np') and token._original_np:
            # Check if this is a ConjunctionPhrase (CONJ-NP)
            from engraf.pos.conjunction_phrase import ConjunctionPhrase
            if isinstance(token._original_np, ConjunctionPhrase):
                # Show the coordination structure with individual NP contents
                conj_phrase = token._original_np
                np_parts = []
                for np in conj_phrase.flatten():
                    if hasattr(np, 'get_original_text'):
                        np_parts.append(f'"{np.get_original_text()}"')
                    else:
                        np_parts.append(str(np))
                
                coordination_text = f" {conj_phrase.conjunction} ".join(np_parts)
                print(f"      [{i}] VS(word='CONJ-NP', {{ NP=1.00, conj=1.00 }}) → [{coordination_text}]")
            else:
                # Regular NP token
                print(f"      [{i}] {token}")
        else:
            print(f"      [{i}] {token}")
    
    def __repr__(self):
        """Standard representation showing tokens and confidence."""
        token_words = self.token_words()
        return f"TokenizationHypothesis(conf={self.confidence:.2f}, tokens={token_words}, desc='{self.description}')"
