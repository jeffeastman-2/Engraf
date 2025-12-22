#!/usr/bin/env python3
"""
LATN Tokenization Hypothesis with Layer-6 Structural Support

This module defines a unified hypothesis class for all layers of the 
LATN (Layer-Aware Tokenization Network) system, including Layer-6 LLM integration.

Layer-6 uses a STRUCTURAL-ONLY representation where:
- Tokens are purely structural markers: [NP, ]NP, [VP, ]VP, etc.
- ALL semantics flow through 76-dim vectors (not token strings)
- Grounding IDs attach to closing brackets
- Individual words (the, red, sphere) are NOT included
"""

from typing import List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Import moved here to avoid circular imports
from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
from engraf.lexer.vector_space import VectorSpace
from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.noun_phrase import NounPhrase


@dataclass
class TokenizationHypothesis:
    """Unified tokenization hypothesis for all LATN layers with Layer-6 support.
    
    This provides a consistent interface across Layer 1, 2, 3, 4, and 5.
    
    The tokens field maintains LATN's compact representation (existing behavior).
    
    The layer6_* fields maintain a STRUCTURAL representation where:
    - Tokens are structural markers only: [NP, ]NP, [VP, ]VP, [PP, ]PP, [SP, ]SP
    - No individual words (the, red, sphere, above, etc.)
    - All semantic content in 76-dim vectors
    - SceneObject references on closing brackets
    """
    tokens: List[VectorSpace]
    confidence: float
    description: str
    replacements: List[tuple] = field(default_factory=list)  # For debugging
    
    # Layer-6 structural representation (NEW)
    layer6_tokens: List[str] = field(default_factory=list)  # Pure structure: [NP, ]NP, etc.
    layer6_vectors: List[np.ndarray] = field(default_factory=list)  # 76-dim semantic vectors
    layer6_scene_refs: List[Optional[Any]] = field(default_factory=list)  # SceneObject references
    
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

    # ========== Layer-6 Support Methods (Structural Approach) ==========
    
    def initialize_layer6_structural(self):
        """Initialize Layer-6 with empty structural representation.
        
        Called after Layer-1. Unlike the verbose approach, we start with
        an empty layer6 representation that gets built up as phrases form.
        
        Layer-1 tokens don't appear in Layer-6 - only the phrase structures
        that emerge in Layers 2-5.
        """
        self.layer6_tokens = []
        self.layer6_vectors = []
        self.layer6_scene_refs = []
    
    def add_layer6_phrase(
        self, 
        phrase_type: str,
        phrase_vector: np.ndarray,
        scene_object: Optional[Any] = None
    ):
        """Add a complete phrase structure to Layer-6 representation.
        
        This adds: [PHRASE_TYPE, ]PHRASE_TYPE as a complete unit.
        
        Args:
            phrase_type: "NP", "PP", "VP", or "SP"
            phrase_vector: The composite 76-dim vector for this phrase
            scene_object: Optional SceneObject reference (for grounded NPs)
        
        Example:
            For an NP "the red sphere" grounded to sphere_1:
            - Adds tokens: ["[NP", "]NP"]
            - Adds vectors: [opening_marker_vec, full_np_vector]
            - Adds refs: [None, sphere_1]
        """
        # Create opening marker
        open_marker = f"[{phrase_type}"
        open_vector = self._create_marker_vector(phrase_type, is_opening=True)
        
        # Create closing marker - this carries the full phrase semantics
        close_marker = f"]{phrase_type}"
        close_vector = phrase_vector.copy()  # The composite vector!
        
        # Add opening marker
        self.layer6_tokens.append(open_marker)
        self.layer6_vectors.append(open_vector)
        self.layer6_scene_refs.append(None)
        
        # Add closing marker with semantics and grounding
        self.layer6_tokens.append(close_marker)
        self.layer6_vectors.append(close_vector)
        self.layer6_scene_refs.append(scene_object)
    
    def wrap_layer6_with_phrase(
        self,
        start_idx: int,
        end_idx: int,
        phrase_type: str,
        phrase_vector: np.ndarray,
        scene_object: Optional[Any] = None
    ):
        """Wrap existing Layer-6 content with a new phrase structure.
        
        Used when a phrase contains other phrases (e.g., PP contains NP,
        VP contains NP and PP, etc.)
        
        Args:
            start_idx: Starting index in layer6_tokens to wrap
            end_idx: Ending index in layer6_tokens to wrap (inclusive)
            phrase_type: "PP", "VP", or "SP"
            phrase_vector: The composite 76-dim vector for this phrase
            scene_object: Optional SceneObject reference
        
        Example:
            Before: ["[NP", "]NP"]
            After wrapping with PP: ["[PP", "[NP", "]NP", "]PP"]
        """
        # Create markers
        open_marker = f"[{phrase_type}"
        open_vector = self._create_marker_vector(phrase_type, is_opening=True)
        
        close_marker = f"]{phrase_type}"
        close_vector = phrase_vector.copy()
        
        # Insert opening marker at start
        self.layer6_tokens.insert(start_idx, open_marker)
        self.layer6_vectors.insert(start_idx, open_vector)
        self.layer6_scene_refs.insert(start_idx, None)
        
        # Insert closing marker at end (adjusted for insertion)
        close_idx = end_idx + 2  # +1 for the opening marker, +1 for after end_idx
        self.layer6_tokens.insert(close_idx, close_marker)
        self.layer6_vectors.insert(close_idx, close_vector)
        self.layer6_scene_refs.insert(close_idx, scene_object)
    
    def _create_marker_vector(self, phrase_type: str, is_opening: bool) -> np.ndarray:
        """Create a 76-dimensional vector for a structural marker.
        
        Args:
            phrase_type: "NP", "PP", "VP", or "SP"
            is_opening: True for "[NP", False for "]NP"
        
        Returns:
            76-dimensional numpy array (mostly zeros, structural marker only)
        """
        vec = np.zeros(len(VECTOR_DIMENSIONS))
        
        # Set the appropriate dimension for this phrase type
        if phrase_type in VECTOR_DIMENSIONS:
            idx = VECTOR_DIMENSIONS.index(phrase_type)
            vec[idx] = 1.0 if is_opening else -1.0
        
        return vec
    
    def get_layer6_representation(self) -> Tuple[List[str], List[np.ndarray], List[Optional[Any]]]:
        """Get the complete Layer-6 representation.
        
        Returns:
            Tuple of (structural_tokens, semantic_vectors, scene_references)
        """
        return (self.layer6_tokens, self.layer6_vectors, self.layer6_scene_refs)
    
    def print_layer6_tokens(self):
        """Print the Layer-6 structural representation with vector info."""
        print("Layer-6 Structural Sequence:")
        for i, (token, vec, scene_ref) in enumerate(zip(
            self.layer6_tokens, 
            self.layer6_vectors, 
            self.layer6_scene_refs
        )):
            # Show non-zero dimensions in vector
            non_zero = np.nonzero(vec)[0]
            if len(non_zero) > 0 and len(non_zero) <= 5:
                dims = [VECTOR_DIMENSIONS[idx] for idx in non_zero[:5]]
                vec_str = f"dims={dims}"
            else:
                vec_str = f"nnz={len(non_zero)}"
            
            ref_str = f" -> {scene_ref.object_id}" if scene_ref else ""
            print(f"  [{i:2d}] {token:6s} ({vec_str}){ref_str}")
    
    def layer6_to_string(self) -> str:
        """Convert Layer-6 representation to a string for LLM input.
        
        Returns:
            Space-separated structural tokens with object references.
            Example: "[SP [VP [NP]NP<sphere_1> [PP[NP]NP<cube_1>]PP ]VP ]SP"
        """
        result = []
        for token, scene_ref in zip(self.layer6_tokens, self.layer6_scene_refs):
            if scene_ref:
                result.append(f"{token}<{scene_ref.object_id}>")
            else:
                result.append(token)
        return " ".join(result)
    
    def layer6_vocabulary(self) -> set:
        """Return the minimal vocabulary needed for Layer-6 structural tokens.
        
        This is typically just: [NP, ]NP, [PP, ]PP, [VP, ]VP, [SP, ]SP
        plus special tokens like <SEP>, <BOS>, <EOS>, <OBJ>
        """
        return {
            "[NP", "]NP",
            "[PP", "]PP", 
            "[VP", "]VP",
            "[SP", "]SP",
            "<SEP>", "<BOS>", "<EOS>", "<OBJ>"
        }
    
    # ========== Existing Methods (Unchanged) ==========
    
    spaces = "        "

    def printNP(self, i, token):
        """Print a noun phrase token."""
        if token.phrase is not None:
            str = token.phrase.printString()
            if token.isa("conj"):
                print(f"{self.spaces}[{i}] [CONJ-NP] {str}")
            else:
                print(f"{self.spaces}[{i}] [NP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word} = {token.non_zero_dims()}")

    def printPP(self, i, token):
        original_pp = token.phrase
        if original_pp:
            str = original_pp.printString()
            if token.isa("conj"):
                print(f"{self.spaces}[{i}] [CONJ-PP] {str}")
            else:
                print(f"{self.spaces}[{i}] [PP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word} = {token.non_zero_dims()}")

    def printVP(self, i, token):
        """Print a verb phrase token."""
        original_vp = token.phrase
        if original_vp:
            str = original_vp.printString()
            if token.isa("conj"):
                print(f"{self.spaces}[{i}] [CONJ-VP] {str}")
            else:
                print(f"{self.spaces}[{i}] [VP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word}")

    def printSentence(self, i, token):
        """Print a sentence token."""
        original_sp = token.phrase
        if original_sp:
            str = original_sp.printString()
            if token.isa("conj"):
                print(f"{self.spaces}[{i}] [CONJ-SP] {str}")
            else:
                print(f"{self.spaces}[{i}] [SP] {str}")
        else:
            print(f"{self.spaces}[{i}] {token.word}")

    def print_tokens(self):
        """Print the Layer-6 structural tokenization followed by compact tokens.

        This prints the new Layer-6 representation first (structural markers
        with any attached grounding IDs), then falls back to the existing
        per-token compact diagnostic output to aid debugging.
        """
        # Print Layer-6 structural string (if present)
        if self.layer6_tokens:
            print("--- Layer-6 Structural String ---")
            try:
                print(self.layer6_to_string())
            except Exception:
                # Defensive fallback
                print("(unable to render layer6 string)")
            print()

            # Print detailed Layer-6 token listing
            try:
                self.print_layer6_tokens()
            except Exception:
                pass

            print()  # separator

        # Then print the traditional compact token diagnostics
        for i, token in enumerate(self.tokens):
            if (token).isa("NP") :
                self.printNP(i,token)
            elif (token).isa("PP") :
                self.printPP(i,token)
            elif token.isa("VP") :
                self.printVP(i,token)
            elif token.isa("SP") :
                self.printSentence(i,token)
            else:
                print(f"{self.spaces}[{i}] {token}")

    def print_detailed(token): 
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
                for np in conj_phrase.phrases:
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
