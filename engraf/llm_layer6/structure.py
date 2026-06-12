"""Layer-6 structural representation.

Phase-1 seam 5 of factoring the LATN core out of Engraf. Previously the
layer6_* fields and methods lived on TokenizationHypothesis in the core
LATN module. L1-5 grounding never touches them -- only L6 training-data
construction (dataset_generator, synthetic_generator) does -- so they
were core baggage the extracted LATN didn't need.

They now live here, on Layer6Structure, attached to a hypothesis as
`hyp.l6` by whichever L6 code wants it. The LATN core is L6-free.

Field/method names drop the `layer6_` prefix since the type and module
already say L6:
  hyp.layer6_tokens         -> hyp.l6.tokens
  hyp.layer6_vectors        -> hyp.l6.vectors
  hyp.layer6_scene_refs     -> hyp.l6.scene_refs
  hyp.initialize_layer6_structural() -> hyp.l6 = Layer6Structure()
  hyp.add_layer6_phrase(...)         -> hyp.l6.add_phrase(...)
  hyp.wrap_layer6_with_phrase(...)   -> hyp.l6.wrap_with_phrase(...)
  hyp.layer6_to_string()             -> hyp.l6.to_string()
  hyp.print_layer6_tokens()          -> hyp.l6.print_tokens()
"""

from typing import Any, List, Optional

import numpy as np

from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS


class Layer6Structure:
    """A bracketed phrase tree with a composite semantic vector and an
    optional grounded-entity reference attached to each closing bracket.

    L6 uses a structural-only representation:
    - Tokens are markers only: "[NP", "]NP", "[VP", "]VP", "[PP", "]PP",
      "[SP", "]SP". No individual words.
    - All semantic content lives in the VECTOR_LENGTH-dim vectors.
    - Grounding ids attach to closing brackets.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.vectors: List[np.ndarray] = []
        self.scene_refs: List[Optional[Any]] = []

    def add_phrase(
        self,
        phrase_type: str,
        phrase_vector: np.ndarray,
        scene_object: Optional[Any] = None,
    ):
        """Append a complete phrase as `[PHRASE_TYPE ]PHRASE_TYPE` with
        the composite phrase vector on the closing bracket.

        Example:
            For an NP "the red sphere" grounded to sphere_1:
              tokens:     ["[NP", "]NP"]
              vectors:    [opening_marker_vec, full_np_vector]
              scene_refs: [None, sphere_1]
        """
        open_marker = f"[{phrase_type}"
        open_vector = self._create_marker_vector(phrase_type, is_opening=True)

        close_marker = f"]{phrase_type}"
        close_vector = phrase_vector.copy()  # The composite vector!

        self.tokens.append(open_marker)
        self.vectors.append(open_vector)
        self.scene_refs.append(None)

        self.tokens.append(close_marker)
        self.vectors.append(close_vector)
        self.scene_refs.append(scene_object)

    def wrap_with_phrase(
        self,
        start_idx: int,
        end_idx: int,
        phrase_type: str,
        phrase_vector: np.ndarray,
        scene_object: Optional[Any] = None,
    ):
        """Wrap existing content with a new phrase bracket pair. Used when
        a phrase contains other phrases (PP contains NP, VP contains NP+PP).

        Example:
            Before: ["[NP", "]NP"]
            After wrapping with PP: ["[PP", "[NP", "]NP", "]PP"]
        """
        open_marker = f"[{phrase_type}"
        open_vector = self._create_marker_vector(phrase_type, is_opening=True)

        close_marker = f"]{phrase_type}"
        close_vector = phrase_vector.copy()

        self.tokens.insert(start_idx, open_marker)
        self.vectors.insert(start_idx, open_vector)
        self.scene_refs.insert(start_idx, None)

        close_idx = end_idx + 2  # +1 for opening marker, +1 for after end_idx
        self.tokens.insert(close_idx, close_marker)
        self.vectors.insert(close_idx, close_vector)
        self.scene_refs.insert(close_idx, scene_object)

    def _create_marker_vector(self, phrase_type: str, is_opening: bool) -> np.ndarray:
        """Build a VECTOR_LENGTH-dim vector for a structural marker."""
        vec = np.zeros(len(VECTOR_DIMENSIONS))
        if phrase_type in VECTOR_DIMENSIONS:
            idx = VECTOR_DIMENSIONS.index(phrase_type)
            vec[idx] = 1.0 if is_opening else -1.0
        return vec

    def to_string(self) -> str:
        """Render as a single string suitable for LLM input.

        Example: "[SP [VP [NP]NP<sphere_1> [PP[NP]NP<cube_1>]PP ]VP ]SP"
        """
        result = []
        for token, scene_ref in zip(self.tokens, self.scene_refs):
            if scene_ref:
                result.append(f"{token}<{scene_ref.object_id}>")
            else:
                result.append(token)
        return " ".join(result)

    def print_tokens(self):
        """Diagnostic dump of the structural sequence with vector info."""
        print("Layer-6 Structural Sequence:")
        for i, (token, vec, scene_ref) in enumerate(
            zip(self.tokens, self.vectors, self.scene_refs)
        ):
            non_zero = np.nonzero(vec)[0]
            if len(non_zero) > 0 and len(non_zero) <= 5:
                dims = [VECTOR_DIMENSIONS[idx] for idx in non_zero[:5]]
                vec_str = f"dims={dims}"
            else:
                vec_str = f"nnz={len(non_zero)}"

            ref_str = f" -> {scene_ref.object_id}" if scene_ref else ""
            print(f"  [{i:2d}] {token:6s} ({vec_str}){ref_str}")


__all__ = ["Layer6Structure"]
