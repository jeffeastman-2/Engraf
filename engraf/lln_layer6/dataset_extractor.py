"""Dataset extractor for Layer-6 LLM training examples.

Provides utilities to convert a final Layer-6 `TokenizationHypothesis`
into a training example dict/json-ready structure. This is intentionally
lightweight and does not import heavy ML dependencies.

Functions
- create_training_pair_from_hyp(final_hyp, answer): return dict
- write_jsonl(path, examples): write list of examples to JSONL

"""
from typing import Any, Dict, List, Optional
import json


def create_training_pair_from_hyp(final_hyp: Any, answer: str) -> Dict[str, Any]:
    """Create a training example from a Layer-6 final hypothesis.

    Args:
        final_hyp: object exposing `get_layer6_representation()` and
                   `layer6_to_string()` as described in the Layer-6 guides.
        answer: gold answer string (natural language)

    Returns:
        dict suitable for JSON serialization containing structural tokens,
        latent vectors (as lists), scene refs (as object IDs), and input/target strings.
    """
    tokens, vectors, scene_refs = final_hyp.get_layer6_representation()

    # Convert numpy arrays to lists if necessary
    conv_vecs: List[List[float]] = []
    for v in vectors:
        try:
            conv_vecs.append(v.tolist())
        except Exception:
            # Already a list
            conv_vecs.append(list(v))

    # Convert scene object references to object IDs (JSON serializable)
    conv_refs: List[Optional[str]] = []
    for ref in scene_refs:
        if ref is None:
            conv_refs.append(None)
        elif hasattr(ref, 'object_id'):
            conv_refs.append(ref.object_id)
        elif hasattr(ref, 'entity_id'):
            conv_refs.append(ref.entity_id)
        else:
            conv_refs.append(str(ref))

    input_string = final_hyp.layer6_to_string() + " <SEP>"
    target_string = "<BOS> " + answer + " <EOS>"

    return {
        "structural_tokens": tokens,
        "semantic_vectors": conv_vecs,
        "scene_grounding": conv_refs,
        "input_string": input_string,
        "target_string": target_string,
    }


def write_jsonl(path: str, examples: List[Dict[str, Any]]) -> None:
    """Write examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load examples from a JSONL file."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out
