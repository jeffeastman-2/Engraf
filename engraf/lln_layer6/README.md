# Layer-6 LLM Integration (engraf/llm_layer6)

This small package is a starting point for the Layer-6 LLM integration
work on the `layer6-llm-integration` branch.

Files
- `dataset_extractor.py` — convert final Layer-6 hypotheses into JSONL
  training pairs. Use `create_training_pair_from_hyp()` and `write_jsonl()`.
- `adapter.py` — skeleton `Layer6Encoder` that projects LATN 76-d vectors
  and structural tokens into a shared embedding space. A PyTorch-based
  encoder is provided if `torch` is installed; otherwise `Layer6Encoder`
  will be `None`.

Usage
1. Parse questions through existing LATN pipeline to obtain final
   hypotheses (Layer-5 final hypotheses exposing `get_layer6_representation()`
   and `layer6_to_string()`).
2. Use `dataset_extractor.create_training_pair_from_hyp(hyp, answer)` to
   convert to a JSON-serializable example.
3. Aggregate examples and write with `write_jsonl()`.

Next steps
- Implement `TokenizationHypothesis` layer6 helper methods (`initialize_*`,
  `add_layer6_phrase`, `wrap_layer6_with_phrase`) in the main codebase.
- Add small CLI to synthesize scenes and generate a seed dataset.
- Implement and train a tiny transformer using the adapter.
