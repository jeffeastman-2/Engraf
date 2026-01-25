"""Layer-6 Response Generator.

Provides utilities for generating expected LLM responses from Layer-6
structural representations and for extracting Layer-6 from parsed sentences.
"""

from typing import Optional, Tuple

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.llm_layer6.synthetic_generator import populate_layer6_from_sentence_phrase
from engraf.utils.verb_inflector import verb_to_gerund


class Layer6ResponseGenerator:
    """Generates Layer-6 representations and expected responses for sentences."""
    
    def __init__(self, scene=None):
        """Initialize the generator.
        
        Args:
            scene: Optional SceneModel for grounded interpretation
        """
        self.scene = scene
    
    def get_layer6_from_parsed(self, sentence: str, sentence_phrase) -> Tuple[Optional[str], Optional[str]]:
        """Get the Layer-6 input and expected response from an already-parsed sentence phrase.
        
        This method should be used when you have already parsed the sentence (e.g., via
        the SentenceInterpreter) and want to generate Layer-6 output from that parse.
        
        Args:
            sentence: The original sentence text
            sentence_phrase: The parsed SentencePhrase from LATN Layer-5
            
        Returns:
            Tuple of (layer6_input, expected_response) or (None, None) on failure
        """
        try:
            if sentence_phrase is None:
                return None, None
            
            # Create a minimal hypothesis to hold Layer-6 tokens
            from engraf.lexer.hypothesis import TokenizationHypothesis
            hyp = TokenizationHypothesis(tokens=[], confidence=1.0)
            
            # Populate Layer-6 from the sentence phrase
            populate_layer6_from_sentence_phrase(hyp, sentence_phrase)
            
            # Get structural tokens
            if not hasattr(hyp, 'layer6_tokens') or not hyp.layer6_tokens:
                return None, None
            
            layer6_input = hyp.layer6_to_string() + " <SEP>"
            expected_response = "<BOS> " + self.generate_response_from_parse(sentence, sentence_phrase) + " <EOS>"
            
            return layer6_input, expected_response
            
        except Exception:
            return None, None
    
    def get_layer6_representation(self, sentence: str) -> Tuple[Optional[str], Optional[str]]:
        """Get the Layer-6 input and expected response for a sentence.
        
        Args:
            sentence: The sentence to process
            
        Returns:
            Tuple of (layer6_input, expected_response) or (None, None) on failure
        """
        try:
            # Don't pass scene for Layer-6 parsing - we want syntactic structure only,
            # not grounded interpretation (which would fail on unresolved pronouns)
            executor = LATNLayerExecutor(scene_model=None)
            result = executor.execute_layer5(sentence, tokenize_only=True, report=False)
            
            if not result.success or not result.hypotheses:
                return None, None
            
            hyp = result.hypotheses[0]
            
            # Populate Layer-6 from the parsed sentence phrase
            sentence_phrase = None
            if hyp.tokens and hasattr(hyp.tokens[0], 'phrase'):
                sentence_phrase = hyp.tokens[0].phrase
                if sentence_phrase is not None:
                    populate_layer6_from_sentence_phrase(hyp, sentence_phrase)
            
            # Get structural tokens
            if not hasattr(hyp, 'layer6_tokens') or not hyp.layer6_tokens:
                return None, None
            
            layer6_input = hyp.layer6_to_string() + " <SEP>"
            expected_response = "<BOS> " + self.generate_response_from_parse(sentence, sentence_phrase) + " <EOS>"
            
            return layer6_input, expected_response
            
        except Exception:
            return None, None
    
    @staticmethod
    def generate_response_from_parse(sentence: str, sentence_phrase) -> str:
        """Generate the expected Layer-6 LLM response from parsed structure.
        
        Uses the parsed sentence phrase's vector dimensions to determine type:
        - Imperative: has verb+action in predicate (e.g., "Drawing the red cube.")
        - Declarative with tobe: has tobe (e.g., "The cube is red.")
        - Interrogative: has question dimension or tobe-initial structure
        
        Args:
            sentence: The original input sentence
            sentence_phrase: Parsed SentencePhrase from LATN
            
        Returns:
            Expected response string
        """
        if sentence_phrase is None:
            return f"Processing: {sentence}"
        
        # Check for interrogative (question dimension in sentence vector)
        if hasattr(sentence_phrase, 'vector') and sentence_phrase.vector['question'] > 0:
            return f"Answering: {sentence}"
        
        # Check for tobe-initial sentences (likely questions like "is the cube red")
        # These have a predicate with tobe verb but no subject before predicate
        if hasattr(sentence_phrase, 'predicate') and sentence_phrase.predicate:
            pred = sentence_phrase.predicate
            if hasattr(pred, 'vector') and pred.vector['tobe'] > 0:
                # If no subject and tobe in predicate, likely a question
                if not sentence_phrase.subject:
                    return f"Answering: {sentence}"
                # Otherwise it's a declarative
                return f"Acknowledged: {sentence}"
        
        # Check for declarative with tobe
        if hasattr(sentence_phrase, 'tobe') and sentence_phrase.tobe:
            return f"Acknowledged: {sentence}"
        
        # Check for imperative (verb+action in predicate)
        if hasattr(sentence_phrase, 'predicate') and sentence_phrase.predicate:
            pred = sentence_phrase.predicate
            if hasattr(pred, 'vector'):
                pred_vec = pred.vector
                if pred_vec['verb'] > 0 and pred_vec['action'] > 0:
                    # Get the verb word from the predicate
                    verb_word = getattr(pred, 'verb_word', None)
                    if verb_word:
                        gerund = verb_to_gerund(verb_word).capitalize()
                        rest = ' '.join(sentence.split()[1:])
                        return f"{gerund} {rest}."
                    else:
                        # Fall back to first word
                        words = sentence.split()
                        if words:
                            gerund = verb_to_gerund(words[0]).capitalize()
                            rest = ' '.join(words[1:])
                            return f"{gerund} {rest}."
        
        # Default
        return f"Processing: {sentence}"
    
    def print_layer6(self, sentence: str, sentence_phrase=None):
        """Print the Layer-6 representation for a sentence.
        
        Args:
            sentence: The sentence to process
            sentence_phrase: Optional pre-parsed SentencePhrase. If provided, uses this
                           instead of re-parsing the sentence (preferred for grounded
                           interpretation where pronouns have been resolved).
        """
        if sentence_phrase is not None:
            layer6_input, expected_response = self.get_layer6_from_parsed(sentence, sentence_phrase)
        else:
            layer6_input, expected_response = self.get_layer6_representation(sentence)
        
        if layer6_input and expected_response:
            print(f"   🧠 Layer-6 Input:    {layer6_input}")
            print(f"   💬 Layer-6 Response: {expected_response}")
        else:
            print(f"   🧠 Layer-6: (parse failed)")
