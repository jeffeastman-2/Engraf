import unittest
from engraf.lexer.token_stream import tokenize
from engraf.lexer.latn_tokenizer import latn_tokenize, TokenizationHypothesis
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features


class TestLATNLexicalLayer(unittest.TestCase):
    """
    Test cases for LATN (Layered Augmented Transition Network) lexical layer.
    
    The lexical layer should return multiple ranked hypotheses for ambiguous tokenizations,
    particularly when multi-word tokens contain existing single words.
    """
    
    def setUp(self):
        """Set up test vocabulary entries that create tokenization ambiguities."""
        # Ensure we have both single words and multi-word compounds in vocabulary
        # that could create parsing ambiguities
        
        # Check if we have the words needed for our test
        self.has_light = 'light' in SEMANTIC_VECTOR_SPACE
        self.has_house = 'house' in SEMANTIC_VECTOR_SPACE  
        self.has_lighthouse = 'light house' in SEMANTIC_VECTOR_SPACE
        
        # Add some test entries if they don't exist (for this test only)
        if not self.has_light:
            SEMANTIC_VECTOR_SPACE['light'] = vector_from_features("noun adj verb")
        
        if not self.has_house:
            SEMANTIC_VECTOR_SPACE['house'] = vector_from_features("noun verb")
            
        if not self.has_lighthouse:
            SEMANTIC_VECTOR_SPACE['light house'] = vector_from_features("noun")
    
    def tearDown(self):
        """Clean up test vocabulary entries."""
        # Remove test entries if we added them
        if not self.has_light and 'light' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['light']
        if not self.has_house and 'house' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['house']
        if not self.has_lighthouse and 'light house' in SEMANTIC_VECTOR_SPACE:
            del SEMANTIC_VECTOR_SPACE['light house']
    
    def test_current_tokenization_commits_early(self):
        """
        Test that current tokenization commits to multi-word token when available.
        This demonstrates the current behavior before LATN modification.
        """
        sentence = "draw a light house at [0,0,0]"
        tokens = tokenize(sentence)
        
        # Current behavior: should prefer "light house" over "light" + "house"
        token_words = [token.word for token in tokens]
        
        # Verify the multi-word token is chosen
        self.assertIn('light house', token_words)
        self.assertNotIn('light', token_words)  # Single 'light' should not appear
        self.assertNotIn('house', token_words)  # Single 'house' should not appear separately
        
        print(f"Current tokenization: {token_words}")
    
    def test_ambiguous_tokenization_scenario(self):
        """
        Test a scenario where both single-word and multi-word interpretations are valid.
        
        "draw a light house" could mean:
        1. Draw a [light house] (lighthouse - a single building)
        2. Draw a [light] [house] (a house that is light in color/weight)
        
        For LATN, we should eventually return both interpretations.
        """
        sentence = "draw a light house at [0,0,0]"
        
        # TODO: This is what we want LATN lexical layer to return
        # expected_hypotheses = [
        #     # Hypothesis 1: Multi-word interpretation
        #     ["draw", "a", "light house", "at", "[0,0,0]"],
        #     # Hypothesis 2: Single-word interpretation  
        #     ["draw", "a", "light", "house", "at", "[0,0,0]"]
        # ]
        
        # For now, just document the current behavior
        tokens = tokenize(sentence)
        token_words = [token.word for token in tokens]
        
        print(f"Current single hypothesis: {token_words}")
        print("LATN goal: Return multiple ranked hypotheses for ambiguous cases")
        
        # Verify current behavior works
        self.assertTrue(len(tokens) > 0)
        self.assertEqual(tokens[0].word, "draw")
    
    def test_no_ambiguity_case(self):
        """
        Test that when there's no ambiguity, only one hypothesis is returned.
        """
        sentence = "draw a box at [1,2,3]"
        tokens = tokenize(sentence)
        token_words = [token.word for token in tokens]
        
        # This should be unambiguous
        expected = ["draw", "a", "box", "at", "[1,2,3]"]
        self.assertEqual(token_words, expected)
        
        print(f"Unambiguous tokenization: {token_words}")
    
    def test_three_word_compound_ambiguity(self):
        """
        Test potential three-word compound ambiguity.
        
        This sets up a case for testing when we implement multi-hypothesis parsing.
        """
        # Add a three-word compound for testing
        original_has_compound = 'very light house' in SEMANTIC_VECTOR_SPACE
        
        if not original_has_compound:
            SEMANTIC_VECTOR_SPACE['very light house'] = vector_from_features("noun adj")
        
        try:
            sentence = "draw a very light house"
            tokens = tokenize(sentence)
            token_words = [token.word for token in tokens]
            
            print(f"Three-word compound test: {token_words}")
            
            # Could be parsed as:
            # 1. "very light house" (three-word compound)
            # 2. "very" + "light house" (modifier + two-word compound)  
            # 3. "very" + "light" + "house" (three separate words)
            
            # Document what LATN should eventually handle
            self.assertTrue(len(tokens) > 0)
            
        finally:
            # Clean up
            if not original_has_compound and 'very light house' in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE['very light house']
    
    def test_latn_multi_hypothesis_tokenization(self):
        """
        Test LATN tokenizer returns multiple hypotheses for ambiguous cases.
        This demonstrates the core LATN Layer 1 functionality.
        """
        sentence = "draw a light house at [0,0,0]"
        hypotheses = latn_tokenize(sentence)
        
        # Should return multiple hypotheses
        self.assertGreater(len(hypotheses), 1, "LATN should return multiple hypotheses for ambiguous tokenization")
        
        # Extract token word lists for easier comparison
        hypothesis_tokens = []
        for hyp in hypotheses:
            self.assertIsInstance(hyp, TokenizationHypothesis)
            self.assertIsInstance(hyp.confidence, float)
            self.assertIsInstance(hyp.tokens, list)
            hypothesis_tokens.append([token.word for token in hyp.tokens])
        
        print(f"LATN hypotheses for '{sentence}':")
        for i, (hyp, tokens) in enumerate(zip(hypotheses, hypothesis_tokens), 1):
            print(f"  {i}. (conf={hyp.confidence:.2f}) {tokens}")
        
        # Verify we get both interpretations
        compound_interpretation = ['draw', 'a', 'light house', 'at', '[0,0,0]']
        separate_interpretation = ['draw', 'a', 'light', 'house', 'at', '[0,0,0]']
        
        self.assertIn(compound_interpretation, hypothesis_tokens, 
                     "Should include compound 'light house' interpretation")
        self.assertIn(separate_interpretation, hypothesis_tokens,
                     "Should include separate 'light' + 'house' interpretation")
        
        # Verify confidence ranking (compound should be preferred when it exists)
        best_hypothesis = hypotheses[0]
        best_tokens = [token.word for token in best_hypothesis.tokens]
        self.assertEqual(best_tokens, compound_interpretation,
                        "Compound interpretation should have highest confidence")
    
    def test_latn_no_ambiguity_single_hypothesis(self):
        """
        Test LATN tokenizer returns single hypothesis when no ambiguity exists.
        """
        sentence = "draw a box at [1,2,3]"
        hypotheses = latn_tokenize(sentence)
        
        # Should return only one hypothesis since no ambiguity
        self.assertEqual(len(hypotheses), 1, "Should return single hypothesis for unambiguous case")
        
        tokens = [token.word for token in hypotheses[0].tokens]
        expected = ["draw", "a", "box", "at", "[1,2,3]"]
        self.assertEqual(tokens, expected)
        
        print(f"LATN single hypothesis for '{sentence}': {tokens}")
    
    def test_latn_three_way_ambiguity(self):
        """
        Test LATN tokenizer with three-way ambiguity (single words, two-word, three-word compounds).
        """
        # Add the three-word compound for this test
        original_has_compound = 'very light house' in SEMANTIC_VECTOR_SPACE
        if not original_has_compound:
            SEMANTIC_VECTOR_SPACE['very light house'] = vector_from_features("noun adj")
        
        try:
            sentence = "draw a very light house"
            hypotheses = latn_tokenize(sentence)
            
            # Should return multiple hypotheses (3 possible interpretations)
            self.assertGreaterEqual(len(hypotheses), 3, "Should return at least 3 hypotheses for three-way ambiguity")
            
            hypothesis_tokens = [[token.word for token in hyp.tokens] for hyp in hypotheses]
            
            print(f"LATN three-way ambiguity for '{sentence}':")
            for i, (hyp, tokens) in enumerate(zip(hypotheses, hypothesis_tokens), 1):
                print(f"  {i}. (conf={hyp.confidence:.2f}) {tokens}")
            
            # Expected interpretations
            three_word_compound = ['draw', 'a', 'very light house']
            two_word_compound = ['draw', 'a', 'very', 'light house'] 
            all_separate = ['draw', 'a', 'very', 'light', 'house']
            
            # Verify all interpretations are present
            self.assertIn(three_word_compound, hypothesis_tokens,
                         "Should include three-word compound interpretation")
            self.assertIn(two_word_compound, hypothesis_tokens,
                         "Should include two-word compound interpretation")
            self.assertIn(all_separate, hypothesis_tokens,
                         "Should include all-separate-words interpretation")
            
        finally:
            # Clean up
            if not original_has_compound and 'very light house' in SEMANTIC_VECTOR_SPACE:
                del SEMANTIC_VECTOR_SPACE['very light house']
    
    def test_latn_confidence_ranking(self):
        """
        Test that LATN confidence scoring properly ranks hypotheses.
        """
        sentence = "draw a light house at [0,0,0]"
        hypotheses = latn_tokenize(sentence)
        
        # Confidences should be in descending order
        confidences = [hyp.confidence for hyp in hypotheses]
        self.assertEqual(confidences, sorted(confidences, reverse=True),
                        "Hypotheses should be ranked by confidence (highest first)")
        
        # Compound should have higher confidence than separate words
        compound_hyp = next(hyp for hyp in hypotheses 
                           if [t.word for t in hyp.tokens] == ['draw', 'a', 'light house', 'at', '[0,0,0]'])
        separate_hyp = next(hyp for hyp in hypotheses 
                           if [t.word for t in hyp.tokens] == ['draw', 'a', 'light', 'house', 'at', '[0,0,0]'])
        
        self.assertGreater(compound_hyp.confidence, separate_hyp.confidence,
                          "Compound interpretation should have higher confidence than separate words")
        
        print(f"Confidence ranking:")
        print(f"  Compound: {compound_hyp.confidence:.2f}")
        print(f"  Separate: {separate_hyp.confidence:.2f}")


if __name__ == '__main__':
    unittest.main()
