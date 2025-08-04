"""
Tests for grammatical feature vector dimensions.

This module tests the grammatical feature dimensions: "disj", "neg", "modal", 
and "question" that were added to distinguish different types of grammatical
words in the semantic vector space.
"""

import pytest
from engraf.lexer.vocabulary import vector_from_word
from engraf.lexer.vector_space import VectorSpace, VECTOR_DIMENSIONS


class TestDisjunctionFeature:
    """Test the 'disj' dimension for disjunctive words (or)."""
    
    def test_disjunction_words_set_disj_dimension(self):
        """Test that disjunctive words have disj=1.0."""
        disjunction_words = ["or"]
        
        for word in disjunction_words:
            vector = vector_from_word(word)
            assert vector["disj"] == 1.0, f"{word} should have disj=1.0, got {vector['disj']}"
            assert vector.isa("disj"), f"{word} should be recognized as disjunction"
    
    def test_or_is_disjunction(self):
        """Test that 'or' is specifically marked as disjunction."""
        vector = vector_from_word("or")
        assert vector["disj"] == 1.0
        assert vector["conj"] == 0.0  # Should not be conjunction
        assert vector.word == "or"


class TestNegationFeature:
    """Test the 'neg' dimension for negation words (not, no)."""
    
    def test_negation_words_set_neg_dimension(self):
        """Test that negation words have neg=1.0."""
        negation_words = ["not", "no"]
        
        for word in negation_words:
            vector = vector_from_word(word)
            assert vector["neg"] == 1.0, f"{word} should have neg=1.0, got {vector['neg']}"
            assert vector.isa("neg"), f"{word} should be recognized as negation"
    
    def test_not_is_negation(self):
        """Test that 'not' is specifically marked as negation."""
        vector = vector_from_word("not")
        assert vector["neg"] == 1.0
        assert vector.word == "not"
    
    def test_no_is_negation(self):
        """Test that 'no' is specifically marked as negation."""
        vector = vector_from_word("no")
        assert vector["neg"] == 1.0
        assert vector.word == "no"


class TestModalFeature:
    """Test the 'modal' dimension for modal verbs."""
    
    def test_modal_verbs_set_modal_dimension(self):
        """Test that modal verbs have modal=1.0 and verb=1.0."""
        modal_verbs = ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]
        
        for verb in modal_verbs:
            vector = vector_from_word(verb)
            assert vector["modal"] == 1.0, f"{verb} should have modal=1.0, got {vector['modal']}"
            assert vector["verb"] == 1.0, f"{verb} should have verb=1.0, got {vector['verb']}"
            assert vector.isa("modal"), f"{verb} should be recognized as modal verb"
            assert vector.isa("verb"), f"{verb} should be recognized as verb"
    
    def test_individual_modal_verbs(self):
        """Test each modal verb individually."""
        # Test auxiliary modal verbs
        can_vector = vector_from_word("can")
        assert can_vector["modal"] == 1.0
        assert can_vector["verb"] == 1.0
        assert can_vector.word == "can"
        
        could_vector = vector_from_word("could")
        assert could_vector["modal"] == 1.0
        assert could_vector["verb"] == 1.0
        assert could_vector.word == "could"
        
        # Test permission/possibility modals
        may_vector = vector_from_word("may")
        assert may_vector["modal"] == 1.0
        assert may_vector["verb"] == 1.0
        assert may_vector.word == "may"
        
        might_vector = vector_from_word("might")
        assert might_vector["modal"] == 1.0
        assert might_vector["verb"] == 1.0
        assert might_vector.word == "might"
        
        # Test obligation modal
        must_vector = vector_from_word("must")
        assert must_vector["modal"] == 1.0
        assert must_vector["verb"] == 1.0
        assert must_vector.word == "must"
        
        # Test formal modals
        shall_vector = vector_from_word("shall")
        assert shall_vector["modal"] == 1.0
        assert shall_vector["verb"] == 1.0
        assert shall_vector.word == "shall"
        
        should_vector = vector_from_word("should")
        assert should_vector["modal"] == 1.0
        assert should_vector["verb"] == 1.0
        assert should_vector.word == "should"
        
        # Test future modals
        will_vector = vector_from_word("will")
        assert will_vector["modal"] == 1.0
        assert will_vector["verb"] == 1.0
        assert will_vector.word == "will"
        
        would_vector = vector_from_word("would")
        assert would_vector["modal"] == 1.0
        assert would_vector["verb"] == 1.0
        assert would_vector.word == "would"


class TestQuestionFeature:
    """Test the 'question' dimension for question words."""
    
    def test_question_words_set_question_dimension(self):
        """Test that question words have question=1.0."""
        question_words = ["who", "what", "where", "when", "why", "how", "which"]
        
        for word in question_words:
            vector = vector_from_word(word)
            assert vector["question"] == 1.0, f"{word} should have question=1.0, got {vector['question']}"
            assert vector.isa("question"), f"{word} should be recognized as question word"
    
    def test_individual_question_words(self):
        """Test each question word individually."""
        # Test person question
        who_vector = vector_from_word("who")
        assert who_vector["question"] == 1.0
        assert who_vector.word == "who"
        
        # Test thing/object question
        what_vector = vector_from_word("what")
        assert what_vector["question"] == 1.0
        assert what_vector.word == "what"
        
        # Test location question
        where_vector = vector_from_word("where")
        assert where_vector["question"] == 1.0
        assert where_vector.word == "where"
        
        # Test time question
        when_vector = vector_from_word("when")
        assert when_vector["question"] == 1.0
        assert when_vector.word == "when"
        
        # Test reason question
        why_vector = vector_from_word("why")
        assert why_vector["question"] == 1.0
        assert why_vector.word == "why"
        
        # Test manner question
        how_vector = vector_from_word("how")
        assert how_vector["question"] == 1.0
        assert how_vector.word == "how"
        
        # Test choice question
        which_vector = vector_from_word("which")
        assert which_vector["question"] == 1.0
        assert which_vector.word == "which"


class TestConjunctionStillWorks:
    """Test that existing conjunction functionality still works."""
    
    def test_and_is_conjunction(self):
        """Test that 'and' is still properly marked as conjunction."""
        vector = vector_from_word("and")
        assert vector["conj"] == 1.0, "'and' should have conj=1.0"
        assert vector["disj"] == 0.0, "'and' should not be disjunction"
        assert vector.isa("conj"), "'and' should be recognized as conjunction"
        assert vector.word == "and"


class TestGrammaticalFeatureDimensionsExist:
    """Test that all grammatical feature dimensions exist in the vector space."""
    
    def test_grammatical_features_in_vector_dimensions(self):
        """Test that all grammatical feature dimensions are in VECTOR_DIMENSIONS."""
        required_dimensions = ["disj", "neg", "modal", "question", "conj"]
        
        for dim in required_dimensions:
            assert dim in VECTOR_DIMENSIONS, f"'{dim}' dimension should exist in VECTOR_DIMENSIONS"
    
    def test_grammatical_features_accessible_in_vector_space(self):
        """Test that all grammatical features are accessible in VectorSpace objects."""
        vs = VectorSpace()
        
        # Should be able to get and set all grammatical feature values
        grammatical_features = ["disj", "neg", "modal", "question", "conj"]
        
        for feature in grammatical_features:
            vs[feature] = 1.0
            assert vs[feature] == 1.0, f"Should be able to set and get {feature} dimension"
        
        # Test with fresh vector space (should all be 0.0 by default)
        vs2 = VectorSpace()
        for feature in grammatical_features:
            assert vs2[feature] == 0.0, f"New VectorSpace should have {feature}=0.0 by default"


class TestMutualExclusivity:
    """Test that certain grammatical features are mutually exclusive where appropriate."""
    
    def test_conjunction_vs_disjunction(self):
        """Test that words are either conjunctions or disjunctions, not both."""
        and_vector = vector_from_word("and")
        assert and_vector["conj"] == 1.0
        assert and_vector["disj"] == 0.0
        
        or_vector = vector_from_word("or")
        assert or_vector["disj"] == 1.0
        assert or_vector["conj"] == 0.0
    
    def test_modal_vs_other_features(self):
        """Test that modal verbs have verb=1.0 but don't get other grammatical features."""
        modal_verbs = ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]
        
        for verb in modal_verbs:
            vector = vector_from_word(verb)
            assert vector["modal"] == 1.0, f"{verb} should be modal"
            assert vector["verb"] == 1.0, f"{verb} should be verb"
            assert vector["neg"] == 0.0, f"{verb} should not be negation"
            assert vector["question"] == 0.0, f"{verb} should not be question"
            assert vector["conj"] == 0.0, f"{verb} should not be conjunction"
            assert vector["disj"] == 0.0, f"{verb} should not be disjunction"


class TestCaseSensitivity:
    """Test that grammatical feature words work regardless of case."""
    
    def test_case_insensitive_access(self):
        """Test that vector_from_word handles case insensitively for grammatical features."""
        # Test modal verbs
        can_lower = vector_from_word("can")
        can_upper = vector_from_word("CAN")
        assert can_lower["modal"] == can_upper["modal"] == 1.0
        
        # Test question words
        what_lower = vector_from_word("what")
        what_upper = vector_from_word("WHAT")
        assert what_lower["question"] == what_upper["question"] == 1.0
        
        # Test negation
        not_lower = vector_from_word("not")
        not_upper = vector_from_word("NOT")
        assert not_lower["neg"] == not_upper["neg"] == 1.0


class TestIntegrationWithExistingFeatures:
    """Test that new grammatical features integrate properly with existing functionality."""
    
    def test_grammatical_features_with_isa_method(self):
        """Test that the isa() method works with new grammatical features."""
        # Test modal
        can_vector = vector_from_word("can")
        assert can_vector.isa("modal")
        assert can_vector.isa("verb")
        assert not can_vector.isa("neg")
        assert not can_vector.isa("question")
        
        # Test question
        what_vector = vector_from_word("what")
        assert what_vector.isa("question")
        assert not what_vector.isa("modal")
        assert not what_vector.isa("neg")
        assert not what_vector.isa("verb")
        
        # Test negation
        not_vector = vector_from_word("not")
        assert not_vector.isa("neg")
        assert not not_vector.isa("modal")
        assert not not_vector.isa("question")
        assert not not_vector.isa("verb")
    
    def test_vector_operations_preserve_grammatical_features(self):
        """Test that vector operations preserve grammatical feature dimensions."""
        modal_vector = vector_from_word("can")
        question_vector = vector_from_word("what")
        
        # Test addition
        combined = modal_vector + question_vector
        assert combined["modal"] == 1.0, "Addition should preserve modal dimension"
        assert combined["question"] == 1.0, "Addition should preserve question dimension"
        
        # Test scalar multiplication
        scaled_modal = modal_vector * 2.0
        assert scaled_modal["modal"] == 2.0, "Multiplication should scale modal dimension"
    
    def test_copy_preserves_grammatical_features(self):
        """Test that copying vectors preserves grammatical features."""
        original = vector_from_word("should")
        copied = original.copy()
        
        assert copied["modal"] == original["modal"]
        assert copied.word == original.word
        assert copied.isa("modal")


class TestEdgeCases:
    """Test edge cases and error conditions for grammatical features."""
    
    def test_unknown_grammatical_words(self):
        """Test that unknown words raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown token"):
            vector_from_word("nonexistentmodal")
        
        with pytest.raises(ValueError, match="Unknown token"):
            vector_from_word("fakequestion")
    
    def test_empty_string_handling(self):
        """Test handling of empty strings for grammatical features."""
        with pytest.raises(ValueError):
            vector_from_word("")
    
    def test_none_handling(self):
        """Test handling of None input."""
        with pytest.raises((TypeError, AttributeError)):
            vector_from_word(None)
