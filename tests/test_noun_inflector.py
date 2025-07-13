import pytest
from engraf.utils.noun_inflector import singularize_noun, is_plural, analyze_word

# Test singularize_noun
@pytest.mark.parametrize("word,expected_singular,is_singularized", [
    ("cats", "cat", True),
    ("buses", "bus", True),
    ("babies", "baby", True),
    ("wolves", "wolf", True),
    ("children", "child", True),
    ("sheep", "sheep", False),
    ("glass", "glass", False),
    ("dog", "dog", False),
])
def test_singularize_noun(word, expected_singular, is_singularized):
    singular, flag = singularize_noun(word)
    assert singular == expected_singular
    assert flag == is_singularized

# Test is_plural
@pytest.mark.parametrize("word,expected", [
    ("cats", True),
    ("dog", False),
    ("wolves", True),
    ("child", False),
    ("children", True),
    ("glass", False),
])
def test_is_plural(word, expected):
    assert is_plural(word) == expected

# Test analyze_word
@pytest.mark.parametrize("word,expected_features", [
    ("cats", ["noun", "plural"]),
    ("dog", ["noun", "singular"]),
    ("wolves", ["noun", "plural"]),
    ("child", ["noun", "singular"]),
    ("children", ["noun", "plural"]),
])
def test_analyze_word(word, expected_features):
    vec = analyze_word(word)
    for feature in expected_features:
        assert vec[feature] > 0.0

# Additional edge case tests for irregular plurals
@pytest.mark.parametrize("word,expected_singular,is_singularized", [
    ("men", "man", True),
    ("women", "woman", True),
    ("geese", "goose", True),
    ("mice", "mouse", True),
    ("feet", "foot", True),
    ("teeth", "tooth", True),
])
def test_irregular_plurals(word, expected_singular, is_singularized):
    singular, flag = singularize_noun(word)
    assert singular == expected_singular
    assert flag == is_singularized

# Test case sensitivity
@pytest.mark.parametrize("word,expected_singular,is_singularized", [
    ("CATS", "cat", True),
    ("Wolves", "wolf", True),
    ("CHILDREN", "child", True),
    ("Dogs", "dog", True),
    ("GLASS", "GLASS", False),
])
def test_case_sensitivity(word, expected_singular, is_singularized):
    singular, flag = singularize_noun(word)
    assert singular == expected_singular
    assert flag == is_singularized

# Test words that look plural but aren't (false positives)
@pytest.mark.parametrize("word,expected", [
    ("glass", False),    # ends in 's' but not plural
    ("dress", False),    # ends in 'ss' so not plural
    ("mess", False),     # ends in 'ss' so not plural
    ("pass", False),     # ends in 'ss' so not plural
    ("grass", False),    # ends in 'ss' so not plural
    ("news", True),      # ends in 's' - current algorithm treats as plural
    ("lens", True),      # ends in 's' - current algorithm treats as plural
    ("bus", True),       # ends in 's' - current algorithm treats as plural
])
def test_false_positive_plurals(word, expected):
    assert is_plural(word) == expected

# Test complex pluralization patterns
@pytest.mark.parametrize("word,expected_singular,is_singularized", [
    ("boxes", "boxe", True),     # -es ending
    ("foxes", "foxe", True),     # -es ending  
    ("dishes", "dishe", True),   # -es ending
    ("matches", "matche", True), # -es ending
    ("stories", "story", True),  # -ies ending
    ("cities", "city", True),    # -ies ending
    ("knives", "knif", True),    # -ves ending
    ("leaves", "leaf", True),    # -ves ending
    ("shelves", "shelf", True),  # -ves ending
])
def test_complex_pluralization_patterns(word, expected_singular, is_singularized):
    singular, flag = singularize_noun(word)
    assert singular == expected_singular
    assert flag == is_singularized

# Test words from the semantic vector space
@pytest.mark.parametrize("word,expected_features", [
    ("cubes", ["noun", "plural"]),      # cube is in semantic space
    ("spheres", ["noun", "plural"]),    # sphere is in semantic space
    ("pyramids", ["noun", "plural"]),   # pyramid is in semantic space
    ("cube", ["noun"]),                 # should use semantic space directly (only has noun, not singular)
    ("sphere", ["noun"]),               # should use semantic space directly (only has noun, not singular)
])
def test_analyze_word_with_semantic_space(word, expected_features):
    vec = analyze_word(word)
    for feature in expected_features:
        assert vec[feature] > 0.0

# Test empty and edge case inputs
@pytest.mark.parametrize("word,expected_singular,is_singularized", [
    ("", "", False),          # empty string
    ("a", "a", False),        # single character
    ("s", "s", False),        # just 's'
    ("es", "e", True),        # just 'es' - treated as plural
])
def test_edge_case_inputs(word, expected_singular, is_singularized):
    singular, flag = singularize_noun(word)
    assert singular == expected_singular
    assert flag == is_singularized

# Test that analyze_word handles unknown words gracefully
def test_analyze_word_unknown_singular():
    """Test that analyze_word assigns 'noun singular' to unknown words"""
    vec = analyze_word("unknownword")
    assert vec["noun"] > 0.0
    assert vec["singular"] > 0.0

def test_analyze_word_unknown_plural():
    """Test that analyze_word assigns 'noun plural' to unknown plural words"""
    vec = analyze_word("unknownwords")  # looks plural
    assert vec["noun"] > 0.0
    assert vec["plural"] > 0.0

# Test consistency between is_plural and singularize_noun
@pytest.mark.parametrize("word", [
    "cats", "dogs", "boxes", "children", "wolves", "babies", 
    "men", "women", "geese", "mice", "feet", "teeth",
    "glass", "dress", "news", "sheep"
])
def test_consistency_between_functions(word):
    """Test that is_plural and singularize_noun are consistent"""
    singular, was_plural = singularize_noun(word)
    is_plural_result = is_plural(word)
    assert was_plural == is_plural_result
