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
