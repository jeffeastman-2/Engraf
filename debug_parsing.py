#!/usr/bin/env python3

from engraf.lexer.vocabulary_builder import vector_from_word, base_adjective_from_comparative
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.sentence import build_sentence_atn
from engraf.atn.core import run_atn
from engraf.pos.sentence_phrase import SentencePhrase

def test_tokens():
    """Test individual token parsing"""
    test_words = ["color", "it", "a", "little", "bit", "redder"]
    
    for word in test_words:
        try:
            vector = vector_from_word(word)
            # Find active POS tags
            active_tags = [tag for tag in ["verb", "tobe", "action", "prep", "det", "def", "adv", "adj", "noun", "pronoun", "conj", "disj", "unit", "comp"] if vector[tag] > 0]
            print(f"'{word}' -> {active_tags} (found)")
        except ValueError as e:
            print(f"'{word}' -> {e}")

def test_comparative():
    """Test comparative adjective handling"""
    test_comparatives = ["redder", "bigger", "smaller", "taller"]
    
    for word in test_comparatives:
        base, is_comp = base_adjective_from_comparative(word)
        print(f"'{word}' -> base: '{base}', is_comparative: {is_comp}")

def test_sentence_parsing():
    """Test the full sentence parsing"""
    sentence = "color it a little bit redder"
    print(f"\nTesting sentence: '{sentence}'")
    
    try:
        tokens = tokenize(sentence)
        print(f"Tokens: {[token.word for token in tokens]}")
        
        ts = TokenStream(tokens)
        sent = SentencePhrase()
        start, end = build_sentence_atn(sent, ts)
        result = run_atn(start, end, ts, sent)
        print(f"Parse result: {result}")
        print(f"Sentence phrase: {sent}")
        
    except Exception as e:
        print(f"Parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Testing individual tokens ===")
    test_tokens()
    
    print("\n=== Testing comparative adjectives ===")
    test_comparative()
    
    print("\n=== Testing sentence parsing ===")
    test_sentence_parsing()
