#!/usr/bin/env python3
"""
Test script to check parsing of "make them more transparent than the purple circle at [3, 3, 3]"
"""

from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence

def test_transparent_sentence():
    """Test parsing 'make them more transparent than the purple circle at [3, 3, 3]'"""
    sentence = "make them more transparent than the purple circle at [3, 3, 3]"
    
    print(f"Testing sentence: '{sentence}'")
    
    try:
        tokens = TokenStream(tokenize(sentence))
        print(f"Tokens: {[token.word for token in tokens]}")
        
        result = run_sentence(tokens)
        print(f"Parse result: {result}")
        
        if result is not None:
            print(f"✅ SUCCESS: Parsed successfully")
            print(f"    Result: {result}")
        else:
            print(f"❌ FAILED: Parser returned None")
            
    except Exception as e:
        print(f"💥 ERROR: Exception during parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transparent_sentence()
