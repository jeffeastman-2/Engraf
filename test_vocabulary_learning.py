#!/usr/bin/env python3
"""
Test vocabulary learning from quoted definitions
"""

from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_sentence import run_sentence
from engraf.lexer.vocabulary import add_to_vocabulary, has_word

def test_vocabulary_learning_sequence():
    """Test that vocabulary is learned from quoted definitions and used in subsequent sentences"""
    
    # Test the sequence of 3 sentences
    sentences = [
        "'huge' is very large",
        "'sky blue' is blue and green", 
        "draw a huge sky blue box"
    ]
    
    print("Testing vocabulary learning sequence:")
    print("=" * 50)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}] Processing: {sentence}")
        print("-" * 40)
        
        try:
            tokens = TokenStream(tokenize(sentence))
            result = run_sentence(tokens)
            
            if result is not None:
                print(f"✅ SUCCESS: Parsed successfully")
                print(f"    Result: {result}")
                
                # Check if this sentence defined a new word
                if result.definition_word:
                    print(f"🔍 Found definition word: '{result.definition_word}'")
                    print(f"🔍 Definition vector: {result.vector}")
                    
                    # Extract the meaning from the sentence and add to vocabulary
                    if result.predicate and hasattr(result.predicate, 'vector'):
                        add_to_vocabulary(result.definition_word, result.predicate.vector)
                        print(f"✅ Added '{result.definition_word}' to vocabulary")
                    elif result.vector:
                        add_to_vocabulary(result.definition_word, result.vector)
                        print(f"✅ Added '{result.definition_word}' to vocabulary")
                
            else:
                print(f"❌ FAILED: Parser returned None")
                
        except Exception as e:
            print(f"💥 ERROR: Exception during parsing: {e}")
    
    # Check if vocabulary was learned
    print(f"\n🔍 Vocabulary check:")
    print(f"  'huge' in vocabulary: {has_word('huge')}")
    print(f"  'sky blue' in vocabulary: {has_word('sky blue')}")
    print(f"  'sky' in vocabulary: {has_word('sky')}")

if __name__ == "__main__":
    test_vocabulary_learning_sequence()
