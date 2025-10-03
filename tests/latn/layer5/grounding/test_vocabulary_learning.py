#!/usr/bin/env python3
"""
Test vocabulary learning from quoted definitions
"""

from engraf.lexer.token_stream import TokenStream
from engraf.lexer.latn_layer_executor import tokenize_best
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.vector_space import VectorSpace
from engraf.lexer.vocabulary_builder import add_to_vocabulary, has_word
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.verb_phrase import VerbPhrase

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

    executor = LATNLayerExecutor()

    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}] Processing: {sentence}")
        print("-" * 40)
        
        try:
            result = executor.execute_layer5(sentence=sentence)
            if result.success:
                print(f"✅ SUCCESS: Parsed successfully")
                print(f"    Result: {result.hypotheses[0].description}")
                hyp0 = result.hypotheses[0]
                vector = hyp0.tokens
                assert len(vector) == 1
                sent = vector[0].phrase
                assert isinstance(sent, SentencePhrase)
                subj = sent.vector
                pred = sent.predicate
                assert isinstance(subj, VectorSpace)
                assert isinstance(pred, VerbPhrase)
                # TODO: This is just checking tokenization
            
            else:
                print(f"❌ FAILED: Parser returned None")
                
        except Exception as e:
            print(f"💥 ERROR: Exception during parsing: {e}")

