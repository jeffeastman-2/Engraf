#!/usr/bin/env python3
"""
Demo: LATN Layer 1 - Multi-Hypothesis Tokenization

This demo showcases Layer 1 of the LATN (Layered Augmented Transition Network) system,
which generates multiple tokenization hypotheses for ambiguous phrases.

Layer 1 demonstrates:
- Multi-word compound detection (e.g., "coffee table" vs "coffee" + "table")
- Alternative tokenization strategies
- Confidence scoring and hypothesis ranking
- Interactive vocabulary expansion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.latn_tokenizer_layer1 import latn_tokenize
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features


def demo_basic_tokenization():
    """Demonstrate basic Layer 1 tokenization with existing vocabulary."""
    print("🔤 LATN Layer 1: Multi-Hypothesis Tokenization Demo")
    print("=" * 60)
    
    # Test cases that show different tokenization scenarios
    test_phrases = [
        "red box",
        "big red sphere", 
        "very large cube",
        "unknown phrase",
        "the table",
        "[1,2,3]"
    ]
    
    executor = LATNLayerExecutor()
    
    for phrase in test_phrases:
        print(f"\n📝 Input: \"{phrase}\"")
        print("-" * 30)
        
        result = executor.execute_layer1(phrase)
        
        if result.success:
            print(f"✅ Generated {len(result.hypotheses)} tokenization hypothesis(es)")
            
            for i, hyp in enumerate(result.hypotheses[:3], 1):  # Show top 3
                tokens = [t.word for t in hyp.tokens]
                print(f"  {i}. {tokens}")
                print(f"     Confidence: {hyp.confidence:.3f}")
                print(f"     Description: {hyp.description}")
                
                # Print all tokens using the new method
                hyp.print_tokens()
                
                # Show token details
                for j, token in enumerate(hyp.tokens):
                    # Get vector features (VectorSpace doesn't have items() method)
                    from engraf.An_N_Space_Model.vector_dimensions import VECTOR_DIMENSIONS
                    features = []
                    for dim in VECTOR_DIMENSIONS[:10]:  # Check first 10 dimensions
                        if hasattr(token, dim):
                            value = getattr(token, dim, 0)
                            if value > 0:
                                features.append(f"{dim}={value:.2f}")
                    
                    if not features:
                        features = ["no-active-features"]
                    
                    print(f"       Token {j+1}: '{token.word}' → {features[:3]}")
            
            if len(result.hypotheses) > 3:
                print(f"     ... and {len(result.hypotheses) - 3} more hypotheses")
        else:
            print("❌ Tokenization failed")


def add_multi_word_compounds():
    """Add some multi-word compounds to the vocabulary to demonstrate multi-hypothesis generation."""
    print("\n🔧 Adding Multi-Word Compounds to Vocabulary")
    print("=" * 50)
    
    # Add some multi-word compounds using existing vocabulary words
    # This creates ambiguity: should "blue green" be one token or "blue" + "green"?
    multi_word_compounds = {
        # Color compounds (using existing colors)
        "blue green": vector_from_features("adj", blue=0.5, green=0.5),  # teal/cyan
        "red blue": vector_from_features("adj", red=0.6, blue=0.4),     # purple
        "green red": vector_from_features("adj", green=0.4, red=0.6),   # brown/orange
        "sky blue": vector_from_features("adj", green=0.5, blue=0.6),  # teal/cyan

        # Add some new nouns and compound objects
        "house": vector_from_features("noun", scaleX=2.0, scaleY=1.5, scaleZ=2.0),
        "car": vector_from_features("noun", scaleX=1.8, scaleY=0.8, scaleZ=0.6),
        "sky": vector_from_features("noun"),
        
        # Compound objects using new + existing words
        "green house": vector_from_features("noun", green=0.3, scaleX=3.0, scaleY=2.0),  # greenhouse
        "red car": vector_from_features("noun", red=0.8, scaleX=1.8, scaleY=0.8),       # red car
        "blue box": vector_from_features("noun", blue=0.7, scaleX=1.0, scaleY=1.0),     # blue box
        
        # Size compounds using existing adjectives
        "large table": vector_from_features("noun", scaleX=2.5, scaleY=1.0, scaleZ=2.5),
        "small cube": vector_from_features("noun", scaleX=0.5, scaleY=0.5, scaleZ=0.5),
    }
    
    # Add to global vocabulary
    SEMANTIC_VECTOR_SPACE.update(multi_word_compounds)
    
    print("Added multi-word compounds:")
    for compound in multi_word_compounds:
        print(f"  • \"{compound}\"")
    
    return list(multi_word_compounds.keys())


def demo_multi_hypothesis_generation(compounds):
    """Demonstrate how multi-word compounds create multiple tokenization hypotheses."""
    print("\n🎭 Multi-Hypothesis Generation Demo")
    print("=" * 40)
    
    # Test phrases that should now generate multiple hypotheses
    ambiguous_phrases = [
        "blue green box",      # Could be compound "blue green" + "box" or "blue" + "green" + "box"
        "green house",         # Could be compound "green house" or "green" + "house"  
        "red car",             # Could be compound "red car" or "red" + "car"
        "large table",         # Could be compound "large table" or "large" + "table"
        "small cube",          # Could be compound "small cube" or "small" + "cube"
        "blue box on table",   # Could be "blue box" + "on" + "table" or "blue" + "box" + "on" + "table"
    ]
    
    for phrase in ambiguous_phrases:
        print(f"\n🔍 Analyzing: \"{phrase}\"")
        print("-" * 35)
        
        hypotheses = latn_tokenize(phrase)
        
        if len(hypotheses) > 1:
            print(f"🎉 Generated {len(hypotheses)} different tokenization hypotheses!")
        else:
            print(f"📝 Generated {len(hypotheses)} tokenization hypothesis")
        
        for i, hyp in enumerate(hypotheses, 1):
            tokens = [t.word for t in hyp.tokens]
            print(f"\n  Hypothesis {i}: {tokens}")
            print(f"    Confidence: {hyp.confidence:.3f}")
            print(f"    Strategy: {hyp.description}")
            
            # Print all tokens using the new method
            hyp.print_tokens()
            
            # Show what makes this hypothesis unique
            compound_tokens = [t.word for t in hyp.tokens if ' ' in t.word]
            if compound_tokens:
                print(f"    Uses compounds: {compound_tokens}")
        
        # Show which hypothesis won and why
        if hypotheses:
            best = hypotheses[0]
            print(f"\n  🏆 Best hypothesis: {[t.word for t in best.tokens]}")
            print(f"    Reason: Highest confidence ({best.confidence:.3f})")


def main():
    """Run the complete Layer 1 demo."""
    print("🚀 Welcome to the LATN Layer 1 Tokenization Demo!")
    print("\nThis demo shows how Layer 1 generates multiple tokenization hypotheses")
    print("for ambiguous phrases, especially when multi-word compounds are involved.\n")
    
    # Part 1: Basic tokenization with current vocabulary
    demo_basic_tokenization()
    
    # Part 2: Add multi-word compounds to create ambiguity
    compounds = add_multi_word_compounds()
    
    # Part 3: Show multi-hypothesis generation
    demo_multi_hypothesis_generation(compounds)
    
    print("\n🎉 Demo Complete!")
    print("\nKey takeaways:")
    print("• Layer 1 creates multiple tokenization hypotheses for ambiguous phrases")
    print("• Multi-word compounds compete with single-word alternatives")
    print("• Confidence scores help rank competing hypotheses")
    print("• The vocabulary can be expanded dynamically for testing")
    print("\nNext: Try demo_layer2_noun_phrases.py to see how Layer 2 processes these tokens!")


if __name__ == "__main__":
    main()
