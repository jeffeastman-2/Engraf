#!/usr/bin/env python3
"""
Demo: LATN Layer 2 - Noun Phrase Formation

This demo showcases Layer 2 of the LATN (Layered Augmented Transition Network) system,
which processes tokens from Layer 1 and forms noun phrases with proper grammatical structure.

Layer 2 demonstrates:
- Noun phrase formation from individual tokens
- Adjective-noun combinations with proper ordering
- Determiner handling and agreement
- Multi-hypothesis noun phrase alternatives
- Grammatical validation and confidence scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.utils.debug import set_debug
from engraf.An_N_Space_Model.vocabulary import SEMANTIC_VECTOR_SPACE
from engraf.lexer.vector_space import vector_from_features


def demo_basic_noun_phrase_formation():
    """Demonstrate how Layer 2 forms noun phrases from Layer 1 tokens."""
    print("🔤 LATN Layer 2: Noun Phrase Formation Demo")
    print("=" * 55)
    
    executor = LATNLayerExecutor()
    
    # Test cases showing different noun phrase patterns
    test_phrases = [
        "red box",              # adj + noun
        "the large sphere",     # det + adj + noun  
        "a very big cube",      # det + adv + adj + noun
        "small green table",    # adj + adj + noun
        "the red box",          # det + adj + noun
        "blue object",          # adj + noun
        "large spheres",        # adj + noun (plural)
        "the object",           # det + noun
    ]
    
    for phrase in test_phrases:
        print(f"\n📝 Input: \"{phrase}\"")
        print("-" * 35)
        
        # Use Layer 2 executor with semantic grounding disabled (tokenization only)
        result = executor.execute_layer2(phrase, enable_semantic_grounding=False)
        
        if result.success:
            print(f"✅ Generated {len(result.hypotheses)} noun phrase hypothesis(es)")
            
            for i, hyp in enumerate(result.hypotheses[:3], 1):
                # Show the token structure
                token_words = [t.word for t in hyp.tokens]
                print(f"  {i}. Tokens: {token_words}")
                print(f"     Confidence: {hyp.confidence:.3f}")
                
                # Show noun phrase replacements
                if hyp.np_replacements:
                    print(f"     NP Replacements: {len(hyp.np_replacements)}")
                    for j, np_replacement in enumerate(hyp.np_replacements, 1):
                        print(f"       {j}. {np_replacement}")
                
                # Show any NP tokens found
                np_tokens = [tok for tok in hyp.tokens if tok.isa("NP")]
                if np_tokens:
                    print(f"     NP Tokens: {[tok.word for tok in np_tokens]}")
        else:
            print("❌ Layer 2 processing failed")


def demo_adjective_ordering():
    """Demonstrate how Layer 2 handles adjective ordering in noun phrases."""
    print("\n🎨 Adjective Ordering and Agreement Demo")
    print("=" * 45)
    
    executor = LATNLayerExecutor()
    
    # Test cases with different adjective patterns
    ordering_examples = [
        "big red sphere",           # size + color + noun
        "the small blue cube",      # det + size + color + noun
        "very large green box",     # adv + size + color + noun
        "red blue object",          # color + color + noun (compound color)
        "smooth red cube",          # texture + color + noun
        "small sphere",             # size + noun
    ]
    
    for phrase in ordering_examples:
        print(f"\n🔍 Analyzing: \"{phrase}\"")
        print("-" * 25)
        
        result = executor.execute_layer2(phrase, enable_semantic_grounding=False)
        
        if result.success:
            best_hyp = result.hypotheses[0]
            token_words = [t.word for t in best_hyp.tokens]
            print(f"Tokens: {token_words}")
            
            # Show noun phrase structure
            if best_hyp.np_replacements:
                for j, np_replacement in enumerate(best_hyp.np_replacements, 1):
                    print(f"NP {j}: {np_replacement}")
                    
                    # Show adjective ordering if available
                    if hasattr(np_replacement, 'adjectives') and np_replacement.adjectives:
                        adj_order = [adj.word for adj in np_replacement.adjectives]
                        print(f"Adjective order: {adj_order}")
        else:
            print("❌ Processing failed")


def demo_determiner_handling():
    """Demonstrate determiner agreement and handling."""
    print("\n🏷️  Determiner Handling Demo")
    print("=" * 35)
    
    executor = LATNLayerExecutor()
    
    determiner_examples = [
        "a box",                    # indefinite article + singular
        "the boxes",                # definite article + plural
        "some spheres",             # quantifier + plural
        "each cube",                # distributive + singular
        "these objects",            # demonstrative + plural
        "that table",               # demonstrative + singular
        "many cubes",               # quantifier + plural
        "two spheres",              # numeral + plural
    ]
    
    for phrase in determiner_examples:
        print(f"\n🏷️  Testing: \"{phrase}\"")
        print("-" * 20)
        
        result = executor.execute_layer2(phrase, enable_semantic_grounding=False)
        
        if result.success:
            best_hyp = result.hypotheses[0]
            print(f"Result: {[t.word for t in best_hyp.tokens]}")
            
            # Show determiner analysis
            if best_hyp.np_replacements:
                for j, np_replacement in enumerate(best_hyp.np_replacements, 1):
                    print(f"NP {j}: {np_replacement}")
                    
                    # Show determiner information
                    if hasattr(np_replacement, 'determiner') and np_replacement.determiner:
                        det = np_replacement.determiner
                        print(f"Determiner: '{det.word}' (type: {det.determiner_type if hasattr(det, 'determiner_type') else 'unknown'})")
                    
                    # Show number agreement
                    if hasattr(np_replacement, 'number'):
                        print(f"Number: {np_replacement.number}")
        else:
            print("❌ Processing failed")


def demo_multi_hypothesis_noun_phrases():
    """Demonstrate cases where multiple noun phrase interpretations are possible."""
    print("\n🔀 Multi-Hypothesis Noun Phrase Demo")
    print("=" * 42)
    
    executor = LATNLayerExecutor()
    
    ambiguous_examples = [
        "red green box",            # Could be compound color or separate adjectives
        "very small large cube",    # Conflicting size adjectives
        "the big small object",     # Another conflicting case
        "blue red green sphere",    # Multiple colors
    ]
    
    for phrase in ambiguous_examples:
        print(f"\n🔀 Analyzing: \"{phrase}\"")
        print("-" * 30)
        
        result = executor.execute_layer2(phrase, enable_semantic_grounding=False)
        
        if result.success:
            print(f"Generated {len(result.hypotheses)} alternative interpretation(s):")
            
            for i, hyp in enumerate(result.hypotheses, 1):
                token_words = [t.word for t in hyp.tokens]
                print(f"  {i}. {token_words} (confidence: {hyp.confidence:.3f})")
                
                # Show the interpretation
                if hyp.np_replacements:
                    for j, np_replacement in enumerate(hyp.np_replacements, 1):
                        print(f"     → NP {j}: {np_replacement}")
        else:
            print("❌ Processing failed")


def show_vocabulary_constraints():
    """Show the vocabulary words used in this demo."""
    print("\n📚 Vocabulary Constraints")
    print("=" * 28)
    print("This demo only uses words from the semantic vector space:")
    print()
    
    # Show relevant vocabulary categories
    vocab_words = list(SEMANTIC_VECTOR_SPACE.keys())
    
    # Categorize the words
    nouns = [w for w in vocab_words if any(cat in SEMANTIC_VECTOR_SPACE[w] for cat in ['shape', 'object'])]
    adjectives = [w for w in vocab_words if any(cat in SEMANTIC_VECTOR_SPACE[w] for cat in ['color', 'size', 'texture'])]
    determiners = [w for w in vocab_words if 'determiner' in SEMANTIC_VECTOR_SPACE.get(w, {})]
    
    print(f"Nouns ({len(nouns)}): {', '.join(sorted(nouns))}")
    print(f"Adjectives ({len(adjectives)}): {', '.join(sorted(adjectives))}")
    if determiners:
        print(f"Determiners ({len(determiners)}): {', '.join(sorted(determiners))}")
    
    print(f"\nTotal vocabulary: {len(vocab_words)} words")


def main():
    """Run all Layer 2 tokenization demonstrations."""
    # Suppress debug output for clean demo
    set_debug(False)
    
    print("LATN Layer 2 Tokenization Demo")
    print("==============================")
    print()
    print("This demo shows how Layer 2 of the Layered Augmented Transition Network")
    print("processes individual tokens from Layer 1 to form grammatical noun phrases.")
    print()
    
    # Show vocabulary constraints
    show_vocabulary_constraints()
    
    # Run the demonstrations
    demo_basic_noun_phrase_formation()
    demo_adjective_ordering()
    demo_determiner_handling()
    demo_multi_hypothesis_noun_phrases()
    
    print("\n" + "=" * 60)
    print("Demo complete! Layer 2 successfully demonstrated:")
    print("✓ Basic noun phrase formation")
    print("✓ Adjective ordering and agreement")
    print("✓ Determiner handling and number agreement")
    print("✓ Multi-hypothesis parsing for ambiguous cases")
    print("✓ Clean output with debug suppression")
    print("✓ Vocabulary-constrained examples")


if __name__ == "__main__":
    main()
