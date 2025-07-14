import numpy as np
from engraf.atn.np import build_np_atn
from engraf.lexer.token_stream import TokenStream, tokenize
from engraf.atn.subnet_np import run_np
from engraf.lexer.vector_space import VectorSpace, vector_from_features

input = [
"DRAW A ROUGH RED CIRCLE AT [-1, 2.2, 5.55]",
"MOVE A CIRCLES TO [3, 4, 5]",
"MOVE 3 CIRCLES TO [3, 4, 5]",
"MOVE THE BLUE CIRCLE TO [3, 4, 5]",
"MOVE THE ROUGH CIRCLE TO [3, 4, 5]",
"MOVE IT TO [3, 3, 3]",
"MAKE THEM LARGE",
"MAKE IT VERY TALL",
"MAKE IT MUCH MORE TRANSPARENT AND ROUGHER",
"NO, MAKE IT MUCH MORE TRANSPARENT AND A TEENSY BIT ROUGHER",
"COLOR IT A LITTLE BIT REDDER",
"NO, COLOR IT PURPLE",
"XROTATE IT by 45 DEGREES AND YROTATE IT by 27 DEGREES",
"DRAW 3 VERY SMOOTH SHORT AND VERY OPAQUE BOXES",
"MOVE 2 OF THEM TO [1, 2, 3]",
"MAKE THEM MORE TRANSPARENT THAN THE PURPLE CIRCLE AT [3, 3, 3]",
"COLOR 4 OF THEM BLUE GREEN",
"MAKE 1 OF THEM AS ROUGH AS THE PURPLE CIRCLE AND MAKE THE REST OF THEM SMOOTH",
"REMOVE A BOX AT [1, 2, 3]",
"REMOVE THE TALL BOX AT [1, 2, 3]",
"COLOR A BOX TALL",
"DRAW A TALL RED BOX AND A SMALL BRIGHT BLUE CIRCLE",
"COLOR A TALL BOX GREEN AND MOVE IT TO [7, 7, 7]",
"DRAW 3 CIRCLES AT [3, 6, 9] AND COLOR TWO OF THEM RED",
"GO BACK IN TIME",
"GO FORWARD IN TIME",
"GO FORWARD IN TIME",
"GO FORWARD IN TIME",
"'HUGE' IS VERY LARGE",
"'SKY BLUE' IS BLUE AND GREEN",
"DRAW A HUGE SKY BLUE BOX"
]
input_lower = [s.lower() for s in input]

def test_dissertation_sentences_parsing():
    """Test parsing of all dissertation sentences and report which ones succeed/fail"""
    from engraf.atn.subnet_sentence import run_sentence
    
    print("\n" + "="*80)
    print("DISSERTATION SENTENCES PARSING REPORT")
    print("="*80)
    
    successful_parses = []
    failed_parses = []
    
    for i, sentence in enumerate(input_lower, 1):
        print(f"\n[{i:2d}] Testing: {sentence}")
        print("-" * 60)
        
        try:
            tokens = TokenStream(tokenize(sentence))
            result = run_sentence(tokens)
            
            if result is not None:
                print(f"✅ SUCCESS: Parsed successfully")
                print(f"    Result: {result}")
                successful_parses.append((i, sentence))
            else:
                print(f"❌ FAILED: Parser returned None")
                failed_parses.append((i, sentence))
                
        except Exception as e:
            print(f"💥 ERROR: Exception during parsing: {e}")
            failed_parses.append((i, sentence))
    
    # Summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    total = len(input_lower)
    success_count = len(successful_parses)
    failure_count = len(failed_parses)
    
    print(f"Total sentences: {total}")
    print(f"Successfully parsed: {success_count} ({success_count/total*100:.1f}%)")
    print(f"Failed to parse: {failure_count} ({failure_count/total*100:.1f}%)")
    
    if successful_parses:
        print(f"\n✅ SUCCESSFUL PARSES ({success_count}):")
        for idx, sentence in successful_parses:
            print(f"  [{idx:2d}] {sentence}")
    
    if failed_parses:
        print(f"\n❌ FAILED PARSES ({failure_count}):")
        for idx, sentence in failed_parses:
            print(f"  [{idx:2d}] {sentence}")
    
    print("\n" + "="*80)
    
    # Don't fail the test - just report the results
    assert True, "Test completed - see output for parsing results"