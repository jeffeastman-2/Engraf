from engraf.lexer.token_stream import tokenize  

def test_tokenize_vector_literal():
    input_text = "draw a tall blue cube at [3.0, 4.0, 5.0]"
    tokens = tokenize(input_text)

    expected = [
        "draw", "a", "tall", "blue", "cube", "at",
        {"type": "VECTOR", "value": [3.0, 4.0, 5.0]}
    ]

    assert tokens == expected, f"Expected {expected}, got {tokens}"
