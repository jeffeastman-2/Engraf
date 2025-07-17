import pytest
from engraf.interpreter.sentence_interpreter import SentenceInterpreter

def test_sentence_interpreter_init():
    """Test basic initialization."""
    interpreter = SentenceInterpreter()
    assert interpreter.scene is not None
    assert interpreter.renderer is not None
    assert interpreter.object_counter == 0

def test_simple_sentence():
    """Test simple sentence parsing."""
    interpreter = SentenceInterpreter()
    result = interpreter.interpret("draw a cube")
    assert result['success'] == True
    assert len(result['objects_created']) == 1
    assert 'cube' in result['objects_created'][0]
