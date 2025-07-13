from engraf.pos.conjunction_phrase import ConjunctionPhrase
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vocabulary import vector_from_word



def test_subject_conjunction_phrase():
    sentence = SentencePhrase()
    one = NounPhrase("one")
    sentence.apply_subject(one)
    assert isinstance (sentence.subject, NounPhrase)

    tok = vector_from_word("and")
    sentence.apply_subject_conjunction(tok)
    assert isinstance (sentence.subject, ConjunctionPhrase)
    print(f"==> Sentence Subject = {sentence.subject}")
    assert sentence.subject.left == one
    assert sentence.subject.conjunction == "and"
    assert sentence.subject.right is None
    two = NounPhrase("two")
    sentence.apply_subject(two)
    print(f"==> Sentence Subject = {sentence.subject}")
    assert sentence.subject.left == one
    assert isinstance(sentence.subject.right, NounPhrase)
    
    tok = vector_from_word("and")
    sentence.apply_subject_conjunction(tok)
    assert sentence.subject.left == one
    assert sentence.subject.conjunction == "and"
    assert isinstance(sentence.subject.right, ConjunctionPhrase)
    assert sentence.subject.right.left == two
    assert sentence.subject.right.conjunction == "and"
    assert sentence.subject.right.right is None
    three = NounPhrase("three")
    sentence.apply_subject(three)
    assert sentence.subject.right.right == three

    tok = vector_from_word("and")
    sentence.apply_subject_conjunction(tok)
    assert isinstance(sentence.subject.right.right, ConjunctionPhrase)
    assert sentence.subject.right.right.left == three
    assert sentence.subject.right.right.conjunction == "and"
    assert sentence.subject.right.right.right is None
    four = NounPhrase("four")
    sentence.apply_subject(four)
    assert sentence.subject.right.right.right == four

def test_predicate_conjunction_phrase():
    sentence = SentencePhrase()
    one = VerbPhrase("one")
    sentence.apply_predicate(one)
    assert isinstance (sentence.predicate, VerbPhrase)

    tok = vector_from_word("and")
    sentence.apply_predicate_conjunction(tok)
    assert isinstance (sentence.predicate, ConjunctionPhrase)
    print(f"==> SentencePredicate = {sentence.predicate}")
    assert sentence.predicate.left == one
    assert sentence.predicate.conjunction == "and"
    assert sentence.predicate.right is None
    two = VerbPhrase("two")
    sentence.apply_predicate(two)
    print(f"==> SentencePredicate = {sentence.predicate}")
    assert sentence.predicate.left == one
    assert isinstance(sentence.predicate.right, VerbPhrase)
    
    tok = vector_from_word("and")
    sentence.apply_predicate_conjunction(tok)
    assert sentence.predicate.left == one
    assert sentence.predicate.conjunction == "and"
    assert isinstance(sentence.predicate.right, ConjunctionPhrase)
    assert sentence.predicate.right.left == two
    assert sentence.predicate.right.conjunction == "and"
    assert sentence.predicate.right.right is None
    three = VerbPhrase("three")
    sentence.apply_predicate(three)
    assert sentence.predicate.right.right == three

    tok = vector_from_word("and")
    sentence.apply_predicate_conjunction(tok)
    assert isinstance(sentence.predicate.right.right, ConjunctionPhrase)
    assert sentence.predicate.right.right.left == three
    assert sentence.predicate.right.right.conjunction == "and"
    assert sentence.predicate.right.right.right is None
    four = VerbPhrase("four")
    sentence.apply_predicate(four)
    assert sentence.predicate.right.right.right == four
