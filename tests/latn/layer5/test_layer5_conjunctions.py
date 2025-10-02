
import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase

class TestLayer5Conjunctions:
    # Test cases for Layer 5 conjunctions from GPT-5

    def setup_method(self):
        """Set up test environment."""
        self.executor = LATNLayerExecutor()
    

# A) NP-level coordination (Layer-2)
    # 1. Basic NP∧NP as subject
    def test_basic_np_and_np_as_subject(self):
        sentence = "the cube and the sphere move"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >=1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [N cube]] ∧ [NP [Det the] [N sphere]] [VP [V move]]]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be a VerbPhrase"

    # 2. Adj coordination inside NP (not NP∧NP)
    def test_adj_coordination_inside_np(self):
        sentence = "the red and blue cube"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [Adj red ∧ blue] [N cube]]]
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase(Adj,Adj)"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

    # 3. Three-way NP list (Oxford comma allowed)
    def test_three_way_np_list(self):
        sentence = "the cube, the sphere and the cone"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >= 1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [N cube]] ∧ [NP [Det the] [N sphere]] ∧ [NP [Det the] [N cone]]]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.phrases]
        assert len(parts) == 3, "Should have three coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

    # 4. Ambiguous nominal (keep both parses)
    def test_ambiguous_nominal(self):
        sentence = "the red and blue boxes and spheres"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >= 1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # A: [NP ( [NP red∧blue boxes] ∧ [NP spheres] )]
        # B: [NP ( [NP red] ∧ [NP blue boxes ∧ spheres] )]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.phrases]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"


    # 5. NP with numeric determiners
    def test_np_with_numeric_determiners(self):
        sentence = "two cubes and three spheres"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >= 1, "Should extract one hypothesis"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        #[NP ( [NP two cubes] ∧ [NP three spheres] )]
        subj = sent.subject
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.phrases]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

# B) PP-level (Layer-3)
    # 6. PP attaches to whole NP coordination
    def test_pp_attaches_to_whole_np_coordination(self):
        sentence = "the cube and the sphere on the table"
        result = self.executor.execute_layer5(sentence,report=True)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >= 1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        np = vector.phrase
        assert isinstance(np, ConjunctionPhrase), "First hypothesis should be a ConjunctionPhrase"
        # Parses (keep both):
        # A (wide): [NP ( [NP the cube] ∧ [NP the sphere] ) [PP on [NP the table]]]
        # B (narrow): [NP [NP the cube] ∧ [NP the sphere [PP on [NP the table]]]]
        assert np.vector.isa("plural"), "Subject number should be plural"
        assert np.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in np.phrases]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pp = hypo[1]._original_pp
        assert isinstance(pp, PrepositionalPhrase), "Second hypothesis should be a PrepositionalPhrase"
        assert pp.vector.isa("prep"), "PP should be a preposition"   

    # 7. PP∧PP predicate complement
    def test_pp_and_pp_predicate_complement(self):
        sentence = "the sphere is above the cube and behind the box"
        result = self.executor.execute_layer5(sentence,report=True)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >= 1, "Should extract two hypotheses"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the sphere] [VP [V is] [PP ( [PP above [NP the cube]] ∧ [PP in front of [NP the box]] ) ]]]
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"
        preps = pred.prepositions
        assert len(preps) == 1, "Should have one coordinated prepositional phrases"
        coord = preps[0]
        assert isinstance(coord, ConjunctionPhrase), "Coordinated PP should be ConjunctionPhrase"
        parts = [pp for pp in coord.phrases]
        assert len(parts) == 2, "Should have two coordinated PPs"

    # 8. Left/Right relation
    def test_left_right_relation(self):
        sentence = "the cube is left of the sphere and right of the cone"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >=1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP [V is] [PP ( [PP left of [NP the sphere]] ∧ [PP right of [NP the cone]] ) ]]]
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"
        preps = pred.prepositions
        assert len(preps) == 1, "Should have one coordinated prepositional phrases"
        coord = preps[0]
        assert isinstance(coord, ConjunctionPhrase), "Coordinated PP should be ConjunctionPhrase"
        parts = [pp for pp in coord.phrases]
        assert len(parts) == 2, "Should have two coordinated PPs"

# C) VP-level coordination (Layer-4)
    # 9. VP∧VP 
    def test_vp_and_vp_under_same_subject(self):
        sentence = "move the sphere to [1,2,3] and rotate the cube by 45 degrees"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >=1, "Should extract three hypotheses"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sp = vector.phrase
        assert isinstance(sp, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sp.subject
        assert subj is None
        pred = sp.predicate
        assert isinstance(pred, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"
        parts = [vp for vp in pred.phrases]
        assert len(parts) == 2, "Should have two VP parts"

    # 10. VP∧VP with shared object (right-node raising)
    def test_vp_and_vp_with_shared_object(self):
        sentence = "color the cube and move the sphere"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sp = vector.phrase
        assert isinstance(sp, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sp.subject
        assert subj is None
        pred = sp.predicate
        assert isinstance(pred, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"
        parts = [vp for vp in pred.phrases]
        assert len(parts) == 2, "Should have two VP parts"
    
    # 11. Mixed: VP∧VP and NP∧NP as object
    def test_vp_and_np_as_object(self):
        sentence = "the cube color the sphere and the box and move the sphere and the box"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) >0 , "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(VP,VP)"
        assert pred.vector.isa("plural"), "Predicate should be plural"
        parts = [vp for vp in pred.phrases]
        assert len(parts) == 2, "Should have two VP parts"

    # 12. Adverb inside VP conj
    def test_vp_with_adverb(self):
        sentence = "rotate and slightly move the cube"
        result = self.executor.execute_layer5(sentence)
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sp = vector.phrase
        assert isinstance(sp, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sp.subject
        assert subj is None
        pred = sp.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"
        assert isinstance(pred.verb, str), "Predicate verb should be a string"
        assert "slightly" in pred.verb, "Predicate verb should contain 'slightly'"

# D) S-level coordination (Layer-5)
    # 13. S∧S with comma
    def test_s_and_s_with_comma(self):
        sentence = "move the cube, rotate the sphere"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        pred = sent.predicate
        assert isinstance(pred, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"
        parts = [vp for vp in pred.phrases]
        assert len(parts) == 2, "Should have two VP parts"

    # 14. Correlative “both…and” gives plural subject
    def test_both_and(self):
        sentence = "both the cube and the sphere are smooth"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.isa("unknown")
        vector = hypo[1]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"

    # 15. Disjunction “or” (document your number policy)
    def test_or_clause(self):
        sentence = "the cube or the spheres are near the table"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"

    # E) Preposition + movement verbs
    # 16. Move with directional PP
    def test_move_with_directional_pp(self):
        sentence = "move the cube to the table"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert subj is None
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"

    # 17. Rotate around axis (axis nouns provided)
    def test_rotate_around_axis(self):
        sentence = "the cube rotate around the x-axis"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"

    # 18. X/Y/Z rotate verbs (semantics in vectors)
    def test_test_x_y_z_rotate(self):
        sentence = "the sphere xrotate and yrotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"
        assert pred.verb == "xrotate and yrotate", "Predicate verb should be 'rotate'"

    # F) Pronouns & agreement
    # 19. Pronoun plural subject + VP
    def test_pronoun_plural_subject(self):
        sentence = "they move"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be VerbPhrase"
        assert subj.vector.isa("plural"), "Subject number should be plural" 

    # 20.Pronoun object reused in VP∧VP
    def test_pronoun_object_reused(self):
        sentence = "the cube color them and move them"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector.phrase
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase"
        pred = sent.predicate
        assert isinstance(pred, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"
        assert subj.vector.isa("def"), "Subject number should be singular"
        assert pred.vector.isa("plural"), "Predicate number should be plural"

    # G) Edge / negative (greediness guard)
    # 21. Avoid bogus NP∧NP when lookahead isn’t NP-start
    def test_avoid_bogus_np_coordination(self):
        sentence = "the cube and slightly rotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        np = vector.phrase
        assert isinstance(np, NounPhrase), "First hypothesis should be a NounPhrase"
        vector = hypo[1]
        assert vector.isa("conj"), "Second hypothesis should be a conjunction"
        vector = hypo[2]
        assert vector.isa("SP"), "Third hypothesis should be a sentence phrase"

    # 22. PP scope ambiguity retained
    def test_pp_scope_ambiguity(self):
        sentence = "the cube and the sphere left of the box"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        np = vector.phrase
        assert isinstance(np, ConjunctionPhrase), "First hypothesis should be a ConjunctionPhrase"
        vector = hypo[1]
        pp = vector.phrase
        assert isinstance(pp, PrepositionalPhrase), "Second hypothesis should be a PrepositionalPhrase" 
